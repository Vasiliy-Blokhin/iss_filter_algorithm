""" Стратегии отбора акций на 1-2 дня.

model          — логистическая регрессия по признакам, обучается walk-forward
                 только на сделках, закрытых до даты прогноза (без заглядывания вперёд).
mean_reversion — бенчмарк: откат (RSI(2) < 10) в восходящем тренде с выкупом внутри дня.
momentum       — бенчмарк: сильный 5-дневный рост на повышенном обороте у максимумов.
breakout       — пробой 20-дневного максимума на объёме с закрытием у хая дня.
trend_pullback — глубокий откат за 3 дня внутри сильного тренда (выше SMA50, рост за 20 дней).
squeeze        — сжатие волатильности (ATR ниже среднего) и закрытие у хая в день роста рынка.
capitulation   — отскок после паники: падение >4% на повышенном обороте с выкупом от минимума
                 (только если сам IMOEX за день упал не больше чем на 2%).

Пороги по умолчанию подобраны 13.10.2026 на истории 2020-2024 (см. README); действующие
пороги хранятся в БД и периодически обновляются самокоррекцией (elements/selftune).
"""
import numpy as np
import pandas as pd

from elements.features.worker import FEATURES
from source.settings.settings import (
    MIN_TURNOVER,
    MIN_PRICE,
    MIN_PROBABILITY,
    MIN_EDGE,
    MAX_SIGNALS,
    MAX_PER_SECTOR,
    TRAIN_WINDOW,
    RETRAIN_EVERY,
    L2_REG,
    REGIME_FILTER,
    HOLD_DAYS,
)


class LogisticModel:
    """ L2-логистическая регрессия (Ньютон) + калибровка ожидаемой доходности по децилям."""

    def __init__(self, l2: float = L2_REG, iterations: int = 25):
        self.l2 = l2
        self.iterations = iterations

    def fit(self, x: np.ndarray, y: np.ndarray, returns: np.ndarray):
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0) + 1e-9
        design = self.design(x)
        weights = np.zeros(design.shape[1])
        penalty = self.l2 * np.eye(design.shape[1])
        penalty[0, 0] = 0
        for _ in range(self.iterations):
            prob = 1 / (1 + np.exp(-design @ weights))
            gradient = design.T @ (prob - y) + penalty @ weights
            hessian = (design * (prob * (1 - prob))[:, None]).T @ design + penalty
            step = np.linalg.solve(hessian, gradient)
            weights -= step
            if np.abs(step).max() < 1e-6:
                break
        self.weights = weights

        # Средняя фактическая доходность сделок по децилям прогноза.
        prob = self.predict_proba(x)
        self.edges = np.unique(np.quantile(prob, np.linspace(0, 1, 11)))
        buckets = np.clip(np.searchsorted(self.edges, prob, side='right') - 1, 0, len(self.edges) - 2)
        self.bucket_returns = np.array([
            returns[buckets == b].mean() if (buckets == b).any() else 0.0
            for b in range(len(self.edges) - 1)
        ])
        return self

    def design(self, x: np.ndarray) -> np.ndarray:
        z = np.nan_to_num((x - self.mean) / self.std)
        z = np.clip(z, -5, 5)
        return np.hstack([np.ones((len(z), 1)), z])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-self.design(x) @ self.weights))

    def expected_return(self, prob: np.ndarray) -> np.ndarray:
        buckets = np.clip(
            np.searchsorted(self.edges, prob, side='right') - 1, 0, len(self.bucket_returns) - 1
        )
        return self.bucket_returns[buckets]

    def importance(self) -> dict:
        return dict(sorted(
            zip(FEATURES, self.weights[1:]), key=lambda item: -abs(item[1])
        ))


def tradable(panel: pd.DataFrame) -> pd.Series:
    return (
        (panel['TURNOVER20'] >= MIN_TURNOVER)
        & (panel['CLOSE'] >= MIN_PRICE)
        & panel['ATR'].notna()
        & panel['dist_sma50'].notna()
    )


def regime_ok(panel: pd.DataFrame) -> pd.Series:
    """ Рынок не в нисходящем режиме: IMOEX выше SMA50 или не падал за 5 дней."""
    if not REGIME_FILTER:
        return pd.Series(True, index=panel.index)
    return (panel['idx_dist_sma50'].fillna(0) > 0) | (panel['idx_r5'].fillna(0) > 0)


def train_model(history: pd.DataFrame, as_of: pd.Timestamp) -> LogisticModel | None:
    """ Обучение на сделках, закрытых строго до as_of."""
    train = history[
        (history['EXIT_DATE'] < as_of) & history['TRADE_RET'].notna() & tradable(history)
    ]
    dates = np.sort(train['DATE'].unique())[-TRAIN_WINDOW:]
    train = train[train['DATE'].isin(dates)]
    if len(dates) < 120 or len(train) < 2000:
        return None
    returns = train['TRADE_RET'].to_numpy()
    return LogisticModel().fit(
        train[FEATURES].to_numpy(dtype=float), (returns > 0).astype(float), returns
    )


def score_model(panel: pd.DataFrame, model: LogisticModel) -> pd.DataFrame:
    scored = panel.copy()
    scored['PROBABILITY'] = model.predict_proba(scored[FEATURES].to_numpy(dtype=float))
    scored['EXPECTED_RETURN'] = model.expected_return(scored['PROBABILITY'].to_numpy())
    scored['SCORE'] = scored['EXPECTED_RETURN']
    scored['CANDIDATE'] = (
        tradable(scored) & regime_ok(scored)
        & (scored['PROBABILITY'] >= MIN_PROBABILITY)
        & (scored['EXPECTED_RETURN'] >= MIN_EDGE)
    )
    return scored


# Пороги правиловых стратегий. Значения по умолчанию подобраны 13.10.2026 (см. README);
# действующие значения хранятся в БД (таблица algo_params) и подставляются
# через set_active_params() — их обновляет самокоррекция (elements/selftune).
# Порог <= OFF_LOW (или >= OFF_HIGH для условий «меньше») выключает условие, clv = -1 — тоже.
OFF_LOW, OFF_HIGH = -9.0, 9.0
DEFAULT_PARAMS = {
    'mean_reversion': {'rsi': 10, 'sma200': -0.1, 'r20': 0.0, 'clv': 0.4, 'regime': 1},
    'momentum': {'vol': 2.5, 'high': -0.005, 'r5cs': 0.0, 'clv': 0.5, 'sma200': 0.0,
                 'gap': -0.01, 'regime': 0},
    'breakout': {'high': -0.001, 'vol': 3.0, 'clv': 0.7, 'sma50': 0.05, 'regime': 1},
    'trend_pullback': {'sma50': 0.03, 'r20': 0.08, 'r3': -0.045, 'rsi': 30, 'clv': -1.0,
                       'regime': 1},
    'squeeze': {'atr': 0.8, 'rng': 0.7, 'clv': 0.6, 'sma20': 0.0, 'idx': 0.0, 'regime': 0},
    'capitulation': {'r1': -0.04, 'vol': 1.3, 'idx': -0.02, 'clv': 0.3, 'sma200': -0.1,
                     'regime': 1},
}
ACTIVE_PARAMS = {name: dict(values) for name, values in DEFAULT_PARAMS.items()}


def set_active_params(params: dict) -> dict:
    """ Действующие пороги: значения из БД поверх значений по умолчанию."""
    for name, defaults in DEFAULT_PARAMS.items():
        ACTIVE_PARAMS[name] = {**defaults, **(params or {}).get(name, {})}
    return ACTIVE_PARAMS


def above(series: pd.Series, threshold: float) -> pd.Series:
    if threshold <= OFF_LOW:
        return pd.Series(True, index=series.index)
    return series > threshold


def below(series: pd.Series, threshold: float) -> pd.Series:
    if threshold >= OFF_HIGH:
        return pd.Series(True, index=series.index)
    return series < threshold


def rule_condition(name: str, panel: pd.DataFrame, p: dict) -> tuple[pd.Series, pd.Series]:
    """ (условие входа без фильтра ликвидности, приоритет) правиловой стратегии."""
    idx_r1 = panel['idx_r1'].fillna(0)
    if name == 'mean_reversion':
        # Без выкупа внутри дня (clv) и роста за 20 дней «падающие ножи» давали -11.7% за год.
        condition = (
            (panel['rsi2'] < p['rsi']) & above(panel['dist_sma200'], p['sma200'])
            & above(panel['r20'], p['r20']) & above(panel['clv'], p['clv'])
        )
        score = -panel['rsi2']
    elif name == 'momentum':
        condition = (
            (panel['vol_ratio'] > p['vol']) & (panel['dist_high20'] > p['high'])
            & (panel['r5_cs'] > p['r5cs']) & above(panel['clv'], p['clv'])
            & above(panel['dist_sma200'], p['sma200']) & above(panel['gap'], p['gap'])
        )
        score = panel['r5_cs']
    elif name == 'breakout':
        condition = (
            (panel['dist_high20'] >= p['high']) & (panel['vol_ratio'] > p['vol'])
            & above(panel['clv'], p['clv']) & above(panel['dist_sma50'], p['sma50'])
        )
        score = panel['vol_ratio']
    elif name == 'trend_pullback':
        # Только глубокий откат после сильного роста: мелкие откаты (-2%) давали -10% за год.
        condition = (
            above(panel['dist_sma50'], p['sma50']) & (panel['r20'] > p['r20'])
            & (panel['r3'] < p['r3']) & (panel['rsi2'] < p['rsi']) & above(panel['clv'], p['clv'])
        )
        score = panel['r20'] - panel['r3']
    elif name == 'squeeze':
        condition = (
            (panel['atr_ratio'] < p['atr']) & (panel['range_pct'] < p['rng'] * panel['atr_pct'])
            & above(panel['clv'], p['clv']) & above(panel['dist_sma20'], p['sma20'])
            & above(idx_r1, p['idx'])
        )
        score = -panel['atr_ratio']
    elif name == 'capitulation':
        # Падение самой бумаги, а не обвал всего рынка: в дни падения IMOEX > 2% отскок не работал.
        condition = (
            (panel['r1'] < p['r1']) & (panel['vol_ratio'] > p['vol']) & above(idx_r1, p['idx'])
            & above(panel['clv'], p['clv']) & above(panel['dist_sma200'], p['sma200'])
        )
        score = panel['vol_ratio'] * -panel['r1']
    else:
        raise KeyError(name)
    if p.get('regime'):
        condition = condition & regime_ok(panel)
    if not p.get('enabled', 1):
        # Самокоррекция выключила стратегию: на окне обучения она убыточна при любых порогах.
        condition = pd.Series(False, index=panel.index)
    return condition, score


def rule_scored(panel: pd.DataFrame, score: pd.Series, candidate: pd.Series) -> pd.DataFrame:
    scored = panel.copy()
    scored['SCORE'] = score
    scored['PROBABILITY'] = np.nan
    scored['EXPECTED_RETURN'] = np.nan
    scored['CANDIDATE'] = (tradable(scored) & candidate).fillna(False)
    return scored


def rule_scorer(name: str):
    """ Функция оценки стратегии с действующими (или явно переданными) порогами."""
    def scorer(panel: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        condition, score = rule_condition(name, panel, params or ACTIVE_PARAMS[name])
        return rule_scored(panel, score, condition)
    scorer.__name__ = f'score_{name}'
    return scorer


score_mean_reversion = rule_scorer('mean_reversion')
score_momentum = rule_scorer('momentum')
score_breakout = rule_scorer('breakout')
score_trend_pullback = rule_scorer('trend_pullback')
score_squeeze = rule_scorer('squeeze')
score_capitulation = rule_scorer('capitulation')


def walk_forward_scores(panel: pd.DataFrame, start: pd.Timestamp | None = None) -> pd.DataFrame:
    """ Прогнозы модели для каждого дня с переобучением раз в RETRAIN_EVERY дней."""
    dates = np.sort(panel['DATE'].unique())
    if start is not None:
        dates = dates[dates >= np.datetime64(start)]
    result = []
    model = None
    for number, current in enumerate(dates):
        if model is None or number % RETRAIN_EVERY == 0:
            model = train_model(panel, pd.Timestamp(current)) or model
        if model is None:
            continue
        result.append(score_model(panel[panel['DATE'] == current], model))
    return pd.concat(result, ignore_index=True) if result else pd.DataFrame()


def dividend_safe(panel: pd.DataFrame) -> pd.Series:
    """ Отсечка дивидендов не попадает на срок удержания (иначе цена упадёт на размер выплаты)."""
    if 'div_days' not in panel:
        return pd.Series(True, index=panel.index)
    return panel['div_days'].isna() | (panel['div_days'] > HOLD_DAYS)


def select(scored_day: pd.DataFrame, sectors: dict, limit: int = MAX_SIGNALS) -> pd.DataFrame:
    """ Топ кандидатов дня с ограничением на число бумаг одного сектора."""
    candidates = scored_day[scored_day['CANDIDATE'] & dividend_safe(scored_day)]
    candidates = candidates.sort_values('SCORE', ascending=False)
    chosen, per_sector = [], {}
    for index, row in candidates.iterrows():
        sector = sectors.get(row['SECID'], 'unknown')
        if sector != 'unknown' and per_sector.get(sector, 0) >= MAX_PER_SECTOR:
            continue
        per_sector[sector] = per_sector.get(sector, 0) + 1
        chosen.append(index)
        if len(chosen) >= limit:
            break
    return candidates.loc[chosen]


# Правиловые стратегии; 'model' считается отдельно (walk_forward_scores).
STRATEGIES = {
    'mean_reversion': score_mean_reversion,
    'momentum': score_momentum,
    'breakout': score_breakout,
    'trend_pullback': score_trend_pullback,
    'squeeze': score_squeeze,
    'capitulation': score_capitulation,
}
ALL_STRATEGIES = ['model', *STRATEGIES]

# Для ИИ-проверки: что именно стратегия считает поводом для покупки.
STRATEGY_LOGIC = {
    'model': 'статистическая модель по признакам цены и оборота',
    'mean_reversion': 'покупка отката (RSI(2) < 10) в долгосрочном росте — низкий RSI ожидаем',
    'momentum': 'покупка сильного роста на обороте у 20-дневных максимумов — высокий RSI ожидаем',
    'breakout': 'покупка пробоя 20-дневного максимума на объёме — цена у максимумов ожидаема',
    'trend_pullback': 'покупка короткого отката внутри сильного тренда',
    'squeeze': 'покупка после сжатия волатильности с закрытием у хая дня',
    'capitulation': 'покупка отскока после панического падения на большом обороте',
    'reverse': 'реверсивный алгоритм: правила, выведенные из опыта прошлых сделок и новостного фона',
}

STRATEGY_TITLES = {
    'model': 'Модель (логистическая регрессия)',
    'mean_reversion': 'Откат RSI(2) в тренде',
    'momentum': 'Импульс 5 дней',
    'breakout': 'Пробой 20-дневного максимума',
    'trend_pullback': 'Откат в сильном тренде',
    'squeeze': 'Сжатие волатильности',
    'capitulation': 'Отскок после паники',
    'reverse': 'Реверсивный (опыт + новости)',
}
