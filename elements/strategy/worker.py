""" Стратегии отбора акций на 1-2 дня.

model          — логистическая регрессия по признакам, обучается walk-forward
                 только на сделках, закрытых до даты прогноза (без заглядывания вперёд).
mean_reversion — бенчмарк: откат (RSI(2) < 10) в восходящем тренде.
momentum       — бенчмарк: сильный 5-дневный рост на повышенном обороте у максимумов.
breakout       — пробой 20-дневного максимума на объёме с закрытием у хая дня.
trend_pullback — откат 2-3 дня внутри сильного тренда (выше SMA50, рост за 20 дней).
squeeze        — сжатие волатильности (ATR ниже среднего) и закрытие у хая: ждём расширения.
capitulation   — отскок после паники: падение >3% на двойном обороте с выкупом от минимума.
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


def score_mean_reversion(panel: pd.DataFrame) -> pd.DataFrame:
    scored = panel.copy()
    scored['SCORE'] = -scored['rsi2']
    scored['PROBABILITY'] = np.nan
    scored['EXPECTED_RETURN'] = np.nan
    scored['CANDIDATE'] = (
        tradable(scored) & (scored['rsi2'] < 10) & (scored['dist_sma200'] > 0)
    )
    return scored


def score_momentum(panel: pd.DataFrame) -> pd.DataFrame:
    scored = panel.copy()
    scored['SCORE'] = scored['r5_cs']
    scored['PROBABILITY'] = np.nan
    scored['EXPECTED_RETURN'] = np.nan
    scored['CANDIDATE'] = (
        tradable(scored) & regime_ok(scored)
        & (scored['vol_ratio'] > 1.5) & (scored['dist_high20'] > -0.01) & (scored['r5_cs'] > 0)
    )
    return scored


def rule_scored(panel: pd.DataFrame, score: pd.Series, candidate: pd.Series) -> pd.DataFrame:
    scored = panel.copy()
    scored['SCORE'] = score
    scored['PROBABILITY'] = np.nan
    scored['EXPECTED_RETURN'] = np.nan
    scored['CANDIDATE'] = (tradable(scored) & candidate).fillna(False)
    return scored


def score_breakout(panel: pd.DataFrame) -> pd.DataFrame:
    return rule_scored(
        panel,
        score=panel['vol_ratio'],
        candidate=regime_ok(panel)
        & (panel['dist_high20'] >= -0.001) & (panel['vol_ratio'] > 1.3) & (panel['clv'] > 0.7),
    )


def score_trend_pullback(panel: pd.DataFrame) -> pd.DataFrame:
    return rule_scored(
        panel,
        score=panel['r20'] - panel['r3'],
        candidate=(panel['dist_sma50'] > 0.03) & (panel['r20'] > 0.05)
        & (panel['r3'] < -0.02) & (panel['rsi2'] < 25),
    )


def score_squeeze(panel: pd.DataFrame) -> pd.DataFrame:
    return rule_scored(
        panel,
        score=-panel['atr_ratio'],
        candidate=(panel['atr_ratio'] < 0.8) & (panel['range_pct'] < 0.7 * panel['atr_pct'])
        & (panel['clv'] > 0.6) & (panel['dist_sma20'] > 0),
    )


def score_capitulation(panel: pd.DataFrame) -> pd.DataFrame:
    return rule_scored(
        panel,
        score=panel['vol_ratio'] * -panel['r1'],
        candidate=(panel['r1'] < -0.03) & (panel['vol_ratio'] > 2)
        & (panel['clv'] > 0.5) & (panel['dist_sma200'] > -0.2),
    )


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
}

STRATEGY_TITLES = {
    'model': 'Модель (логистическая регрессия)',
    'mean_reversion': 'Откат RSI(2) в тренде',
    'momentum': 'Импульс 5 дней',
    'breakout': 'Пробой 20-дневного максимума',
    'trend_pullback': 'Откат в сильном тренде',
    'squeeze': 'Сжатие волатильности',
    'capitulation': 'Отскок после паники',
}
