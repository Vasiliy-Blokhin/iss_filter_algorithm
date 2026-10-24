""" Режим рынка: сводная оценка по IMOEX и ширине рынка (тренд, пересечение средних,
импульс 3 мес., просадка от годового максимума, перцентиль волатильности, доля бумаг выше SMA50),
аварийный кризис по просадке/панике и подтверждение смены режима (гистерезис).
Всё причинно: значение дня зависит только от прошлых и текущих данных."""
import numpy as np
import pandas as pd

# Константы
REGIMES = [
    {'key': 'crisis',  'title': 'Кризис',          'emoji': '🔴', 'upper': -0.60, 'eq_cap': 0.0},
    {'key': 'bear',    'title': 'Спад',            'emoji': '🟠', 'upper': -0.20, 'eq_cap': 0.3},
    {'key': 'neutral', 'title': 'Нейтральный',     'emoji': '⚪', 'upper': 0.15,  'eq_cap': 0.5},
    {'key': 'bull',    'title': 'Рост',            'emoji': '🟢', 'upper': 0.50,  'eq_cap': 0.7},
    {'key': 'boom',    'title': 'Сильный рост',    'emoji': '💚', 'upper': 9.0,   'eq_cap': 0.9},
]
REGIME_KEYS = [r['key'] for r in REGIMES]
WEIGHTS = {'trend': 0.20, 'cross': 0.15, 'momentum': 0.20, 'drawdown': 0.20, 'volatility': 0.10, 'breadth': 0.15}
CONFIRM_DAYS = 3
CRISIS_DRAWDOWN = -0.20
PANIC_R5 = -0.07
PANIC_VOL_PCT = 0.95
CRISIS_R63 = -0.10
CRISIS_VOL_PCT = 0.80

def regime_info(key: str) -> dict:
    """Возвращает словарь с информацией о режиме."""
    for r in REGIMES:
        if r['key'] == key:
            return r
    raise KeyError(f"Unknown regime key: {key}")

def classify(score: float) -> str:
    """Определяет режим по значению score."""
    if pd.isna(score):
        return 'neutral'
    for r in REGIMES:
        if score < r['upper']:
            return r['key']
    return REGIMES[-1]['key']

def indicators(index: pd.DataFrame, breadth: pd.Series | None = None) -> pd.DataFrame:
    """Вычисляет индикаторы и общий score."""
    df = index.copy()
    df = df.sort_values('DATE')
    df = df.set_index('DATE')
    close = df['CLOSE']

    # dist_sma200
    sma200 = close.rolling(200, min_periods=120).mean()
    df['dist_sma200'] = close / sma200 - 1

    # sma_cross
    sma50 = close.rolling(50, min_periods=30).mean()
    df['sma_cross'] = sma50 / sma200 - 1

    # r5, r63
    df['r5'] = close.pct_change(5)
    df['r63'] = close.pct_change(63)

    # drawdown
    max200 = close.rolling(252, min_periods=20).max()
    df['drawdown'] = close / max200 - 1

    # vol20
    log_ret = np.log(close / close.shift(1))
    df['vol20'] = log_ret.rolling(20, min_periods=10).std() * np.sqrt(252)

    # vol_pct
    def pct_rank(w):
        return (w <= w[-1]).mean()
    df['vol_pct'] = df['vol20'].rolling(756, min_periods=126).apply(pct_rank, raw=True)

    # breadth
    if breadth is not None:
        breadth = breadth.reindex(df.index).ffill()
    df['breadth'] = breadth

    # component scores
    df['trend'] = np.clip(df['dist_sma200'] / 0.10, -1, 1)
    df['cross'] = np.clip(df['sma_cross'] / 0.05, -1, 1)
    df['momentum'] = np.clip(df['r63'] / 0.15, -1, 1)
    df['drawdown_s'] = np.clip((df['drawdown'] + 0.10) / 0.10, -1, 1)
    df['volatility'] = np.clip(1 - 2 * df['vol_pct'], -1, 1)
    df['breadth_s'] = np.clip((df['breadth'] - 0.5) / 0.3, -1, 1)

    # сводная оценка: взвешенное среднее доступных составляющих
    columns = {'trend': 'trend', 'cross': 'cross', 'momentum': 'momentum', 'drawdown': 'drawdown_s',
               'volatility': 'volatility', 'breadth': 'breadth_s'}
    values = df[list(columns.values())].to_numpy(dtype=float)
    weights = np.array([WEIGHTS[name] for name in columns])
    present = ~np.isnan(values)
    total = (present * weights).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        df['score'] = np.where(total > 0, np.nansum(values * weights, axis=1) / total, np.nan)

    # panic
    # аварийный кризис: глубокая просадка с быстрым падением или всплеском волатильности,
    # либо паническая неделя при экстремальной волатильности (просто долгий спад — не кризис)
    df['panic'] = (((df['drawdown'] <= CRISIS_DRAWDOWN)
                    & ((df['r63'] <= CRISIS_R63) | (df['vol_pct'] >= CRISIS_VOL_PCT)))
                   | ((df['vol_pct'] >= PANIC_VOL_PCT) & (df['r5'] <= PANIC_R5)))

    # raw regime
    df['raw'] = df['score'].apply(classify)
    df.loc[df['panic'], 'raw'] = 'crisis'

    return df

def apply_hysteresis(raw: pd.Series, confirm: int = CONFIRM_DAYS) -> pd.Series:
    """ Подтверждённый режим: в кризис — сразу, иначе смена после confirm дней подряд
    одного и того же нового сырого режима; из кризиса — не выше чем на ступень (в спад)."""
    out = pd.Series(index=raw.index, dtype='object')
    if raw.empty:
        return out
    current, candidate, count = raw.iloc[0], None, 0
    for i, val in enumerate(raw.to_numpy()):
        if val == current:
            candidate, count = None, 0
        elif val == 'crisis':
            current, candidate, count = 'crisis', None, 0
        else:
            count = count + 1 if val == candidate else 1
            candidate = val
            if count >= confirm:
                if current == 'crisis':
                    val = REGIME_KEYS[min(REGIME_KEYS.index(val), REGIME_KEYS.index('crisis') + 1)]
                current, candidate, count = val, None, 0
        out.iloc[i] = current
    return out


def regime_series(index: pd.DataFrame, breadth: pd.Series | None = None, confirm: int = CONFIRM_DAYS) -> pd.DataFrame:
    """Возвращает DataFrame с индикаторами и окончательным режимом."""
    df = indicators(index, breadth)
    df['regime'] = apply_hysteresis(df['raw'], confirm)
    return df

def breadth_from_panel(panel: pd.DataFrame) -> pd.Series:
    """Вычисляет ширину рынка из панели."""
    panel = panel.copy()
    panel = panel.sort_values('DATE')
    panel = panel.set_index('DATE')
    cond = panel['dist_sma50'] > 0
    counts = cond.groupby(panel.index).sum()
    total = panel['dist_sma50'].groupby(panel.index).count()
    return (counts / total).sort_index()
