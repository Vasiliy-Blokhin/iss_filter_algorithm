""" Признаки для модели и симуляция исхода сделки на 1-2 дня.

Все признаки дня t считаются только по данным до закрытия дня t включительно.
Вход — по цене открытия дня t+1, выход — по стопу/тейку или по закрытию дня t+HOLD.
"""
import numpy as np
import pandas as pd

from source.settings.settings import (
    INDEX_SECID,
    HOLD_DAYS,
    STOP_ATR,
    TAKE_ATR,
    COMMISSION,
    SLIPPAGE,
)


FEATURES = [
    'r1', 'r3', 'r5', 'r10', 'r20',
    'r1_cs', 'r5_cs', 'rel5',
    'rsi2', 'rsi14',
    'dist_sma20', 'dist_sma50', 'dist_sma200', 'dist_high20', 'dist_low20',
    'atr_pct', 'range_pct', 'clv', 'gap', 'vol_ratio', 'vol_ratio5',
    'idx_r1', 'idx_r5', 'idx_dist_sma50',
]


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def security_features(frame: pd.DataFrame) -> pd.DataFrame:
    """ Признаки одной бумаги (frame отсортирован по дате)."""
    data = frame.copy()
    close, high, low = data['CLOSE'], data['HIGH'], data['LOW']
    prev_close = close.shift(1)

    for period in (1, 3, 5, 10, 20):
        data[f'r{period}'] = close.pct_change(period)
    data['rsi2'] = rsi(close, 2)
    data['rsi14'] = rsi(close, 14)
    for period in (20, 50, 200):
        data[f'dist_sma{period}'] = close / close.rolling(period, min_periods=period // 2).mean() - 1
    data['dist_high20'] = close / high.rolling(20).max() - 1
    data['dist_low20'] = close / low.rolling(20).min() - 1

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    data['ATR'] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    data['atr_pct'] = data['ATR'] / close
    data['range_pct'] = (high - low) / close
    data['clv'] = ((close - low) / (high - low).replace(0, np.nan)).fillna(0.5)
    data['gap'] = data['OPEN'] / prev_close - 1

    value = data['VALUE'].fillna(0)
    data['TURNOVER20'] = value.rolling(20, min_periods=10).median()
    data['vol_ratio'] = value / data['TURNOVER20'].replace(0, np.nan)
    data['vol_ratio5'] = value.rolling(5).mean() / data['TURNOVER20'].replace(0, np.nan)
    return data


def trade_outcomes(
    frame: pd.DataFrame,
    hold: int = HOLD_DAYS,
    stop_atr: float = STOP_ATR,
    take_atr: float = TAKE_ATR,
) -> pd.DataFrame:
    """ Исход сделки, открытой по сигналу дня t (для меток и бэктеста).

    Консервативно: если в один день задеты и стоп, и тейк — считаем стоп.
    """
    opens = frame['OPEN'].to_numpy()
    highs = frame['HIGH'].to_numpy()
    lows = frame['LOW'].to_numpy()
    closes = frame['CLOSE'].to_numpy()
    atr = frame['ATR'].to_numpy()
    count = len(frame)

    result = np.full(count, np.nan)
    exit_offset = np.full(count, np.nan)
    exit_kind = np.full(count, '', dtype=object)
    costs = 2 * (COMMISSION + SLIPPAGE)

    for t in range(count - hold):
        entry = opens[t + 1]
        if not entry or np.isnan(entry) or np.isnan(atr[t]):
            continue
        stop = entry - stop_atr * atr[t]
        take = entry + take_atr * atr[t]
        exit_price, kind, offset = closes[t + hold], 'time', hold
        for day in range(1, hold + 1):
            index = t + day
            if day > 1 and opens[index] <= stop:
                exit_price, kind, offset = opens[index], 'stop', day
                break
            if day > 1 and opens[index] >= take:
                exit_price, kind, offset = opens[index], 'take', day
                break
            if lows[index] <= stop:
                exit_price, kind, offset = stop, 'stop', day
                break
            if highs[index] >= take:
                exit_price, kind, offset = take, 'take', day
                break
        result[t] = exit_price / entry - 1 - costs
        exit_offset[t] = offset
        exit_kind[t] = kind

    return pd.DataFrame(
        {'TRADE_RET': result, 'EXIT_OFFSET': exit_offset, 'EXIT_KIND': exit_kind},
        index=frame.index,
    )


def build_dataset(candles: pd.DataFrame, with_outcomes: bool = True, **trade_params) -> pd.DataFrame:
    """ Панель признаков по всем бумагам + признаки индекса + кросс-секция."""
    # Сессии выходного дня (с 2025 г.) тонкие — работаем только с будними свечами.
    candles = candles[candles['DATE'].dt.weekday < 5]
    index = candles[candles['SECID'] == INDEX_SECID].set_index('DATE')['CLOSE'].sort_index()
    index_features = pd.DataFrame({
        'idx_r1': index.pct_change(1),
        'idx_r5': index.pct_change(5),
        'idx_dist_sma50': index / index.rolling(50, min_periods=25).mean() - 1,
    })

    parts = []
    for secid, frame in candles[candles['SECID'] != INDEX_SECID].groupby('SECID', sort=False):
        frame = frame.sort_values('DATE').reset_index(drop=True)
        if len(frame) < 60:
            continue
        data = security_features(frame)
        if with_outcomes:
            data = data.join(trade_outcomes(data, **trade_params))
            offsets = data['EXIT_OFFSET'].fillna(0).astype(int).to_numpy()
            dates = data['DATE'].to_numpy()
            data['EXIT_DATE'] = [
                dates[i + offsets[i]] if offsets[i] else pd.NaT for i in range(len(data))
            ]
        parts.append(data)

    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.join(index_features, on='DATE')
    panel['rel5'] = panel['r5'] - panel['idx_r5']
    by_date = panel.groupby('DATE')
    panel['r1_cs'] = panel['r1'] - by_date['r1'].transform('median')
    panel['r5_cs'] = panel['r5'] - by_date['r5'].transform('median')
    return panel.replace([np.inf, -np.inf], np.nan)
