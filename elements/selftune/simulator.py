""" Быстрый симулятор для самокоррекции: сделки стратегий и портфель с коррекцией весов.

Сделки стратегии — те же правила, что Backtest.trades (вход по открытию t+1, стоп/тейк/время,
не более MAX_SIGNALS бумаг в день и MAX_PER_SECTOR на сектор, без повторного входа в открытую
бумагу), но на массивах numpy — в десятки раз быстрее, что нужно для перебора.

Портфель: каждый день сигналы всех стратегий; позиция = slot * w доли капитала, где w —
вес стратегии по её теневым сделкам, закрытым к этому дню (correction_weight);
ограничения — риск до стопа RISK_PER_TRADE, не больше MAX_POSITIONS позиций и 100% капитала.
"""
import numpy as np
import pandas as pd

from elements.strategy.worker import rule_condition, tradable, dividend_safe
from source.settings.settings import (
    MAX_SIGNALS,
    MAX_PER_SECTOR,
    HOLD_DAYS,
    STOP_ATR,
    RISK_PER_TRADE,
    MAX_POSITIONS,
)
from source.settings.sectors import SECTORS


SLOT_WEIGHT = 1 / (MAX_SIGNALS * HOLD_DAYS)  # вклад сделки в целевой функции подбора порогов


class Market:
    """ Панель признаков, отсортированная по дате, и её числовые массивы."""

    def __init__(self, panel: pd.DataFrame, offset: int = 0, dates: np.ndarray | None = None):
        self.panel = panel
        self.dates = dates if dates is not None else np.sort(panel['DATE'].unique())
        self.offset = offset  # номер первой строки в исходной панели
        self.didx = np.searchsorted(self.dates, panel['DATE'].to_numpy())
        exits = pd.to_datetime(panel['EXIT_DATE']).to_numpy()
        known = ~np.isnat(exits)
        self.eidx = np.full(len(panel), -1)
        self.eidx[known] = np.searchsorted(self.dates, exits[known])
        self.ret = panel['TRADE_RET'].to_numpy(dtype=float)
        self.base = (tradable(panel) & dividend_safe(panel) & panel['TRADE_RET'].notna()).to_numpy()
        self.secid = pd.factorize(panel['SECID'])[0] if offset == 0 else None
        self.sector = None
        self.atr_pct = panel['atr_pct'].to_numpy(dtype=float)

    @classmethod
    def build(cls, panel: pd.DataFrame) -> 'Market':
        panel = panel.sort_values(['DATE', 'SECID']).reset_index(drop=True)
        market = cls(panel)
        sectors = panel['SECID'].map(SECTORS).fillna('unknown')
        codes = pd.factorize(sectors)[0]
        codes[(sectors == 'unknown').to_numpy()] = -1
        market.sector = codes
        return market

    def view(self, first_day: int, last_day: int) -> 'Market':
        """ Строки с номерами дней [first_day, last_day] (панель отсортирована по дате)."""
        start = int(np.searchsorted(self.didx, max(first_day, 0), side='left'))
        stop = int(np.searchsorted(self.didx, last_day, side='right'))
        part = Market.__new__(Market)
        part.panel = self.panel.iloc[start:stop]
        part.dates = self.dates
        part.offset = start
        for name in ('didx', 'eidx', 'ret', 'base', 'secid', 'sector', 'atr_pct'):
            setattr(part, name, getattr(self, name)[start:stop])
        return part

    def day(self, date) -> int:
        return int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(date))))


def strategy_trades(market: Market, name: str, params: dict, max_exit: int | None = None) -> np.ndarray:
    """ Номера строк market со сделками стратегии (локальные номера в market)."""
    condition, score = rule_condition(name, market.panel, params)
    mask = market.base & condition.fillna(False).to_numpy(dtype=bool)
    if max_exit is not None:
        mask &= market.eidx <= max_exit
    rows = np.nonzero(mask)[0]
    if not len(rows):
        return rows
    priority = np.nan_to_num(score.to_numpy(dtype=float)[rows], nan=-np.inf)
    rows = rows[np.lexsort((-priority, market.didx[rows]))]
    didx, eidx, secid, sector = market.didx, market.eidx, market.secid, market.sector
    chosen, busy, per_sector = [], {}, {}
    current, taken = -1, 0
    for row in rows:
        day = didx[row]
        if day != current:
            current, taken = day, 0
            per_sector.clear()
        if taken >= MAX_SIGNALS or busy.get(secid[row], -1) > day:
            continue
        sec = sector[row]
        if sec >= 0 and per_sector.get(sec, 0) >= MAX_PER_SECTOR:
            continue
        per_sector[sec] = per_sector.get(sec, 0) + 1
        busy[secid[row]] = eidx[row]
        taken += 1
        chosen.append(row)
    return np.array(chosen, dtype=int)


def block_objective(market: Market, rows: np.ndarray, first_day: int, last_day: int,
                    block_days: int, lam: float, min_trades: int) -> float:
    """ Сумма результатов блоков + lam * худший блок (вклад сделок при равных слотах)."""
    if len(rows) < min_trades:
        return -1.0 + len(rows) * 1e-4  # мало сделок — хуже любого рабочего варианта
    edges = np.arange(first_day, last_day + 1, block_days)
    blocks = np.searchsorted(edges, market.didx[rows], side='right') - 1
    sums = np.bincount(blocks, weights=market.ret[rows], minlength=len(edges)) * SLOT_WEIGHT
    return float(sums.sum() + lam * sums.min())


def correction_weight(exit_days: np.ndarray, cum_returns: np.ndarray, day: int, correction: dict) -> float:
    """ Вес стратегии на день day по сделкам, закрытым в (day - lookback, day].

    exit_days — отсортированные дни выхода сделок, cum_returns — накопленная сумма их
    доходностей с ведущим нулём (len(exit_days) + 1).
    """
    hi = np.searchsorted(exit_days, day, side='right')
    lo = np.searchsorted(exit_days, day - correction['lookback'], side='right')
    count = hi - lo
    if count < correction['min_trades']:
        return 1.0
    average = (cum_returns[hi] - cum_returns[lo]) / count
    if average <= correction['cut']:
        return float(correction['w_min'])
    if average >= correction['boost']:
        return float(correction['w_max'])
    return 1.0


def trade_history(exit_days: np.ndarray, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(exit_days, kind='stable')
    return exit_days[order], np.concatenate([[0.0], np.cumsum(returns[order])])


def simulate_portfolio(market: Market, segments: list[dict], correction: dict,
                       first_day: int, last_day: int) -> dict:
    """ Портфель за [first_day, last_day].

    segments — отрезки {'start', 'end', 'trades': {стратегия: глобальные номера строк}};
    сделки отрезка (включая более ранние — для статистики коррекции) получены с порогами,
    подобранными на его начало. Входы берутся только из сделок с днём сигнала в отрезке.
    """
    daily = np.zeros(len(market.dates) + 1)
    exposure_by_day = np.zeros(len(market.dates) + 1)
    open_positions = []  # (день выхода, доля капитала, бумага)
    taken_count, exposure_sum = 0, 0.0
    weight_log = []
    for segment in segments:
        histories = {
            name: trade_history(market.eidx[rows], market.ret[rows])
            for name, rows in segment['trades'].items()
        }
        by_day: dict[int, list] = {}
        for name, rows in segment['trades'].items():
            for rank, row in enumerate(rows):
                day = market.didx[row]
                if segment['start'] <= day <= min(segment['end'], last_day) and day >= first_day:
                    by_day.setdefault(day, []).append((name, row))
        for day in range(max(segment['start'], first_day), min(segment['end'], last_day) + 1):
            open_positions = [p for p in open_positions if p[0] > day]
            exposure = sum(p[1] for p in open_positions)
            exposure_sum += exposure
            candidates = by_day.get(day)
            if not candidates:
                continue
            weights = {name: correction_weight(*histories[name], day, correction) for name in histories}
            weight_log.append(weights)
            held = {p[2] for p in open_positions}
            # Бумагу выбрали несколько стратегий — доли складываются (как в живых сигналах).
            merged: dict[int, list] = {}
            for name, row in candidates:
                w = weights[name]
                if w <= 0:
                    continue
                secid = market.secid[row]
                if secid in merged:
                    merged[secid][1] += w
                else:
                    merged[secid] = [row, w]
            for secid, (row, w) in sorted(merged.items(), key=lambda item: -item[1][1]):
                if secid in held or len(open_positions) >= MAX_POSITIONS:
                    continue
                size = correction['slot'] * w
                atr = market.atr_pct[row]
                if atr > 0:
                    size = min(size, RISK_PER_TRADE / (STOP_ATR * atr))
                size = min(size, 1.0 - exposure)
                if size < 0.01:
                    continue
                exposure += size
                exposure_by_day[day + 1:market.eidx[row] + 1] += size  # в позиции с t+1 по выход
                open_positions.append((market.eidx[row], size, secid))
                held.add(secid)
                daily[market.eidx[row]] += size * market.ret[row]
                taken_count += 1
    returns = daily[first_day:last_day + 1]
    stats = equity_stats(returns)
    stats['TRADES'] = taken_count
    stats['EXPOSURE'] = exposure_sum / max(1, last_day - first_day + 1)
    return {'stats': stats, 'daily': returns, 'weights': weight_log,
            'exposure': np.minimum(exposure_by_day[first_day:last_day + 1], 1.0)}


def equity_stats(returns: np.ndarray) -> dict:
    if not len(returns):
        return {'CAGR': 0.0, 'TOTAL_RETURN': 0.0, 'MAX_DRAWDOWN': 0.0, 'SHARPE': 0.0}
    equity = np.cumprod(1 + returns)
    years = len(returns) / 252
    total = float(equity[-1] - 1)
    cagr = float(equity[-1] ** (1 / years) - 1) if equity[-1] > 0 else -1.0
    drawdown = float((equity / np.maximum.accumulate(equity) - 1).min())
    std = returns.std()
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    return {'CAGR': cagr, 'TOTAL_RETURN': total, 'MAX_DRAWDOWN': drawdown, 'SHARPE': sharpe}
