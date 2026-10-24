""" Реалистичный счёт в рублях: лоты, комиссии АЛОР и биржи, проскальзывание по ликвидности,
фильтр гэпа на входе, лимиты позиций и риска, часть капитала в ротации фондов."""
import numpy as np
from typing import Dict, Any, Tuple, List

def liquidity_slippage(turnover: float) -> float:
    """Определяет дополнительный слиппинг по объёму."""
    if np.isnan(turnover):
        return 0.0005
    if turnover < 1.5e8:
        return 0.0005
    if turnover < 3e8:
        return 0.0002
    return 0.0

def simulate_account(
    trades: Dict[str, np.ndarray],
    first_day: int,
    last_day: int,
    n_days: int,
    capital: float,
    *,
    max_positions: int,
    risk_per_trade: float,
    stop_atr: float,
    cost_per_side: float,
    max_gap: float,
    short_share: float = 1.0,
    rotation_daily: np.ndarray | None = None
) -> Dict[str, Any]:
    """Симуляция счёта."""
    rot_value = capital * (1 - short_share)
    free_cash = capital * short_share
    invested = 0.0
    fees_total = 0.0
    equity_arr = np.empty(last_day - first_day + 1, dtype=float)
    trade_days: List[int] = []

    by_tag: Dict[int, List[float]] = {}
    realized_returns: List[float] = []

    open_trades: List[Tuple[int, float, float, int, int, float]] = []  # (exit, value, pnl, secid, tag, ret)

    # Pre-sort trades by day asc, priority desc (already sorted)
    days = trades['day']
    exits = trades['exit']
    ret = trades['ret']
    weight = trades['weight']
    entry = trades['entry']
    close = trades['close']
    lot = trades['lot']
    atr_pct = trades['atr_pct']
    turnover = trades['turnover']
    secid = trades['secid']
    tag = trades['tag']

    # Map day to indices of trades entering that day
    day_to_indices = {}
    for idx, d in enumerate(days):
        day_to_indices.setdefault(d, []).append(idx)

    for d in range(first_day, last_day + 1):
        # 1) закрываем позиции
        remaining_trades = []
        for item in open_trades:
            ex, val, pnl, sid, code, trade_ret = item
            if ex <= d:
                free_cash += val + pnl
                invested -= val
                realized_returns.append(trade_ret)
                by_tag.setdefault(int(code), [0, 0.0])
                by_tag[int(code)][0] += 1
                by_tag[int(code)][1] += trade_ret
            else:
                remaining_trades.append(item)
        open_trades = remaining_trades

        # 2) rotation
        if rotation_daily is not None:
            rot_value *= (1 + rotation_daily[d])

        # 3) входы
        held = {t[3] for t in open_trades}
        for idx in day_to_indices.get(d - 1, []):
            if secid[idx] in held:
                continue
            if len(open_trades) >= max_positions:
                continue
            if abs(entry[idx] / close[idx] - 1) > max_gap:
                continue
            if exits[idx] <= d:
                continue
            if not np.isfinite(entry[idx]) or entry[idx] <= 0:
                continue
            if not np.isfinite(ret[idx]):
                continue

            equity_now = free_cash + invested + rot_value
            target = min(weight[idx] * equity_now,
                        risk_per_trade * equity_now / (stop_atr * atr_pct[idx]) if atr_pct[idx] > 0 else weight[idx] * equity_now)
            target = min(target, free_cash)
            slip = liquidity_slippage(turnover[idx])
            price_in = entry[idx] * (1 + cost_per_side + slip)
            lots = int(np.floor(target / (price_in * lot[idx])))
            if lots < 1:
                continue
            value = lots * lot[idx] * entry[idx]
            trade_pnl = value * (ret[idx] - 2 * slip)
            trade_fees = value * 2 * (cost_per_side + slip)
            free_cash -= value
            invested += value
            open_trades.append((exits[idx], value, trade_pnl, secid[idx], tag[idx], ret[idx] - 2 * slip))
            held.add(secid[idx])
            trade_days.append(d)
            fees_total += trade_fees

        # 4) запись equity
        equity_arr[d - first_day] = free_cash + invested + rot_value

        # rebalance every 21 days
        if (d - first_day + 1) % 21 == 0:
            total = free_cash + invested + rot_value
            rot_value = (1 - short_share) * total
            free_cash = total * short_share - invested
            if free_cash < 0:
                free_cash = 0.0
                rot_value = total - invested

    equity = equity_arr
    daily_ret = np.empty_like(equity)
    daily_ret[0] = equity[0] / capital - 1
    daily_ret[1:] = equity[1:] / equity[:-1] - 1

    win_rate = sum(r > 0 for r in realized_returns) / len(realized_returns) if realized_returns else 0.0
    avg_trade = np.mean(realized_returns) if realized_returns else 0.0

    return {
        'equity': equity,
        'daily': daily_ret,
        'trades': len(realized_returns),
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'fees': fees_total,
        'by_tag': {k: v for k, v in by_tag.items()},
        'trade_days': np.array(trade_days, dtype=int)
    }

def period_stats(daily: np.ndarray, periods_per_year: int = 252) -> Dict[str, float]:
    """Статистика периода."""
    if daily.size == 0:
        return {'TOTAL_RETURN': 0.0, 'CAGR': 0.0, 'MAX_DRAWDOWN': 0.0, 'SHARPE': 0.0}
    equity = np.cumprod(1 + daily)
    total = equity[-1] - 1
    years = len(daily) / periods_per_year
    cagr = equity[-1] ** (1 / years) - 1 if equity[-1] > 0 else -1.0
    running_max = np.maximum(np.maximum.accumulate(equity), 1.0)
    drawdown = np.min(equity / running_max - 1)
    std = np.std(daily, ddof=1) if len(daily) > 1 else 0.0
    sharpe = np.mean(daily) / std * np.sqrt(periods_per_year) if std > 0 else 0.0
    return {'TOTAL_RETURN': float(total), 'CAGR': float(cagr), 'MAX_DRAWDOWN': float(min(0.0, drawdown)),
            'SHARPE': float(sharpe)}

def after_tax(daily: np.ndarray, years_index: np.ndarray, tax_rate: float) -> float:
    """Налоги по годам."""
    if daily.size == 0:
        return 0.0
    equity = np.cumprod(1 + daily)
    years = np.unique(years_index)
    total = 1.0
    for y in years:
        mask = years_index == y
        year_ret = np.prod(1 + daily[mask]) - 1
        if year_ret > 0:
            year_ret *= (1 - tax_rate)
        total *= (1 + year_ret)
    return total - 1.0


UNIT = {'m': 21, 'y': 252}
PERIOD_TITLES = {'m': 'мес.', 'y': 'г.'}


def parse_periods(spec: str) -> list[tuple[str, int]]:
    """ '1m,6m,1y,2y' -> [('1m', 21), ('6m', 126), ('1y', 252), ('2y', 504)]."""
    result = []
    for part in spec.split(','):
        part = part.strip().lower()
        if part and part[-1] in UNIT and part[:-1].isdigit():
            result.append((part, int(part[:-1]) * UNIT[part[-1]]))
    return result


def period_title(label: str) -> str:
    number, unit = label[:-1], label[-1]
    if unit == 'm':
        return {'1': '1 мес.', '6': 'полгода'}.get(number, f'{number} мес.')
    return {'1': '1 год', '2': '2 года', '3': '3 года', '4': '4 года'}.get(number, f'{number} лет')
