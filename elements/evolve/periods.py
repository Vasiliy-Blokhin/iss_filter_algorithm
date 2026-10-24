""" Реалистичная статистика генома по периодам до текущего торгового дня (1 мес. … 5 лет) —
основная метрика отбора этапов (PERIOD_SCORE). Счёт как в слепом тесте всей системы
(elements/regime/realistic): лоты, комиссии АЛОР и биржи, проскальзывание, гэп, лимиты.
Для этапа, найденного на этих же данных, это не слепая оценка: поиск её не видит, но
по ней выбирается порядок топа — поэтому истинно слепая часть остаётся отдельно (BLIND_*)."""
import numpy as np

from elements.evolve.genome import Genome
from elements.evolve.market import EMarket
from elements.regime.pools import entry_arrays, trades_of
from elements.regime.realistic import simulate_account, period_stats, parse_periods
from source.settings.settings import (
    BLIND_PERIODS, BLIND_CAPITAL, MAX_ENTRY_GAP, MAX_POSITIONS, RISK_PER_TRADE, STOP_ATR, COMMISSION,
    SLIPPAGE, SELFTUNE_DD_PENALTY,
)

SCORE_PERIODS = ('6m', '1y', '2y', '3y', '5y')  # месяц слишком шумный для отбора


class PeriodMeter:

    def __init__(self, market: EMarket, periods: str = BLIND_PERIODS):
        self.market = market
        self.periods = parse_periods(periods)
        self.arrays = entry_arrays(market)

    def measure(self, genome: Genome) -> tuple[dict, float]:
        """ {период: {TOTAL_RETURN, CAGR, MAX_DRAWDOWN, TRADES}} и оценка: средний CAGR периодов
        SCORE_PERIODS + SELFTUNE_DD_PENALTY × средняя просадка."""
        market = self.market
        view, picks = market.picks(genome)
        items = [(day, 0, row, weight, float(view.ret[row]), int(view.eidx[row]), 0)
                 for day, row, weight, _ in picks if np.isfinite(view.ret[row])]
        trades = trades_of(market, self.arrays, items)
        result = {}
        for label, days in self.periods:
            first = max(market.first, market.last - days + 1)
            account = simulate_account(trades, first, market.last, len(market.dates), BLIND_CAPITAL,
                                       max_positions=MAX_POSITIONS, risk_per_trade=RISK_PER_TRADE,
                                       stop_atr=STOP_ATR, cost_per_side=COMMISSION + SLIPPAGE,
                                       max_gap=MAX_ENTRY_GAP)
            stats = period_stats(account['daily'])
            result[label] = {key: round(stats[key], 4) for key in ('TOTAL_RETURN', 'CAGR', 'MAX_DRAWDOWN')}
            result[label]['TRADES'] = account['trades']
        chosen = [result[label] for label in SCORE_PERIODS if label in result] or list(result.values())
        score = float(np.mean([r['CAGR'] for r in chosen])
                      + SELFTUNE_DD_PENALTY * np.mean([r['MAX_DRAWDOWN'] for r in chosen]))
        return result, score
