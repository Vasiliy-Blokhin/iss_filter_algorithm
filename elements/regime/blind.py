""" Полностью слепой тест всей системы — имитация её самостоятельной работы.

С начала самого длинного периода (по умолчанию 5 лет до текущего торгового дня) система идёт
вперёд и каждые BLIND_REFIT_DAYS торговых дней «переобучается» только на прошлых данных:
  - пороги генов (квантили) — по дням до точки переобучения;
  - генетика: для общего пула и каждого режима — BLIND_GA_STEPS шагов от лучших геномов прошлого
    переобучения, в каждой пачке EVOLVE_BATCH шагов не меньше EVOLVE_FRESH_MIN совсем новых геномов;
  - пулы режимов — по сделкам, закрытым до точки переобучения.
До следующего переобучения в каждый день торгуется пул режима этого дня (режим известен по
закрытию дня, вход — по открытию следующего). Счёт — реалистичный (elements/regime/realistic):
капитал BLIND_CAPITAL, лоты, комиссия АЛОР + биржи, проскальзывание по ликвидности, фильтр гэпа,
лимиты позиций и риска, ROTATION_SHARE капитала — в ротации фондов с долей акций по режиму.
Периоды 1 мес. … 5 лет считаются отдельными счетами, открытыми N времени назад до текущего дня.
Остаточное заглядывание вперёд: пороги правиловых стратегий подобраны на 2020–2024 (см. README).
"""
import json
import logging
import random
import secrets
from datetime import datetime
from time import monotonic

import numpy as np
import pandas as pd

from elements.evolve.genome import Genome, OPERATORS
from elements.evolve.market import EMarket
from elements.evolve.worker import fitness, search_ok
from elements.regime.detector import REGIME_KEYS, regime_info
from elements.regime.pools import (
    Candidate, build_pools, genome_candidate, regimes_by_day, rule_candidates, strategy_label,
    entry_arrays, trades_of,
)
from elements.regime.realistic import simulate_account, period_stats, after_tax, parse_periods, period_title
from elements.regime.worker import share_by_regime
from elements.rotation.worker import Rotation, regime_daily, select_caps, eq_signal
from source.settings.settings import (
    handler, BLIND_PERIODS, BLIND_REFIT_DAYS, BLIND_GA_STEPS, BLIND_CAPITAL, MAX_ENTRY_GAP,
    ROTATION_SHARE, ROTATION_EQ_CAP, TAX_RATE, MAX_POSITIONS, RISK_PER_TRADE, STOP_ATR, COMMISSION,
    SLIPPAGE, EVOLVE_BATCH, EVOLVE_FRESH_MIN,
)
from source.sql.main import SQLmain
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

GA_KEEP = 3        # лучших геномов группы — кандидаты в пулы и родители следующего переобучения
GA_POOL = 10       # геномов в пуле поиска группы
def mini_ga(market: EMarket, rng: random.Random, steps: int, seeds: list[Genome]) -> list[tuple[Genome, dict]]:
    """ Короткий генетический поиск на текущем окне рынка (маска дней уже задана).
    Каждая пачка из EVOLVE_BATCH шагов начинается с EVOLVE_FRESH_MIN совсем новых геномов."""
    pool: list[tuple[Genome, dict]] = []
    for genome in seeds:
        pool.append((genome, market.evaluate(genome)))
    names = [n for n in OPERATORS if n not in ('fresh', 'seeded')]
    weights = [OPERATORS[n][1] for n in names]
    seen = {g.key() for g, _ in pool}
    for step in range(steps):
        donors = [g for g, _ in pool]
        if step % EVOLVE_BATCH < EVOLVE_FRESH_MIN or not pool:
            child = OPERATORS['fresh'][0](rng, market, None, donors)
        else:
            parent = max(rng.sample(pool, k=min(3, len(pool))), key=lambda item: fitness(item[1]))[0]
            child = OPERATORS[rng.choices(names, weights=weights)[0]][0](rng, market, parent, donors)
        child = child.repair(rng, market)
        if child.key() in seen:
            continue
        seen.add(child.key())
        pool.append((child, market.evaluate(child)))
        pool.sort(key=lambda item: -fitness(item[1]))
        del pool[GA_POOL:]
    return [(g, r) for g, r in pool if search_ok(r)][:GA_KEEP]


class FullBlind:

    def __init__(self, panel: pd.DataFrame, history: pd.DataFrame, ga_steps: int = BLIND_GA_STEPS,
                 refit_days: int = BLIND_REFIT_DAYS, periods: str = BLIND_PERIODS, seed: int | None = None):
        self.market = EMarket(panel, holdout_days=0, blind_days=0)
        self.regimes = regimes_by_day(self.market, history)
        self.ga_steps, self.refit_days = ga_steps, refit_days
        self.periods = parse_periods(periods)
        self.rng = random.Random(seed if seed is not None else secrets.randbits(64))
        self.log: list[str] = []
        self.prices = Rotation.load_prices()
        self.signal = eq_signal(self.prices)
        self.regime_by_date = pd.Series(self.regimes, index=pd.DatetimeIndex(self.market.dates))
        self.caps: list[tuple[pd.Timestamp, dict]] = []  # (точка переобучения, доли акций по режимам)
        self.arrays = entry_arrays(self.market)

    # --- шаги ------------------------------------------------------------------------
    def refit(self, day: int, carry: dict) -> list[Candidate]:
        """ Кандидаты на точке переобучения day: всё считается по данным до day − 1."""
        market = self.market
        market.set_days(None)
        market.set_search_end(day - 1)
        candidates = rule_candidates(market)
        if self.ga_steps > 0:
            for group in ['all', *REGIME_KEYS]:
                mask = None if group == 'all' else (self.regimes == group)
                if mask is not None and mask[market.first:day].sum() < 40:
                    continue  # режима почти не было — искать не на чем
                market.set_days(mask)
                found = mini_ga(market, self.rng, self.ga_steps, carry.get(group, []))
                carry[group] = [g for g, _ in found]
                market.set_days(None)
                for number, (genome, _) in enumerate(found):
                    candidates.append(genome_candidate(market, f'gen:{group}:{day}.{number}', genome))
        return candidates

    def run(self, progress=None) -> dict:
        market = self.market
        started = monotonic()
        longest = max(days for _, days in self.periods)
        start = max(market.first + 252, market.last - longest + 1)
        carry: dict[str, list[Genome]] = {}
        trades: list[tuple] = []
        tags: list[str] = []
        refits = list(range(start, market.last + 1, self.refit_days))
        for number, day in enumerate(refits):
            end = refits[number + 1] - 1 if number + 1 < len(refits) else market.last
            candidates = self.refit(day, carry)
            caps = select_caps(self.prices, self.regime_by_date, market.dates[day - 1], self.signal)
            self.caps.append((pd.Timestamp(market.dates[day]), caps))
            pools = build_pools(market, candidates, self.regimes, market.first, day - 1)
            by_name = {c.name: c for c in candidates}
            for key, members in pools.items():
                for rank, member in enumerate(members):
                    candidate = by_name[member['STRATEGY']]
                    label = strategy_label(member['STRATEGY']) if candidate.genome is None else \
                        f'гены:{member["STRATEGY"].split(":")[1]}'
                    if label not in tags:
                        tags.append(label)
                    view = candidate.view(market)
                    for pick_day, row, weight, score in candidate.picks:
                        if day <= pick_day <= end and self.regimes[pick_day] == key:
                            trades.append((pick_day, rank, -score, row, weight, view.ret[row],
                                           view.eidx[row], tags.index(label)))
            line = (f'{market.date(day)}: ' + '; '.join(
                f'{regime_info(k)["emoji"]} акции {caps[k]:.0%} ' + (', '.join(strategy_label(m['STRATEGY']) for m in v) or '—')
                for k, v in pools.items()))
            self.log.append(line)
            logger.info(f'Слепой тест, переобучение {number + 1}/{len(refits)}: {line}')
            if progress:
                progress(number + 1, len(refits))
        result = self.account(trades, tags)
        result['seconds'] = monotonic() - started
        result['refits'] = len(refits)
        return result

    # --- счёт ------------------------------------------------------------------------
    def trade_arrays(self, trades: list[tuple]) -> dict:
        trades.sort(key=lambda t: (t[0], t[1], t[2]))
        return trades_of(self.market, self.arrays, [(t[0], 0, t[3], t[4], t[5], t[6], t[7]) for t in trades])

    def cap_of(self, day, regime: str) -> float:
        """ Доля акций режима по последнему переобучению до дня (до первого — постоянная)."""
        caps = None
        for moment, values in self.caps:
            if moment > day:
                break
            caps = values
        return caps[regime] if caps else ROTATION_EQ_CAP

    def rotation(self, by_regime: bool = True) -> np.ndarray:
        daily = regime_daily(self.prices, self.regime_by_date, self.cap_of if by_regime else None)
        return daily.reindex(pd.DatetimeIndex(self.market.dates)).fillna(0).to_numpy()

    def account(self, trades: list[tuple], tags: list[str]) -> dict:
        market = self.market
        arrays = self.trade_arrays(trades)
        rotation, old_rotation = self.rotation(), self.rotation(by_regime=False)
        dates = pd.DatetimeIndex(market.dates)
        cash = self.prices['CASH'].reindex(dates).ffill().to_numpy()
        stocks = self.prices['EQ'].reindex(dates).ffill().to_numpy()  # MCFTR: акции с дивидендами
        years = pd.DatetimeIndex(market.dates).year.to_numpy()
        common = dict(max_positions=MAX_POSITIONS, risk_per_trade=RISK_PER_TRADE, stop_atr=STOP_ATR,
                      cost_per_side=COMMISSION + SLIPPAGE, max_gap=MAX_ENTRY_GAP)
        rows = []
        for label, days in self.periods:
            first = max(market.first, market.last - days + 1)
            system = simulate_account(arrays, first, market.last, len(market.dates), BLIND_CAPITAL,
                                      short_share=1 - ROTATION_SHARE, rotation_daily=rotation, **common)
            short = simulate_account(arrays, first, market.last, len(market.dates), BLIND_CAPITAL, **common)
            stats = period_stats(system['daily'])
            part = slice(first, market.last + 1)
            rows.append({
                'PERIOD': label, 'START': market.date(first), 'END': market.date(market.last),
                **stats, 'AFTER_TAX': after_tax(system['daily'], years[part], TAX_RATE),
                'TRADES': system['trades'], 'WIN_RATE': system['win_rate'], 'FEES': system['fees'],
                'SHORT_RETURN': period_stats(short['daily'])['TOTAL_RETURN'], 'SHORT_TRADES': short['trades'],
                'SHORT_BY_TAG': {tags[k]: v for k, v in short['by_tag'].items()},
                'ROTATION_RETURN': float(np.prod(1 + rotation[part]) - 1),
                'OLD_ROTATION_RETURN': float(np.prod(1 + old_rotation[part]) - 1),
                'BENCH_RETURN': float(stocks[market.last] / stocks[first - 1] - 1) if first > 0 else 0.0,
                'CASH_RETURN': float(cash[market.last] / cash[first - 1] - 1) if first > 0 else 0.0,
                'REGIMES': share_by_regime(pd.Series(self.regimes[part])),
            })
        return {'periods': rows, 'log': self.log, 'trades_total': len(arrays['day'])}

    # --- сохранение и текст -----------------------------------------------------------
    @staticmethod
    def save(result: dict, note: str = '') -> str:
        key = f'{datetime.now():%Y%m%d%H%M%S}'
        now = datetime.now()
        rows = [{
            'CREATED_AT': now, 'RUN_KEY': key, 'PERIOD': r['PERIOD'], 'START': r['START'], 'END': r['END'],
            'TOTAL_RETURN': r['TOTAL_RETURN'], 'CAGR': r['CAGR'], 'AFTER_TAX': r['AFTER_TAX'],
            'MAX_DRAWDOWN': r['MAX_DRAWDOWN'], 'SHARPE': r['SHARPE'], 'TRADES': r['TRADES'],
            'WIN_RATE': r['WIN_RATE'], 'FEES': r['FEES'], 'BENCH_RETURN': r['BENCH_RETURN'],
            'CASH_RETURN': r['CASH_RETURN'], 'REGIMES': json.dumps(r['REGIMES']),
            'NOTE': (note + '; ' if note else '') + json.dumps({
                'short': round(r['SHORT_RETURN'], 4), 'short_trades': r['SHORT_TRADES'],
                'rotation': round(r['ROTATION_RETURN'], 4), 'old_rotation': round(r['OLD_ROTATION_RETURN'], 4),
                'by_tag': {k: [v[0], round(v[1] / v[0], 4) if v[0] else 0] for k, v in r['SHORT_BY_TAG'].items()},
            }, ensure_ascii=False)[:2000],
        } for r in result['periods']]
        for row in rows:
            SQLmain.append_data(row, tables.BlindRun)
        return key


def summary_lines(result: dict) -> list[str]:
    """ Короткие строки итога: период — итог счёта (годовых), просадка, против IMOEX и фонда денег."""
    lines = []
    for r in result['periods']:
        lines.append(
            f'{period_title(r["PERIOD"])}: {r["TOTAL_RETURN"]:+.1%}'
            + (f' ({r["CAGR"]:+.1%} год.)' if r['PERIOD'][-1] == 'y' else '')
            + f', просадка {r["MAX_DRAWDOWN"]:.1%}; акции {r["BENCH_RETURN"]:+.1%}, фонд денег {r["CASH_RETURN"]:+.1%}')
    return lines


def decide_switches(result: dict) -> tuple[dict, str]:
    """ Самонастройка по слепому тесту: доля акций ротации по режиму включается, только если она
    не хуже прежней постоянной ни на одном периоде от года и лучше хотя бы на одном; краткосрочная
    часть в live — только если она в плюсе и за год, и за два."""
    long = [r for r in result['periods'] if r['PERIOD'][-1] == 'y']
    rotation = bool(long) and all(r['ROTATION_RETURN'] >= r['OLD_ROTATION_RETURN'] for r in long) \
        and any(r['ROTATION_RETURN'] > r['OLD_ROTATION_RETURN'] for r in long)
    check = [r for r in long if r['PERIOD'] in ('1y', '2y')] or long
    short = bool(check) and all(r['SHORT_RETURN'] > 0 for r in check)
    note = (f'доли акций по режиму: {"включены" if rotation else "выключены (прежняя ротация лучше)"}; '
            f'краткосрочные сделки в live: {"разрешены" if short else "запрещены (в слепом тесте в минусе)"}')
    return {'rotation': rotation, 'short': short}, note
