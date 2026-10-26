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
лимиты позиций и риска, ROTATION_SHARE капитала — в режимном распределении по фондам
(elements/regime/allocation: своя политика у каждого режима, GA каждые BLIND_REFIT_DAYS дней
только на прошлом, исполнение с задержкой, издержки и комиссии фондов).
Периоды 1 мес. … 5 лет считаются отдельными счетами, открытыми N времени назад до текущего дня;
календарные годы — по счёту самого длинного периода (строки Y<год>).
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
from elements.regime import allocation as alloc
from elements.regime.assets import load_prices as allocation_prices
from elements.regime.realistic import simulate_account, period_stats, after_tax, parse_periods, period_title
from elements.regime.worker import share_by_regime
from elements.rotation.worker import Rotation, regime_daily, select_caps, eq_signal
from elements.dividend import worker as dividend
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
DIVIDEND_TOLERANCE = 0.005  # разгон «не хуже», если проигрыш периода меньше 0.5%
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
        self.universe = alloc.Universe(allocation_prices(), history['regime'])
        self.allocation: dict | None = None
        # дивидендный разгон: события считаются только по данным на день решения
        dates = pd.DatetimeIndex(self.market.dates)
        frames = dividend.load_market()
        self.div_frames = {k: v.reindex(dates) for k, v in frames.items()}
        self.div_events = dividend.plan_events(frames['close'], frames['turnover'], dividend.load_dividends())
        self.eq_returns = self.prices['EQ'].reindex(dates).ffill().pct_change().fillna(0)

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
        self.allocation = self.run_allocation()
        result = self.account(trades, tags)
        result['seconds'] = monotonic() - started
        result['refits'] = len(refits)
        return result

    def run_allocation(self) -> dict:
        """ Режимное распределение вслепую с начала самого длинного периода до последнего дня."""
        market, universe = self.market, self.universe
        longest = max(days for _, days in self.periods)
        start = market.dates[max(market.first + 252, market.last - longest + 1)]
        end = market.dates[market.last]
        return alloc.walk_forward(universe, start, end, alloc.PRODUCTION, refit_days=self.refit_days,
                                  seed=self.rng.randrange(1 << 30))

    def allocation_daily(self, first: int, with_eq: bool = False):
        """ Дневная доходность режимного распределения по дням рынка акций для счёта, открытого
        в день first (старт в деньгах, далее — доли слепого прохода); with_eq — ещё и доля
        фонда акций по дням (для дивидендного разгона)."""
        universe, market = self.universe, self.market
        weights, days = self.allocation['targets']
        begin = universe.index_of(market.dates[first])
        last = self.allocation['last']
        keep = days >= begin
        account = universe.account((weights[keep], days[keep]), begin, last)
        equity = pd.Series(account['equity'], index=universe.dates[begin:last + 1])
        dates = pd.DatetimeIndex(market.dates)
        aligned = equity.reindex(equity.index.union(dates)).ffill().reindex(dates)
        daily = aligned.pct_change()
        daily.iloc[first] = aligned.iloc[first] - 1 if np.isfinite(aligned.iloc[first]) else 0.0
        result = daily.fillna(0).to_numpy().copy()
        result[:first] = 0.0
        if not with_eq:
            return result
        from elements.regime.assets import ASSETS as ALLOC_ASSETS
        columns = [ALLOC_ASSETS.index(a) for a in ('EQ', 'SMALL')]
        eq = pd.Series(account['weights'][:, columns].sum(axis=1), index=equity.index)
        # доля на начало дня — по закрытию прошлого
        eq = eq.shift(1).reindex(eq.index.union(dates)).ffill().reindex(dates).fillna(0).to_numpy().copy()
        eq[:first] = 0.0
        return result, eq

    def dividend_extra(self, eq_weight: np.ndarray, first: int) -> np.ndarray:
        """ Добавка дивидендного разгона к дневной доходности капитала ротации с дня first."""
        dates = pd.DatetimeIndex(self.market.dates)
        part = slice(first, self.market.last + 1)
        frames = {k: v.iloc[part] for k, v in self.div_frames.items()}
        extra = dividend.overlay_daily(self.div_events, frames['open'], frames['close'], frames['turnover'],
                                       pd.Series(eq_weight[part], index=dates[part]), self.eq_returns.iloc[part])
        result = np.zeros(len(dates))
        result[part] = extra.to_numpy()
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

    def rotation(self, by_regime: bool = True, with_eq: bool = False):
        daily, weights = regime_daily(self.prices, self.regime_by_date, self.cap_of if by_regime else None,
                                      with_weights=True)
        dates = pd.DatetimeIndex(self.market.dates)
        daily = daily.reindex(dates).fillna(0).to_numpy()
        if not with_eq:
            return daily
        return daily, weights['EQ'].reindex(dates).ffill().fillna(0).to_numpy()

    def account(self, trades: list[tuple], tags: list[str]) -> dict:
        market = self.market
        arrays = self.trade_arrays(trades)
        rotation = self.rotation()
        old_rotation, old_eq = self.rotation(by_regime=False, with_eq=True)
        rotation_eq = None
        dates = pd.DatetimeIndex(market.dates)
        cash = self.prices['CASH'].reindex(dates).ffill().to_numpy()
        stocks = self.prices['EQ'].reindex(dates).ffill().to_numpy()  # MCFTR: акции с дивидендами
        years = pd.DatetimeIndex(market.dates).year.to_numpy()
        common = dict(max_positions=MAX_POSITIONS, risk_per_trade=RISK_PER_TRADE, stop_atr=STOP_ATR,
                      cost_per_side=COMMISSION + SLIPPAGE, max_gap=MAX_ENTRY_GAP)
        rows, system = [], None
        for label, days in self.periods:
            first = max(market.first, market.last - days + 1)
            if self.allocation is not None:
                rotation, rotation_eq = self.allocation_daily(first, with_eq=True)
            elif rotation_eq is None:
                _, rotation_eq = self.rotation(with_eq=True)
            # дивидендный разгон — в доле фонда акций и режимного распределения, и прежней ротации
            div_extra = self.dividend_extra(rotation_eq, first)
            old_extra = self.dividend_extra(old_eq, first)
            system = simulate_account(arrays, first, market.last, len(market.dates), BLIND_CAPITAL,
                                      short_share=1 - ROTATION_SHARE, rotation_daily=rotation + div_extra, **common)
            plain = simulate_account(arrays, first, market.last, len(market.dates), BLIND_CAPITAL,
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
                'NODIV_RETURN': period_stats(plain['daily'])['TOTAL_RETURN'],
                'ROTATION_DIV_RETURN': float(np.prod(1 + rotation[part] + div_extra[part]) - 1),
                'OLD_ROTATION_DIV_RETURN': float(np.prod(1 + old_rotation[part] + old_extra[part]) - 1),
                'BENCH_RETURN': float(stocks[market.last] / stocks[first - 1] - 1) if first > 0 else 0.0,
                'CASH_RETURN': float(cash[market.last] / cash[first - 1] - 1) if first > 0 else 0.0,
                'REGIMES': share_by_regime(pd.Series(self.regimes[part])),
            })
        years_rows = self.year_rows(system['daily'], first, cash, stocks) if system is not None else []
        return {'periods': rows, 'years': years_rows, 'log': self.log, 'trades_total': len(arrays['day']),
                'allocation_log': self.allocation['log'] if self.allocation else []}

    def year_rows(self, daily: np.ndarray, first: int, cash: np.ndarray, stocks: np.ndarray) -> list[dict]:
        """ Календарные годы счёта самого длинного периода: итог, годовых (для неполного года —
        в пересчёте на год), просадка, акции и фонд денег за тот же год."""
        market = self.market
        years = pd.DatetimeIndex(market.dates[first:market.last + 1]).year.to_numpy()
        rows = []
        for year in np.unique(years):
            part = np.nonzero(years == year)[0]
            stats = period_stats(daily[part])
            a, b = first + part[0], first + part[-1]
            rows.append({
                'PERIOD': f'Y{year}', 'START': market.date(a), 'END': market.date(b), **stats,
                'CAGR': stats['TOTAL_RETURN'] if len(part) >= 240 else stats['CAGR'],
                'AFTER_TAX': stats['TOTAL_RETURN'] * (1 - TAX_RATE) if stats['TOTAL_RETURN'] > 0 else stats['TOTAL_RETURN'],
                'TRADES': 0, 'WIN_RATE': None, 'FEES': None,
                'BENCH_RETURN': float(stocks[b] / stocks[a - 1] - 1) if a > 0 else 0.0,
                'CASH_RETURN': float(cash[b] / cash[a - 1] - 1) if a > 0 else 0.0,
                'REGIMES': share_by_regime(pd.Series(self.regimes[a:b + 1])),
            })
        return rows

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
                'nodiv': round(r['NODIV_RETURN'], 4), 'rotation_div': round(r['ROTATION_DIV_RETURN'], 4),
                'old_rotation_div': round(r['OLD_ROTATION_DIV_RETURN'], 4),
                'by_tag': {k: [v[0], round(v[1] / v[0], 4) if v[0] else 0] for k, v in r['SHORT_BY_TAG'].items()},
            }, ensure_ascii=False)[:2000],
        } for r in result['periods']]
        rows += [{
            'CREATED_AT': now, 'RUN_KEY': key, 'PERIOD': r['PERIOD'], 'START': r['START'], 'END': r['END'],
            'TOTAL_RETURN': r['TOTAL_RETURN'], 'CAGR': r['CAGR'], 'AFTER_TAX': r['AFTER_TAX'],
            'MAX_DRAWDOWN': r['MAX_DRAWDOWN'], 'SHARPE': r['SHARPE'], 'TRADES': 0, 'WIN_RATE': None,
            'FEES': None, 'BENCH_RETURN': r['BENCH_RETURN'], 'CASH_RETURN': r['CASH_RETURN'],
            'REGIMES': json.dumps(r['REGIMES']), 'NOTE': note[:2000],
        } for r in result.get('years', [])]
        for row in rows:
            SQLmain.append_data(row, tables.BlindRun)
        return key


def summary_lines(result: dict) -> list[str]:
    """ Короткие строки итога: период — итог счёта (годовых), просадка, против IMOEX и фонда денег;
    затем календарные годы."""
    lines = []
    for r in result['periods']:
        lines.append(
            f'{period_title(r["PERIOD"])}: {r["TOTAL_RETURN"]:+.1%} ({r["CAGR"]:+.1%} год.)'
            + f', просадка {r["MAX_DRAWDOWN"]:.1%}; акции {r["BENCH_RETURN"]:+.1%}, фонд денег {r["CASH_RETURN"]:+.1%}')
    if result.get('years'):
        lines.append('По годам: ' + ', '.join(f'{r["PERIOD"][1:]} {r["CAGR"]:+.1%}' for r in result['years']))
    return lines


def decide_switches(result: dict) -> tuple[dict, str]:
    """ Самонастройка по слепому тесту: режимное распределение фондов заменяет прежнюю ротацию в live,
    только если оно лучше неё на большинстве периодов от года и на самом длинном; краткосрочная
    часть в live — только если она в плюсе и за год, и за два."""
    long = [r for r in result['periods'] if r['PERIOD'][-1] == 'y']
    better = [r['ROTATION_RETURN'] > r['OLD_ROTATION_RETURN'] for r in long]
    rotation = bool(long) and sum(better) * 2 > len(better) and better[-1]
    check = [r for r in long if r['PERIOD'] in ('1y', '2y')] or long
    short = bool(check) and all(r['SHORT_RETURN'] > 0 for r in check)
    # дивидендный разгон — в той ротации, что будет в live
    base, with_div = ('ROTATION_RETURN', 'ROTATION_DIV_RETURN') if rotation else \
        ('OLD_ROTATION_RETURN', 'OLD_ROTATION_DIV_RETURN')
    # без акций в ротации разгон не работает и результат равен — это не «хуже»
    div_better = [r.get(with_div, r[base]) > r[base] for r in long]
    div_worse = [r.get(with_div, r[base]) < r[base] - DIVIDEND_TOLERANCE for r in long]
    dividend = bool(long) and not any(div_worse) and div_better[-1]
    note = (f'режимное распределение фондов: {"включено" if rotation else "выключено (прежняя ротация лучше)"} '
            f'(лучше прежней на {sum(better)} из {len(better)} периодов от года); '
            f'краткосрочные сделки в live: {"разрешены" if short else "запрещены (в слепом тесте в минусе)"}; '
            f'дивидендный разгон в live: {"разрешён" if dividend else "запрещён"} '
            f'(лучше, чем без него, на {sum(div_better)} из {len(div_better)} периодов от года, '
            f'хуже — на {sum(div_worse)})')
    return {'rotation': rotation, 'short': short, 'dividend': dividend}, note
