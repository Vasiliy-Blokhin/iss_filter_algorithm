""" Слепой (walk-forward) бэктест механики коррекции.

Период с BLIND_START до последнего дня данных делится на отрезки по reopt_days торговых дней.
На начало каждого отрезка t пороги стратегий подбираются покоординатным спуском только по
сделкам, открытым и закрытым до t (окно train_days), затем этими порогами торгуется отрезок.
Веса стратегий (коррекция) каждый день считаются по их теневым сделкам, закрытым к этому дню.
Результаты отрезков складываются в одну кривую капитала — это и есть «слепой» результат:
ни одно решение в нём не использует данных из будущего.

Отрезки привязаны к BLIND_START, поэтому с приходом новых данных уже подобранные пороги
не меняются (кэш selftune_segments), пересчитывается только последний отрезок.
"""
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from elements.selftune.params import RULES, STRATEGY_GRID, NEUTRAL_PARAMS, params_key
from elements.selftune.simulator import (
    Market, strategy_trades, block_objective, simulate_portfolio, equity_stats,
)
from source.settings.settings import (
    handler, BLIND_START, SELFTUNE_DD_PENALTY, HOLD_DAYS, EXIT_AT_OPEN, STOP_ATR, TAKE_ATR,
)
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

MIN_TRAIN_DAYS = 200  # раньше пороги не подбираются (остаются стартовыми)
WARM_DAYS = 260  # история сделок отрезка для статистики коррекции (>= максимального lookback)


def segment_key(tuner: dict) -> str:
    """ Ключ кэша отрезков: настройки переподбора + правила выхода (от них зависят исходы сделок)."""
    return params_key({**tuner, 'exit': [HOLD_DAYS, EXIT_AT_OPEN, STOP_ATR, TAKE_ATR]})


def score_of(stats: dict) -> float:
    return stats['CAGR'] + SELFTUNE_DD_PENALTY * stats['MAX_DRAWDOWN']


def optimize_params(market: Market, start: dict, decision_day: int, tuner: dict) -> dict:
    """ Пороги на день decision_day: покоординатный спуск по STRATEGY_GRID от start,
    по сделкам с сигналом и выходом строго до decision_day. При allow_off стратегия
    выключается (enabled = 0), если и лучший вариант на окне обучения убыточен."""
    first = decision_day - tuner['train_days']
    if decision_day < MIN_TRAIN_DAYS:
        return {name: dict(start[name]) for name in RULES}
    first = max(first, 0)
    view = market.view(first, decision_day - 1)
    result = {}
    for name in RULES:
        def objective(params):
            rows = strategy_trades(view, name, params, max_exit=decision_day - 1)
            return block_objective(view, rows, first, decision_day - 1, tuner['block_days'],
                                   tuner['lam'], tuner['min_trades'])
        current = {**start[name], 'enabled': 1}
        best = objective(current)
        for _ in range(tuner['passes']):
            improved = False
            for key, values in STRATEGY_GRID[name].items():
                for value in values:
                    if value == current.get(key):
                        continue
                    candidate = {**current, key: value}
                    value_score = objective(candidate)
                    if value_score > best + 1e-6:
                        best, current, improved = value_score, candidate, True
            if not improved:
                break
        if tuner.get('allow_off') and best < 0:
            current['enabled'] = 0
        result[name] = current
    return result


class SegmentStore(SQLmain):
    """ Кэш подобранных на начало отрезка порогов (БД + память)."""

    def __init__(self):
        self.memory: dict[tuple[str, str], dict] = {}
        self.loaded: set[str] = set()

    def get(self, tuner_key: str, start) -> dict | None:
        if tuner_key not in self.loaded:
            with Session(bind=main_engine) as s:
                rows = s.execute(
                    sa.select(tables.SelftuneSegment.SEGMENT_START, tables.SelftuneSegment.PARAMS)
                    .where(tables.SelftuneSegment.TUNER_KEY == tuner_key)
                ).all()
            for segment_start, params in rows:
                self.memory[(tuner_key, str(segment_start))] = json.loads(params)
            self.loaded.add(tuner_key)
        return self.memory.get((tuner_key, str(start)))

    def put(self, tuner_key: str, start, params: dict):
        self.memory[(tuner_key, str(start))] = params
        self.upsert(tables.SelftuneSegment, [{
            'TUNER_KEY': tuner_key, 'SEGMENT_START': start,
            'PARAMS': json.dumps(params, sort_keys=True), 'CREATED_AT': datetime.now(),
        }], ['TUNER_KEY', 'SEGMENT_START'])


class BlindBacktest:

    def __init__(self, panel: pd.DataFrame, store: SegmentStore | None = None, persist: bool = True):
        self.market = Market.build(panel)
        self.store = store or SegmentStore()
        self.persist = persist
        self.trade_cache: dict[tuple[str, int, int], dict] = {}

    @property
    def last_day(self) -> int:
        return len(self.market.dates) - 1

    @property
    def data_end(self):
        return pd.Timestamp(self.market.dates[-1]).date()

    def segment_starts(self, tuner: dict) -> list[int]:
        first = max(self.market.day(BLIND_START), 1)
        return list(range(first, self.last_day + 1, tuner['reopt_days']))

    def segment_params(self, tuner: dict) -> list[tuple[int, dict]]:
        """ Пороги на начало каждого отрезка (из кэша или подбором только по прошлому)."""
        key = segment_key(tuner)
        params = NEUTRAL_PARAMS
        result = []
        for start in self.segment_starts(tuner):
            date = pd.Timestamp(self.market.dates[start]).date()
            cached = self.store.get(key, date)
            if cached is None:
                cached = optimize_params(self.market, params, start, tuner)
                if self.persist:
                    self.store.put(key, date, cached)
                else:
                    self.store.memory[(key, str(date))] = cached
            params = cached
            result.append((start, params))
        return result

    def segments(self, tuner: dict) -> list[dict]:
        key = segment_key(tuner)
        starts = self.segment_params(tuner)
        result = []
        for number, (start, params) in enumerate(starts):
            end = starts[number + 1][0] - 1 if number + 1 < len(starts) else self.last_day
            cache_key = (key, start, end)
            if cache_key not in self.trade_cache:
                view = self.market.view(start - WARM_DAYS, end)
                self.trade_cache[cache_key] = {
                    name: strategy_trades(view, name, params[name]) + view.offset for name in RULES
                }
            result.append({'start': start, 'end': end, 'params': params,
                           'trades': self.trade_cache[cache_key]})
        return result

    def run(self, tuner: dict, correction: dict) -> dict:
        segments = self.segments(tuner)
        first, last = segments[0]['start'], self.last_day
        portfolio = simulate_portfolio(self.market, segments, correction, first, last)
        stats = portfolio['stats']
        stats['SCORE'] = score_of(stats)
        half = len(portfolio['daily']) // 2
        halves = [equity_stats(portfolio['daily'][:half]), equity_stats(portfolio['daily'][half:])]
        return {
            'stats': stats,
            'halves': [{**h, 'SCORE': score_of(h)} for h in halves],
            'blind_start': pd.Timestamp(self.market.dates[first]).date(),
            'data_end': self.data_end,
            'daily': portfolio['daily'],
            'exposure': portfolio['exposure'],
            'dates': self.market.dates[first:last + 1],
            'last_params': segments[-1]['params'],
        }

    def live_params(self, tuner: dict) -> dict:
        """ Пороги для торговли с завтрашнего дня: тот же подбор, окно обучения по текущий день."""
        start = self.segment_params(tuner)[-1][1]
        return optimize_params(self.market, start, self.last_day + 1, tuner)
