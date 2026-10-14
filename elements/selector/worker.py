""" Теневая статистика стратегий и выбор лучшей для текущей работы.

Каждая стратегия каждый день «торгует» виртуально на реальных свечах по тем же правилам,
что и исполнитель (вход по открытию t+1, стоп/тейк по ATR, выход по времени, издержки).
Сделки относятся к периоду по дате выхода.

Периоды (торговых дней, считая последний закрытый день): день — 1, неделя — 5, месяц — 21.
Результат периода — вклад сделок в капитал (капитал делится на MAX_SIGNALS * HOLD_DAYS слотов),
то есть учитывается и качество, и количество сделок.

Режим best: лучшая по неделе; если разница с лидером меньше SELECT_TIE — решает месяц,
затем день. Если у выбранной и неделя, и месяц не в плюсе, а REQUIRE_POSITIVE включён —
торговля на следующий день не ведётся.

Режим diversify: капитал делится между стратегиями с положительным месячным результатом
пропорционально этому результату.

Режим adaptive (по умолчанию): коррекция по текущей статистике — вес каждой правиловой
стратегии по её теневым сделкам, закрытым за последние lookback дней (срезать «остывшие»,
усилить «в форме»), позиция = slot * вес; параметры — в БД (algo_params, вид correction),
их подбирает самокоррекция по слепому бэктесту (elements/selftune).

Режим equal: капитал поровну между правиловыми стратегиями, без выбора по
недавним результатам — лидерство стратегий почти не сохраняется. На 2020-2026 это дало
+10.5% (6 лет из 7 в плюсе) против +11.4% у diversify и +6.1% у best, но с меньшими
колебаниями: за последний год +1.3% против -1.5% и -2.3%; см. README.
"""
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from elements.backtest.worker import Backtest
from elements.selftune.params import load_active
from elements.selftune.simulator import correction_weight, trade_history
from elements.strategy.worker import STRATEGIES, ALL_STRATEGIES, walk_forward_scores
from source.settings.settings import (
    handler,
    HOLD_DAYS,
    MAX_SIGNALS,
    SELECT_TIE,
    REQUIRE_POSITIVE,
    ALLOCATION_MODE,
)
from source.settings.sectors import SECTORS
from source.sql.main import SQLmain
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

PERIODS = {'day': 1, 'week': 5, 'month': 21}
PRIORITY = ['week', 'month', 'day']


class StrategySelector(SQLmain):

    @staticmethod
    def shadow_trades(panel: pd.DataFrame, lookback: int = 45) -> dict[str, pd.DataFrame]:
        """ Виртуальные сделки всех стратегий за последние lookback торговых дней."""
        dates = np.sort(panel['DATE'].unique())
        start = pd.Timestamp(dates[max(0, len(dates) - lookback)])
        window = panel[panel['DATE'] >= start]
        scored = {'model': walk_forward_scores(panel, start)}
        for name, scorer in STRATEGIES.items():
            scored[name] = scorer(window)
        return {
            name: Backtest.trades(frame, SECTORS) if not frame.empty else pd.DataFrame()
            for name, frame in scored.items()
        }

    @staticmethod
    def period_stats(trades: pd.DataFrame, dates: np.ndarray, as_of: pd.Timestamp) -> dict:
        """ Статистика сделок, закрытых в каждом из периодов, заканчивающихся as_of."""
        weight = 1 / (MAX_SIGNALS * HOLD_DAYS)
        position = np.searchsorted(dates, np.datetime64(as_of), side='right')
        result = {}
        for period, length in PERIODS.items():
            start = pd.Timestamp(dates[max(0, position - length)])
            if trades.empty:
                closed = pd.DataFrame(columns=['TRADE_RET'])
            else:
                closed = trades[(trades['EXIT_DATE'] >= start) & (trades['EXIT_DATE'] <= as_of)]
            returns = closed['TRADE_RET'].astype(float)
            result[period] = {
                'RETURN': float(returns.sum() * weight),
                'TRADES': int(len(returns)),
                'WIN_RATE': float((returns > 0).mean()) if len(returns) else 0.0,
                'AVG_TRADE': float(returns.mean()) if len(returns) else 0.0,
            }
        return result

    @classmethod
    def compute(cls, panel: pd.DataFrame, trades: dict | None = None) -> pd.DataFrame:
        """ Статистика всех стратегий на последний день с известными исходами сделок."""
        trades = trades if trades is not None else cls.shadow_trades(panel)
        dates = np.sort(panel['DATE'].unique())
        as_of = pd.Timestamp(dates[-1])
        rows = []
        for name in ALL_STRATEGIES:
            stats = cls.period_stats(trades.get(name, pd.DataFrame()), dates, as_of)
            for period, values in stats.items():
                rows.append({'DATE': as_of.date(), 'STRATEGY': name, 'PERIOD': period, **values})
        frame = pd.DataFrame(rows)
        cls.upsert(tables.StrategyStat, frame.to_dict('records'), ['DATE', 'STRATEGY', 'PERIOD'])
        return frame

    @classmethod
    def backfill(cls, panel: pd.DataFrame, days: int = 60) -> pd.DataFrame:
        """ Статистика на каждый из последних days дней (история для отчётов)."""
        trades = cls.shadow_trades(panel, lookback=days + 30)
        dates = np.sort(panel['DATE'].unique())
        rows = []
        for as_of in dates[-days:]:
            as_of = pd.Timestamp(as_of)
            for name in ALL_STRATEGIES:
                stats = cls.period_stats(trades.get(name, pd.DataFrame()), dates, as_of)
                for period, values in stats.items():
                    rows.append({'DATE': as_of.date(), 'STRATEGY': name, 'PERIOD': period, **values})
        cls.upsert(tables.StrategyStat, rows, ['DATE', 'STRATEGY', 'PERIOD'])
        return pd.DataFrame(rows)

    @classmethod
    def evaluate(cls, panel: pd.DataFrame, days: int = 250) -> dict:
        """ Мета-бэктест распределения: каждый день доли считаются по статистике на этот день,
        сделки стратегий по сигналам этого дня входят в портфель со своей долей."""
        trades = cls.shadow_trades(panel, lookback=days + 30)
        dates = np.sort(panel['DATE'].unique())
        test_dates = pd.Series(dates[-days:])
        result = {}
        for mode in ('equal', 'diversify', 'best'):
            taken, used = [], []
            for as_of in test_dates:
                as_of = pd.Timestamp(as_of)
                rows = [
                    {'STRATEGY': name, 'PERIOD': period, **values}
                    for name in ALL_STRATEGIES
                    for period, values in cls.period_stats(
                        trades.get(name, pd.DataFrame()), dates, as_of
                    ).items()
                ]
                allocation, _ = cls.allocate(pd.DataFrame(rows), mode)
                used.append(sum(allocation.values()))
                for name, share in allocation.items():
                    own = trades.get(name, pd.DataFrame())
                    if not own.empty:
                        own = own[own['DATE'] == as_of].copy()
                        # Доля стратегии масштабирует вклад сделки в капитал.
                        own['TRADE_RET'] = own['TRADE_RET'] * share
                        taken.append(own)
            portfolio = pd.concat(taken, ignore_index=True) if taken else pd.DataFrame()
            stats = Backtest.metrics(portfolio, test_dates)
            stats['CAPITAL_USED'] = float(np.mean(used))
            result[mode] = stats
        for name in ALL_STRATEGIES:
            own = trades.get(name, pd.DataFrame())
            own = own[own['DATE'] >= test_dates.iloc[0]] if not own.empty else own
            result[name] = Backtest.metrics(own, test_dates)
        return result

    @staticmethod
    def choose(stats: pd.DataFrame, tie: float = SELECT_TIE) -> tuple[str | None, str]:
        """ Неделя -> (при близких значениях) месяц -> день."""
        table = stats.pivot(index='STRATEGY', columns='PERIOD', values='RETURN')
        candidates = table
        steps = []
        for period in PRIORITY:
            best = candidates[period].max()
            candidates = candidates[candidates[period] >= best - tie]
            steps.append(f'{period}: лидер {best:+.2%}, в пределах {tie:.1%} — {len(candidates)}')
            if len(candidates) == 1:
                break
        # Полная ничья (например, у всех нет сделок) — по среднему месячному результату сделки.
        if len(candidates) > 1:
            trades = stats[(stats['PERIOD'] == 'month')].set_index('STRATEGY')
            candidates = candidates.loc[[trades.loc[candidates.index, 'AVG_TRADE'].idxmax()]]
        chosen = candidates.index[0]
        reason = '; '.join(steps)
        row = table.loc[chosen]
        if REQUIRE_POSITIVE and row['week'] <= 0 and row['month'] <= 0:
            return None, (
                f'лучшая — {chosen} (неделя {row["week"]:+.2%}, месяц {row["month"]:+.2%}), '
                'но обе в минусе или без сделок: торговля приостановлена'
            )
        return chosen, (
            f'{chosen}: неделя {row["week"]:+.2%}, месяц {row["month"]:+.2%}, '
            f'день {row["day"]:+.2%} ({reason})'
        )

    @staticmethod
    def diversify(stats: pd.DataFrame) -> tuple[dict, str]:
        """ Доли капитала пропорционально месячному результату; минусовые исключаются."""
        month = stats[stats['PERIOD'] == 'month'].set_index('STRATEGY')['RETURN']
        positive = month[month > 0]
        if positive.empty:
            return {}, 'у всех стратегий месячный результат не в плюсе: торговля приостановлена'
        shares = (positive / positive.sum()).round(4).sort_values(ascending=False)
        excluded = ', '.join(month[month <= 0].index)
        reason = 'доли по месячному результату: ' + ', '.join(
            f'{name} {share:.0%} (месяц {month[name]:+.2%})' for name, share in shares.items()
        ) + (f'; исключены: {excluded}' if excluded else '')
        return shares.to_dict(), reason

    @staticmethod
    def adaptive(trades: dict, dates: np.ndarray, correction: dict) -> tuple[dict, str]:
        """ Веса правиловых стратегий по их теневым сделкам, закрытым к последнему дню."""
        day = len(dates) - 1
        weights = {}
        for name in STRATEGIES:
            own = trades.get(name, pd.DataFrame())
            if own.empty:
                exits, returns = np.array([], dtype=int), np.array([])
            else:
                exits = np.searchsorted(dates, pd.to_datetime(own['EXIT_DATE']).to_numpy())
                returns = own['TRADE_RET'].to_numpy(dtype=float)
            weights[name] = correction_weight(*trade_history(exits, returns), day, correction)
        allocation = {name: w for name, w in weights.items() if w > 0}
        reason = 'коррекция по статистике за {lookback} дн.: '.format(**correction) + ', '.join(
            f'{name} {w:g}' for name, w in weights.items()
        ) + f'; позиция = {correction["slot"]:.0%} капитала x вес'
        if not allocation:
            reason += ' — все стратегии остыли: торговля приостановлена'
        return allocation, reason

    @staticmethod
    def position_unit(mode: str = ALLOCATION_MODE) -> float:
        """ Доля капитала на сделку при доле (весе) стратегии 1."""
        if mode == 'adaptive':
            return float(load_active('correction')['slot'])
        return 1 / (MAX_SIGNALS * HOLD_DAYS)

    @staticmethod
    def equal() -> tuple[dict, str]:
        """ Поровну между правиловыми стратегиями (модель без преимущества не торгует)."""
        share = round(1 / len(STRATEGIES), 4)
        return {name: share for name in STRATEGIES}, f'равные доли {share:.0%} между {len(STRATEGIES)} стратегиями'

    @classmethod
    def allocate(cls, stats: pd.DataFrame, mode: str = ALLOCATION_MODE) -> tuple[dict, str]:
        """ {стратегия: доля капитала} и текстовое объяснение."""
        if mode == 'equal':
            return cls.equal()
        if mode == 'best':
            chosen, reason = cls.choose(stats)
            return ({chosen: 1.0} if chosen else {}), reason
        return cls.diversify(stats)

    @classmethod
    def select_strategy(cls, panel: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        if ALLOCATION_MODE == 'adaptive':
            correction = load_active('correction')
            trades = cls.shadow_trades(panel, lookback=max(45, int(correction['lookback']) + 10))
            stats = cls.compute(panel, trades)
            allocation, reason = cls.adaptive(trades, np.sort(panel['DATE'].unique()), correction)
        else:
            stats = cls.compute(panel)
            allocation, reason = cls.allocate(stats)
        as_of = stats['DATE'].iloc[0]
        label = (
            None if not allocation
            else next(iter(allocation)) if len(allocation) == 1 else ALLOCATION_MODE
        )
        cls.upsert(tables.StrategySelection, [{
            'DATE': as_of, 'STRATEGY': label, 'REASON': reason,
            'ALLOCATION': json.dumps(allocation), 'CREATED_AT': datetime.now(),
        }], ['DATE'])
        logger.info(f'Распределение на {as_of}: {reason}')
        return allocation, stats
