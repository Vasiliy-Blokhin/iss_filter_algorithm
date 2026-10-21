""" Пул стратегий и их приоритет.

Пул — все стратегии, дающие сигналы: правиловые и model (elements/strategy), реверсивный
(если REVERSE_ENABLED) и топ-EVOLVE_LIVE_STAGES этапов геномного (если EVOLVE_SIGNALS).
Состав обновляется перед каждыми сигналами: сменились этапы геномного или включился
реверсивный — пул меняется, об изменениях пишется в Telegram.

Приоритет — по средней сделке:
  - бэктест (последний прогон стратегии, этапы — из последнего бэктеста), с поправкой на
    малую выборку: avg × n / (n + PRIORITY_BT_K);
  - при PRIORITY_MIN_REAL_TRADES и более закрытых позициях исполнителя (paper и live) —
    смесь с их средней доходностью, вес реальных n / (n + PRIORITY_REAL_K).
Приоритет = 1 + оценка / PRIORITY_SCALE в пределах [PRIORITY_MIN, PRIORITY_MAX].
"""
import logging
from datetime import date, datetime

import sqlalchemy as sa
from sqlalchemy.orm import Session

from elements.strategy.worker import ALL_STRATEGIES
from source.settings.settings import (
    handler,
    EVOLVE_SIGNALS,
    REVERSE_ENABLED,
    PRIORITY_SCALE,
    PRIORITY_MIN,
    PRIORITY_MAX,
    PRIORITY_BT_K,
    PRIORITY_MIN_REAL_TRADES,
    PRIORITY_REAL_K,
)
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def kind_of(name: str) -> str:
    if name.startswith('evolve:'):
        return 'evolve'
    if name in ('model', 'reverse'):
        return name
    return 'rules'


def strategy_parts(label: str | None) -> list[str]:
    """ Составная метка сигнала -> стратегии: 'momentum+reverse', 'evolve:12+15'."""
    parts, evolve = [], False
    for part in (label or '').split('+'):
        if part.startswith('evolve:'):
            evolve = True
            parts.append(part)
        elif part.isdigit() and evolve:
            parts.append(f'evolve:{part}')
        elif part:
            parts.append(part)
    return parts


def score(bt_trades: int, bt_avg: float | None, real_trades: int, real_avg: float | None) -> float:
    """ Оценка средней сделки: бэктест с поправкой на выборку, затем смесь с реальными."""
    value = (bt_avg or 0.0) * bt_trades / (bt_trades + PRIORITY_BT_K) if bt_trades else 0.0
    if real_trades >= PRIORITY_MIN_REAL_TRADES and real_avg is not None:
        weight = real_trades / (real_trades + PRIORITY_REAL_K)
        value = (1 - weight) * value + weight * real_avg
    return value


def priority_of(value: float) -> float:
    return round(min(PRIORITY_MAX, max(PRIORITY_MIN, 1 + value / PRIORITY_SCALE)), 3)


class StrategyPool(SQLmain):

    @staticmethod
    def members() -> list[str]:
        names = list(ALL_STRATEGIES)
        if REVERSE_ENABLED:
            names.append('reverse')
        if EVOLVE_SIGNALS:
            from elements.evolve.live import best_stages
            try:
                names += [f'evolve:{stage["id"]}' for stage in best_stages()]
            except sa.exc.OperationalError:
                pass  # таблицы геномного ещё нет
        return names

    @classmethod
    def backtest_stats(cls) -> dict[str, dict]:
        """ Последний бэктест каждой стратегии; этапы геномного — только из последнего прогона."""
        rows = cls.run_sql_command(
            'SELECT * FROM backtest_runs WHERE id IN (SELECT MAX(id) FROM backtest_runs GROUP BY STRATEGY)')
        return {row['STRATEGY']: dict(row) for row in rows}

    @classmethod
    def real_stats(cls) -> dict[str, tuple[int, float]]:
        """ Закрытые позиции исполнителя по стратегиям: (число, средняя доходность)."""
        try:
            rows = cls.run_sql_command(
                "SELECT STRATEGY, PNL_PCT FROM positions WHERE STATUS = 'closed' AND PNL_PCT IS NOT NULL")
        except sa.exc.OperationalError:
            return {}  # исполнитель ещё не создал таблицу
        values: dict[str, list[float]] = {}
        for row in rows:
            for part in strategy_parts(row['STRATEGY']):
                values.setdefault(part, []).append(float(row['PNL_PCT']))
        return {name: (len(items), sum(items) / len(items)) for name, items in values.items()}

    @classmethod
    def evaluate(cls) -> list[dict]:
        backtest, real = cls.backtest_stats(), cls.real_stats()
        rows = []
        for name in cls.members():
            bt = backtest.get(name, {})
            real_trades, real_avg = real.get(name, (0, None))
            value = score(int(bt.get('TRADES') or 0), bt.get('AVG_TRADE'), real_trades, real_avg)
            rows.append({
                'STRATEGY': name, 'KIND': kind_of(name),
                'BT_TRADES': int(bt.get('TRADES') or 0), 'BT_AVG': bt.get('AVG_TRADE'),
                'BT_SHARPE': bt.get('SHARPE'), 'REAL_TRADES': real_trades, 'REAL_AVG': real_avg,
                'SCORE': value, 'PRIORITY': priority_of(value),
            })
        return rows

    @classmethod
    def previous_members(cls, day: date) -> set[str]:
        rows = cls.run_sql_command(
            'SELECT STRATEGY FROM strategy_pool WHERE DATE = (SELECT MAX(DATE) FROM strategy_pool WHERE DATE < :day)',
            {'day': day})
        return {row['STRATEGY'] for row in rows}

    @classmethod
    def refresh(cls, day: date) -> dict[str, float]:
        """ Пересчитать пул на день, сохранить и сообщить об изменении состава.
        Возвращает {стратегия: приоритет}."""
        rows = cls.evaluate()
        before = cls.previous_members(day)
        now = datetime.now()
        with Session(bind=main_engine) as s:
            s.execute(sa.delete(tables.StrategyPool).where(tables.StrategyPool.DATE == day))
            s.commit()
        cls.upsert(tables.StrategyPool, [{**row, 'DATE': day, 'CREATED_AT': now} for row in rows],
                   ['DATE', 'STRATEGY'])
        names = {row['STRATEGY'] for row in rows}
        added, removed = sorted(names - before), sorted(before - names)
        if before and (added or removed):
            from elements.notification.worker import TelegramNotification as tlg
            text = ('Пул стратегий обновлён: '
                    + (f'добавлены {", ".join(added)}' if added else '')
                    + ('; ' if added and removed else '')
                    + (f'выбыли {", ".join(removed)}' if removed else ''))
            logger.info(text)
            tlg.send_message(text)
        logger.info('Приоритеты стратегий: ' + ', '.join(
            f'{row["STRATEGY"]} {row["PRIORITY"]:g}' for row in sorted(rows, key=lambda r: -r['PRIORITY'])))
        return {row['STRATEGY']: row['PRIORITY'] for row in rows}

    @staticmethod
    def signal_priority(label: str | None, priorities: dict[str, float]) -> float:
        """ Приоритет сигнала — лучший из стратегий, которые его дали; вне пула — 1."""
        known = [priorities[part] for part in strategy_parts(label) if part in priorities]
        return max(known) if known else 1.0
