""" Живой реверсивный алгоритм: состояние в БД, шаг по каждому новому торговому дню, сигналы.

Состояние жизненного цикла хранится в БД и переживает перезапуски:
    reverse_rules    — пул правил (кандидаты, живые, умершие — тоже опыт);
    reverse_picks    — собственные сигналы (реальные и теневые) с исходами по свечам;
    reverse_journal  — журнал: рождения, переводы в торговлю, смерти, самокоррекция, разбор ИИ;
    algo_params      — reverse (свой пул параметров) и reverse_state (строгость, вес новостей...).

При первом запуске алгоритм «проживает» последние BOOTSTRAP_DAYS дней истории (без
новостей — их архива нет), чтобы начать с опытом, а не с пустого пула.
"""
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from elements.reverse.engine import (
    RMarket, ReverseEngine, Rule, simulate_entries, describe, DEFAULT_REVERSE,
)
from elements.reverse.news import news_features
from elements.selftune.params import load_active, save_active
from source.settings.settings import handler, BLIND_START
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

BOOTSTRAP_DAYS = 250
NOTABLE = ('promote', 'death', 'params', 'mode', 'reflection')


def to_date(value):
    return pd.Timestamp(value).date() if value not in (None, '') else None


class ReverseAlgorithm(SQLmain):

    @staticmethod
    def market(panel: pd.DataFrame) -> RMarket:
        return RMarket(news_features(panel))

    # --- загрузка и сохранение состояния -----------------------------------------------
    @classmethod
    def load_engine(cls, market: RMarket) -> tuple[ReverseEngine, dict]:
        params = load_active('reverse')
        stored = load_active('reverse_state')
        state = {
            **stored,
            'last_mine': cls.day_of(market, stored.get('last_mine')),
            'last_adapt': cls.day_of(market, stored.get('last_adapt')),
        }
        engine = ReverseEngine(market, params, state)
        with Session(bind=main_engine) as s:
            rules = s.execute(sa.select(tables.ReverseRule.__table__)).mappings().all()
            picks = s.execute(sa.select(tables.ReversePick.__table__)).mappings().all()
        last = len(market.dates) - 1
        for row in rules:
            if market.day(row['BORN']) > last:
                continue  # правило из «будущего» относительно данных (история укорочена)
            rule = Rule(json.loads(row['ATOMS']), born=market.day(row['BORN']), origin=row['ORIGIN'],
                        status=row['STATUS'], is_n=row['IS_N'] or 0, is_mean=row['IS_MEAN'] or 0.0,
                        note=row['NOTE'] or '', died=cls.day_of(market, row['DIED']),
                        promoted=cls.day_of(market, row['PROMOTED']))
            rule.db_id = row['id']
            rule.attach(market)
            engine.rules.append(rule)
        index = {(d, s): i for i, (d, s) in enumerate(zip(market.panel['DATE'].dt.date, market.panel['SECID']))}
        for row in picks:
            position = index.get((to_date(row['DATE']), row['SECID']))
            if position is not None:
                engine.picks.append((int(market.didx[position]), position, row['WEIGHT'], row['SCORE']))
        return engine, stored

    @staticmethod
    def day_of(market: RMarket, value) -> int | None:
        return min(market.day(value), len(market.dates) - 1) if value not in (None, '') else None

    @classmethod
    def save_engine(cls, engine: ReverseEngine, day: int, journal_from: int, picks_from: int,
                    rules_text: dict | None = None):
        market = engine.market
        date = market.date
        now = datetime.now()
        state = engine.state(day)
        for key in ('last_mine', 'last_adapt', 'last_day'):
            if state.get(key) is not None:
                state[key] = date(state[key]).isoformat()
        save_active('reverse_state', state, data_end=date(day), note='состояние жизненного цикла')
        with Session(bind=main_engine) as s:
            for rule in engine.rules:
                values = dict(
                    KEY=rule.key, ATOMS=json.dumps(rule.atoms, ensure_ascii=False), ORIGIN=rule.origin,
                    STATUS=rule.status, BORN=date(rule.born),
                    PROMOTED=date(rule.promoted) if rule.promoted is not None else None,
                    DIED=date(rule.died) if rule.died is not None else None,
                    IS_N=rule.is_n, IS_MEAN=rule.is_mean, OOS_N=rule.oos_n, OOS_MEAN=rule.oos_mean,
                    POST=rule.post, NOTE=rule.note[:500], UPDATED_AT=now,
                )
                if getattr(rule, 'db_id', None):
                    s.execute(sa.update(tables.ReverseRule).where(tables.ReverseRule.id == rule.db_id).values(**values))
                else:
                    record = tables.ReverseRule(**values)
                    s.add(record)
                    s.flush()
                    rule.db_id = record.id
            for d, kind, text in engine.journal[journal_from:]:
                s.add(tables.ReverseJournal(DATE=date(d), CREATED_AT=now, KIND=kind, TEXT=text[:2000]))
            s.commit()
        rows = [{
            'DATE': date(d), 'SECID': market.panel['SECID'].iat[row], 'WEIGHT': float(weight),
            'SCORE': float(score), 'REAL': bool(weight > 0),
            'RULES': (rules_text or {}).get(row),
        } for d, row, weight, score in engine.picks[picks_from:]]
        cls.upsert(tables.ReversePick, rows, ['DATE', 'SECID'])

    # --- шаг жизненного цикла ------------------------------------------------------------
    @classmethod
    def step(cls, panel: pd.DataFrame) -> dict:
        """ Прожить все новые торговые дни до последнего закрытого и выбрать бумаги на завтра."""
        market = cls.market(panel)
        last = len(market.dates) - 1
        engine, stored = cls.load_engine(market)
        journal_from, picks_from = len(engine.journal), len(engine.picks)
        processed = cls.day_of(market, stored.get('last_day'))
        if processed is None:
            first = max(engine.p['lookback'] + 5, last - BOOTSTRAP_DAYS)
            logger.info(f'Реверсивный: первый запуск — проживаю историю с {market.date(first)}')
            engine.log(first, 'mode', f'начало жизни: опыт набирается на истории с {market.date(first)}')
            engine.run(first, last - 1)
        elif processed < last:
            engine.run(processed + 1, last - 1)
        if processed is not None and processed >= last:
            engine.update_rules(last)
            chosen = engine.choose(last)
            fresh = []
        else:
            picks_before = len(engine.picks)
            chosen = engine.step(last)
            fresh = engine.picks[picks_before:]
        rules_text = {row: '; '.join(describe(rule.atoms) for rule in rules[:3]) for row, _, rules in chosen}
        cls.save_engine(engine, last, journal_from, picks_from, rules_text)
        notable = [(market.date(d), kind, text) for d, kind, text in engine.journal[journal_from:]
                   if kind in NOTABLE and d >= last - 5]
        return {
            'date': market.date(last), 'engine': engine, 'chosen': chosen, 'fresh': fresh,
            'trading': engine.trading, 'notable': notable,
        }

    @classmethod
    def signals(cls, panel: pd.DataFrame) -> tuple[list[dict], dict]:
        """ [{строка панели, вес, логика}] на следующий день и итог шага."""
        result = cls.step(panel)
        engine, market = result['engine'], result['engine'].market
        weight = engine.slot() if result['trading'] else 0.0
        picks = []
        for row, score, rules in result['chosen']:
            picks.append({
                'ROW': market.panel.iloc[row], 'WEIGHT': weight, 'SCORE': score,
                'LOGIC': 'реверсивный алгоритм — правила, выведенные из опыта прошлых сделок и '
                         'подтверждённые на сделках после своего появления: '
                         + '; '.join(
                             f'«{describe(rule.atoms)}» (вне выборки {rule.oos_n} дн., средняя '
                             f'{rule.oos_mean:+.2%})' for rule in rules[:3]
                         ),
            })
        live = sum(rule.status == 'live' for rule in engine.rules)
        candidates = sum(rule.status == 'candidate' for rule in engine.rules)
        result['summary'] = (
            f'Реверсивный: правил живых {live}, кандидатов {candidates}, строгость {engine.level}, '
            f'вес новостей {engine.news_weight:+.4f}; '
            + (f'{len(picks)} бумаг, позиция {weight:.1%}' if result['trading']
               else f'торговля в тени ({len(picks)} виртуальных бумаг) — своя история пока не в плюсе')
        )
        return picks, result

    # --- отчёты --------------------------------------------------------------------------
    @classmethod
    def blind(cls, panel: pd.DataFrame, params: dict | None = None, start: str = BLIND_START) -> dict:
        """ Слепой прогон механики с нуля: каждое решение — только по прошлому."""
        market = cls.market(panel)
        params = {**DEFAULT_REVERSE, **(params or load_active('reverse'))}
        first_trade = market.day(start)
        first = max(params['lookback'] + 5, first_trade - BOOTSTRAP_DAYS)
        engine = ReverseEngine(market, params).run(first, len(market.dates) - 1)
        last = len(market.dates) - 1
        middle = (first_trade + last) // 2
        result = simulate_entries(market, engine.picks, first_trade, last)
        halves = [simulate_entries(market, engine.picks, first_trade, middle)['stats'],
                  simulate_entries(market, engine.picks, middle + 1, last)['stats']]
        daily = pd.Series(result['daily'], index=pd.to_datetime(market.dates[first_trade:last + 1]))
        yearly = daily.groupby(daily.index.year).apply(lambda x: float((1 + x).prod() - 1))
        signals = [row for d, row, _, _ in engine.picks if d >= first_trade and np.isfinite(market.ret[row])]
        return {
            'start': market.date(first_trade), 'end': market.date(last), 'stats': result['stats'],
            'halves': halves, 'yearly': yearly.to_dict(), 'engine': engine,
            'signal_avg': float(np.mean(market.ret[signals])) if signals else 0.0,
            'signals': len(signals),
        }

    @staticmethod
    def journal(limit: int = 30) -> list[dict]:
        return [dict(row) for row in SQLmain.run_sql_command(
            'SELECT * FROM reverse_journal ORDER BY id DESC LIMIT :limit', {'limit': limit})]

    @staticmethod
    def rules(status: str = 'live') -> list[dict]:
        return [dict(row) for row in SQLmain.run_sql_command(
            'SELECT * FROM reverse_rules WHERE STATUS = :status ORDER BY POST DESC', {'status': status})]
