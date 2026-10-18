""" Формирование сигналов на следующий торговый день."""
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from elements.ai.worker import AIReviewer, fetch_news
from elements.reverse.news import headlines, recent_since
from elements.features.worker import FEATURES, build_dataset
from elements.features.intraday import dividend_features
from elements.selector.worker import StrategySelector
from elements.selftune.params import apply_active_strategy_params
from elements.strategy.worker import STRATEGIES, score_model, select, train_model
from source.settings.settings import (
    handler,
    AI_ENABLED,
    AI_MIN_CONFIDENCE,
    EVOLVE_SIGNALS,
    HOLD_DAYS,
    REVERSE_ENABLED,
    REVERSE_REVIEW,
    STOP_ATR,
    TAKE_ATR,
    TIMEZONE,
)
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class Algorithm(SQLmain):

    @classmethod
    def complete_candles(cls) -> pd.DataFrame:
        """ Свечи без текущего дня, пока не закончилась вечерняя сессия."""
        candles = cls.load_candles()
        if candles.empty:
            raise RuntimeError('Нет истории свечей: выполните `python main.py update`')
        now = datetime.now(ZoneInfo(TIMEZONE))
        if now.hour < 23 or now.minute < 55:
            candles = candles[candles['DATE'] < pd.Timestamp(now.date())]
        return candles

    @classmethod
    def build_panel(cls, candles: pd.DataFrame) -> pd.DataFrame:
        panel = build_dataset(candles)
        dividends = pd.DataFrame(cls.run_sql_command('SELECT * FROM dividends'))
        return dividend_features(panel, dividends)

    @classmethod
    def generate_signals(cls, use_ai: bool = AI_ENABLED) -> list[dict]:
        # Действующие пороги стратегий — из БД (их обновляет самокоррекция).
        apply_active_strategy_params()
        candles = cls.complete_candles()
        panel = cls.build_panel(candles)
        last_date = panel['DATE'].max()

        securities = cls.securities().set_index('SECID')
        sectors = securities['SECTOR'].to_dict()

        # Доли капитала по стратегиям (adaptive, equal, diversify или best — ALLOCATION_MODE).
        allocation, stats = StrategySelector.select_strategy(panel)
        unit = StrategySelector.position_unit()
        rows = panel[panel['DATE'] == last_date]
        merged: dict[str, dict] = {}
        for strategy, share in allocation.items():
            for _, row in cls.strategy_picks(strategy, rows, panel, stats, sectors).iterrows():
                weight = share * unit
                current = merged.get(row['SECID'])
                if current:
                    # Бумагу выбрали несколько стратегий — доли складываются.
                    current['WEIGHT'] += weight
                    current['STRATEGY'] += f'+{strategy}'
                    continue
                merged[row['SECID']] = cls.make_signal(row, securities, sectors, last_date, strategy, weight)
        if REVERSE_ENABLED:
            cls.add_reverse(panel, merged, securities, sectors, last_date)
        if EVOLVE_SIGNALS:
            cls.add_evolve(panel, merged, securities, sectors, last_date)
        logger.info(f'Сигналов на {last_date.date()}: {len(merged)} ({list(merged)})')

        reviewer = AIReviewer() if use_ai and merged else None
        if reviewer and not reviewer.available():
            logger.warning('AI_Manager недоступен — сигналы без ИИ-проверки')
            reviewer = None

        signals = []
        for signal in merged.values():
            if reviewer and (REVERSE_REVIEW or signal['STRATEGY'] != 'reverse'):
                signal.update(cls.ai_review(reviewer, signal, candles))
            signals.append(signal)

        cls.save_signals(signals, last_date.date())
        return signals

    @classmethod
    def add_reverse(cls, panel, merged, securities, sectors, last_date):
        """ Сигналы реверсивного алгоритма (теневые, с весом 0, в сигналы не попадают)."""
        from elements.notification.worker import TelegramNotification as tlg
        from elements.reverse.worker import ReverseAlgorithm
        try:
            picks, result = ReverseAlgorithm.signals(panel)
        except Exception as error:
            logger.exception(f'Реверсивный алгоритм: {error}')
            return
        logger.info(result['summary'])
        notable = '\n'.join(f'{date:%d.%m} {kind}: {text}' for date, kind, text in result['notable'][-8:])
        tlg.send_message(result['summary'] + (f'\n{notable}' if notable else ''))
        for pick in picks:
            if pick['WEIGHT'] <= 0:
                continue
            row = pick['ROW']
            current = merged.get(row['SECID'])
            if current:
                current['WEIGHT'] += pick['WEIGHT']
                current['STRATEGY'] += '+reverse'
                current['LOGIC'] = pick['LOGIC']
                continue
            row = row.copy()
            row['PROBABILITY'] = 0.5
            row['EXPECTED_RETURN'] = pick['SCORE']
            signal = cls.make_signal(row, securities, sectors, last_date, 'reverse', pick['WEIGHT'])
            signal['LOGIC'] = pick['LOGIC']
            merged[row['SECID']] = signal

    @classmethod
    def add_evolve(cls, panel, merged, securities, sectors, last_date):
        """ Сигналы лучших этапов абстрактного алгоритма со своим правилом выхода.
        Бумагу, уже выбранную другой стратегией, не трогаем (у неё другой выход);
        исполнитель берёт сигналы evolve только в paper-режиме."""
        from elements.evolve.live import stage_signals
        try:
            picks = stage_signals(panel)
        except Exception as error:
            logger.exception(f'Абстрактный алгоритм: {error}')
            return
        for pick in picks:
            row, strategy = pick['ROW'], f'evolve:{pick["STAGE"]}'
            current = merged.get(row['SECID'])
            if current:
                if current['STRATEGY'].startswith('evolve:'):
                    current['WEIGHT'] += pick['WEIGHT']
                    current['STRATEGY'] += f'+{pick["STAGE"]}'
                continue
            row = row.copy()
            row['PROBABILITY'] = pick['PROBABILITY']
            row['EXPECTED_RETURN'] = pick['EXPECTED_RETURN']
            signal = cls.make_signal(row, securities, sectors, last_date, strategy, pick['WEIGHT'])
            signal.update({
                'STOP_PRICE': float(row['CLOSE'] - pick['STOP'] * row['ATR']),
                'TAKE_PRICE': float(row['CLOSE'] + pick['TAKE'] * row['ATR']),
                'HOLD_DAYS': pick['HOLD'], 'EXIT_AT_OPEN': pick['AT_OPEN'],
            })
            merged[row['SECID']] = signal

    @staticmethod
    def strategy_picks(strategy, rows, panel, stats, sectors) -> pd.DataFrame:
        """ Отобранные стратегией бумаги дня (с оценками вероятности и ожидания)."""
        last_date = rows['DATE'].max()
        if strategy == 'model':
            model = train_model(panel, last_date + pd.Timedelta(days=1))
            if model is None:
                return rows.iloc[0:0]
            scored = score_model(rows, model)
        else:
            # Для правиловых стратегий оценки — по месячной теневой статистике.
            month = stats[(stats['STRATEGY'] == strategy) & (stats['PERIOD'] == 'month')].iloc[0]
            scored = STRATEGIES[strategy](rows)
            scored['PROBABILITY'] = month['WIN_RATE'] if month['TRADES'] else 0.5
            scored['EXPECTED_RETURN'] = month['AVG_TRADE']
        picks = select(scored, sectors)
        logger.info(f'{strategy}: {len(picks)} бумаг {picks["SECID"].tolist()}')
        return picks

    @staticmethod
    def make_signal(row, securities, sectors, last_date, strategy, weight) -> dict:
        secid = row['SECID']
        info = securities.loc[secid] if secid in securities.index else {}
        return {
            'SECID': secid,
            'SHORTNAME': info.get('SHORTNAME') if len(info) else None,
            'SECTOR': sectors.get(secid, 'unknown'),
            'SIGNAL_DATE': last_date.date(),
            'CREATED_AT': datetime.now(),
            'CLOSE': float(row['CLOSE']),
            'ATR': float(row['ATR']),
            'PROBABILITY': float(row['PROBABILITY']),
            'EXPECTED_RETURN': float(row['EXPECTED_RETURN']),
            # Уровни от закрытия; исполнитель пересчитает их от фактической цены входа.
            'STOP_PRICE': float(row['CLOSE'] - STOP_ATR * row['ATR']),
            'TAKE_PRICE': float(row['CLOSE'] + TAKE_ATR * row['ATR']),
            'HOLD_DAYS': HOLD_DAYS,
            'EXIT_AT_OPEN': None,
            'FEATURES': {name: round(float(row[name]), 4) for name in FEATURES if pd.notna(row[name])},
            'AI_DECISION': 'n/a',
            'AI_CONFIDENCE': None,
            'AI_REASON': None,
            'STATUS': 'approved',
            'STRATEGY': strategy,
            'WEIGHT': weight,
        }

    @staticmethod
    def ai_review(reviewer: AIReviewer, signal: dict, candles: pd.DataFrame) -> dict:
        history = candles[candles['SECID'] == signal['SECID']].tail(10)
        last_candles = [
            [row.DATE.strftime('%d.%m'), row.OPEN, row.HIGH, row.LOW, row.CLOSE,
             round((row.VALUE or 0) / 1e6, 1)]
            for row in history.itertuples()
        ]
        # Оценённые локальной моделью новости о бумаге и рынке; без них — заголовки RSS.
        news = headlines(signal['SECID'], recent_since()) or fetch_news([signal['SECID'], signal.get('SHORTNAME') or ''])
        try:
            verdict = reviewer.review(signal, last_candles, news)
        except Exception as error:
            logger.error(f'ИИ-проверка {signal["SECID"]} не удалась: {error}')
            return {'AI_DECISION': 'error', 'AI_REASON': str(error)[:500]}
        approved = verdict['decision'] == 'buy' or verdict['confidence'] < AI_MIN_CONFIDENCE
        logger.info(f'ИИ {signal["SECID"]}: {verdict}')
        return {
            'AI_DECISION': verdict['decision'],
            'AI_CONFIDENCE': verdict['confidence'],
            'AI_REASON': verdict['reason'],
            'STATUS': 'approved' if approved else 'rejected',
        }

    @classmethod
    def save_signals(cls, signals: list[dict], signal_date):
        with Session(bind=main_engine) as s:
            # Неисполненные сигналы прошлых дней больше не актуальны.
            s.execute(
                sa.update(tables.Signal)
                .where(tables.Signal.SIGNAL_DATE < signal_date)
                .where(tables.Signal.STATUS.in_(['new', 'approved']))
                .values(STATUS='expired')
            )
            s.commit()
        rows = [
            {**{k: v for k, v in signal.items() if k != 'LOGIC'}, 'FEATURES': json.dumps(signal['FEATURES'])}
            for signal in signals
        ]
        cls.upsert(tables.Signal, rows, ['SECID', 'SIGNAL_DATE'])
