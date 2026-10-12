""" Формирование сигналов на следующий торговый день."""
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from elements.ai.worker import AIReviewer, fetch_news
from elements.features.worker import FEATURES, build_dataset
from elements.features.intraday import dividend_features
from elements.strategy.worker import score_model, select, train_model
from source.settings.settings import (
    handler,
    AI_ENABLED,
    AI_MIN_CONFIDENCE,
    HOLD_DAYS,
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
    def generate_signals(cls, use_ai: bool = AI_ENABLED) -> list[dict]:
        candles = cls.load_candles()
        if candles.empty:
            raise RuntimeError('Нет истории свечей: выполните `python main.py update`')
        # Свеча текущего дня неполная, пока не закончилась вечерняя сессия.
        now = datetime.now(ZoneInfo(TIMEZONE))
        if now.hour < 23 or now.minute < 55:
            candles = candles[candles['DATE'] < pd.Timestamp(now.date())]
        panel = build_dataset(candles)
        last_date = panel['DATE'].max()
        dividends = pd.DataFrame(cls.run_sql_command('SELECT * FROM dividends'))
        panel = dividend_features(panel, dividends)

        model = train_model(panel, last_date + pd.Timedelta(days=1))
        if model is None:
            raise RuntimeError('Недостаточно истории для обучения модели')
        logger.info(f'Важность признаков: {list(model.importance())[:8]}')

        securities = cls.securities().set_index('SECID')
        sectors = securities['SECTOR'].to_dict()
        today = score_model(panel[panel['DATE'] == last_date], model)
        chosen = select(today, sectors)
        logger.info(f'Кандидатов на {last_date.date()}: {len(chosen)}')

        reviewer = AIReviewer() if use_ai else None
        if reviewer and not reviewer.available():
            logger.warning('AI_Manager недоступен — сигналы без ИИ-проверки')
            reviewer = None

        signals = []
        for _, row in chosen.iterrows():
            secid = row['SECID']
            info = securities.loc[secid] if secid in securities.index else {}
            features = {name: round(float(row[name]), 4) for name in FEATURES if pd.notna(row[name])}
            signal = {
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
                'FEATURES': features,
                'AI_DECISION': 'n/a',
                'AI_CONFIDENCE': None,
                'AI_REASON': None,
                'STATUS': 'approved',
            }
            if reviewer:
                signal.update(cls.ai_review(reviewer, signal, candles))
            signals.append(signal)

        cls.save_signals(signals, last_date.date())
        return signals

    @staticmethod
    def ai_review(reviewer: AIReviewer, signal: dict, candles: pd.DataFrame) -> dict:
        history = candles[candles['SECID'] == signal['SECID']].tail(10)
        last_candles = [
            [row.DATE.strftime('%d.%m'), row.OPEN, row.HIGH, row.LOW, row.CLOSE,
             round((row.VALUE or 0) / 1e6, 1)]
            for row in history.itertuples()
        ]
        news = fetch_news([signal['SECID'], signal.get('SHORTNAME') or ''])
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
        rows = [{**signal, 'FEATURES': json.dumps(signal['FEATURES'])} for signal in signals]
        cls.upsert(tables.Signal, rows, ['SECID', 'SIGNAL_DATE'])
