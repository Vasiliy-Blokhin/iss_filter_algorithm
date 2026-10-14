""" Ротация «акции / короткие облигации / деньги» — основа портфеля.

Раз в месяц, по данным на последний торговый день месяца:
  оценка актива = средний рост его индекса за ROTATION_LOOKBACKS торговых дней;
  выбирается актив с лучшей оценкой; если он не лучше денег — деньги;
  доля акций не больше ROTATION_EQ_CAP, остаток — фонд денежного рынка.
Сигналы считаются по индексам (MCFTR — акции с дивидендами, RUGBITR3Y — ОФЗ до 3 лет,
RUSFAR — ставка денег), покупаются биржевые фонды (EQMX / SBRB / LQDT), которые можно продать
в любой торговый день.

Исследование 13.10.2026 (2021-01..2026-10, с комиссиями фондов и издержками): вариант с долей
акций до 70% — 14.7…19.2% годовых в зависимости от окон и дня ребалансировки (медиана 15.9%),
макс. просадка ~-8%, ни одного 12-месячного периода в минусе; см. README.
"""
import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from source.iss.client import ISSClient
from source.settings.settings import (
    handler,
    HISTORY_DAYS,
    ROTATION_ASSETS,
    ROTATION_LOOKBACKS,
    ROTATION_EQ_CAP,
    ROTATION_CASH_FEE,
    ROTATION_FUND_FEE,
    ROTATION_COST,
)
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

ASSETS = list(ROTATION_ASSETS)  # EQ, BOND, CASH


class Rotation(SQLmain):

    # --- данные ---------------------------------------------------------------
    @classmethod
    def update_indexes(cls, client: ISSClient | None = None) -> int:
        """ Докачивает индексы ротации (с последней даты, при пустой таблице — HISTORY_DAYS)."""
        client = client or ISSClient()
        with Session(bind=main_engine) as s:
            last = dict(s.execute(
                sa.select(tables.IndexCandle.SECID, sa.func.max(tables.IndexCandle.DATE))
                .group_by(tables.IndexCandle.SECID)
            ).all())
        saved = 0
        for index, _ in ROTATION_ASSETS.values():
            start = last[index] - timedelta(days=5) if index in last else \
                date.today() - timedelta(days=HISTORY_DAYS)
            try:
                frame = client.candles(index, start, index=True)
            except ConnectionError as error:
                logger.error(f'{index}: {error}')
                continue
            if frame.empty:
                continue
            rows = [{'SECID': index, 'DATE': row.date.date(), 'CLOSE': float(row.close)}
                    for row in frame.itertuples()]
            cls.upsert(tables.IndexCandle, rows, ['SECID', 'DATE'])
            saved += len(rows)
        logger.info(f'Индексы ротации обновлены: {saved} значений')
        return saved

    @staticmethod
    def load_prices() -> pd.DataFrame:
        """ Цены активов по рабочим дням: индексы акций и облигаций, «деньги» — накопленная ставка."""
        with main_engine.connect() as connection:
            frame = pd.read_sql(sa.select(tables.IndexCandle.__table__), connection)
        if frame.empty:
            return pd.DataFrame(columns=ASSETS)
        frame['DATE'] = pd.to_datetime(frame['DATE'])
        wide = frame.pivot(index='DATE', columns='SECID', values='CLOSE').sort_index()
        return prices_from_indexes(wide)

    # --- решение --------------------------------------------------------------
    @classmethod
    def update_target(cls, today: date | None = None) -> dict | None:
        """ Целевые доли на текущий месяц по данным на последний торговый день прошлого месяца.
        Возвращает новую цель или None, если она уже рассчитана (или нет данных)."""
        today = today or date.today()
        prices = cls.load_prices()
        month_start = pd.Timestamp(today.replace(day=1))
        history = prices[prices.index < month_start]
        if len(history) < max(ROTATION_LOOKBACKS) + 1:
            logger.warning('Ротация: мало истории индексов')
            return None
        decision = history.index[-1].date()
        exists = cls.run_sql_command(
            'SELECT COUNT(*) AS n FROM allocation_targets WHERE "DATE" = :d', {'d': decision}
        )[0]['n']
        if exists:
            return None
        weights, scores = target_weights(history)
        reason = 'оценки (средний рост за {} дн.): '.format('/'.join(map(str, ROTATION_LOOKBACKS))) + \
            ', '.join(f'{asset} {score:+.2%}' for asset, score in scores.items()) + \
            f'; доля акций не больше {ROTATION_EQ_CAP:.0%}'
        rows = [{
            'DATE': decision, 'ASSET': asset, 'SECID': ROTATION_ASSETS[asset][1],
            'WEIGHT': float(weight), 'REASON': reason, 'CREATED_AT': datetime.now(),
        } for asset, weight in weights.items()]
        cls.upsert(tables.AllocationTarget, rows, ['DATE', 'SECID'])
        target = {ROTATION_ASSETS[a][1]: w for a, w in weights.items() if w > 0}
        logger.info(f'Ротация на месяц после {decision}: {target} ({reason})')
        return {'date': decision, 'target': target, 'reason': reason}

    @classmethod
    def current_target(cls) -> list[dict]:
        rows = cls.run_sql_command(
            'SELECT * FROM allocation_targets WHERE "DATE" = (SELECT MAX("DATE") FROM allocation_targets)'
        )
        return [dict(row) for row in rows]


def prices_from_indexes(wide: pd.DataFrame) -> pd.DataFrame:
    """ MCFTR/RUGBITR3Y — как есть; RUSFAR (% годовых) -> накопленная стоимость денег за вычетом
    комиссии фонда денежного рынка (по календарным дням)."""
    eq_index, bond_index, cash_index = (ROTATION_ASSETS[a][0] for a in ASSETS)
    wide = wide[wide.index.weekday < 5].ffill()
    rate = wide[cash_index]
    days = wide.index.to_series().diff().dt.days.fillna(0)
    cash = (1 + (rate.shift(1).fillna(rate) / 100 - ROTATION_CASH_FEE) * days / 365).cumprod()
    return pd.DataFrame({'EQ': wide[eq_index], 'BOND': wide[bond_index], 'CASH': cash}).dropna()


def target_weights(history: pd.DataFrame) -> tuple[dict, dict]:
    """ Доли по истории цен до дня решения включительно."""
    scores = {
        asset: float(np.mean([history[asset].iloc[-1] / history[asset].iloc[-1 - lb] - 1
                              for lb in ROTATION_LOOKBACKS]))
        for asset in ASSETS
    }
    best = max(scores, key=scores.get)
    if scores[best] <= scores['CASH']:
        best = 'CASH'
    weights = {asset: 0.0 for asset in ASSETS}
    weights[best] = 1.0
    if best == 'EQ':
        weights['EQ'], weights['CASH'] = ROTATION_EQ_CAP, 1 - ROTATION_EQ_CAP
    return weights, scores


def backtest(prices: pd.DataFrame, start: str = '2021-01-01') -> dict:
    """ Ребалансировка в последний торговый день месяца, доли действуют со следующего дня."""
    returns = prices.pct_change().fillna(0)
    month = prices.index.to_period('M')
    last_day = prices.index.to_series().groupby(month).transform('max') == prices.index.to_series()
    weights = pd.DataFrame(np.nan, index=prices.index, columns=ASSETS)
    for day in prices.index[last_day.to_numpy()]:
        history = prices.loc[:day]
        if len(history) > max(ROTATION_LOOKBACKS):
            weights.loc[day] = pd.Series(target_weights(history)[0])
    weights = weights.ffill().fillna(0).shift(1).fillna(0)
    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    fees = (weights['EQ'] + weights['BOND']) * ROTATION_FUND_FEE / 252
    daily = (weights * returns).sum(axis=1) - turnover * ROTATION_COST - fees
    daily = daily[daily.index >= start]
    equity = (1 + daily).cumprod()
    yearly = (1 + daily).groupby(daily.index.year).prod() - 1
    worst_year = (equity.shift(-252) / equity - 1).min()
    return {
        'CAGR': float(equity.iloc[-1] ** (252 / len(daily)) - 1),
        'MAX_DRAWDOWN': float((equity / equity.cummax() - 1).min()),
        'WORST_12M': float(worst_year),
        'YEARLY': {int(k): float(v) for k, v in yearly.items()},
        'daily': daily, 'weights': weights,
    }
