""" Дивидендный разгон в доле акций ротации.

Исследование data/dividend_study (25.10.2026; подбор 2020–2023, слепо 2024–2026): бумаги
с объявленным дивидендом ≥ 4% за месяц до последнего дня с правом на дивиденд (L) обгоняют
MCFTR — слепо +2.7% за сделку, плюс в каждом году; после отсечки 5–10 дней отстают.

Правило: пока ротация держит фонд акций, часть его доли переводится в такие бумаги.
Решение по закрытию дня t: дивиденд объявлен (ANNOUNCE_DATE ≤ t), до L от DIVIDEND_LATEST_DAYS
до DIVIDEND_ENTRY_DAYS будней, доходность ≥ DIVIDEND_MIN_YIELD, оборот ≥ DIVIDEND_MIN_TURNOVER.
Покупка — утром t+1, продажа — утром дня L (дивиденд не получаем: гэп и отставание после
отсечки обходим). Одновременно не больше DIVIDEND_SLOTS бумаг, каждой — 1/DIVIDEND_SLOTS
доли фонда акций; места занимают в порядке решений, при равенстве — бо́льшая доходность.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import sqlalchemy as sa

from elements.features.intraday import last_cum_date
from source.settings.settings import (
    handler, COMMISSION, SLIPPAGE, ROTATION_COST,
    DIVIDEND_ENTRY_DAYS, DIVIDEND_LATEST_DAYS, DIVIDEND_MIN_YIELD, DIVIDEND_MIN_TURNOVER, DIVIDEND_SLOTS,
)
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

EMPTY = '-'
TURNOVER_DAYS = 60


def load_market(start: str = '2019-01-01') -> dict[str, pd.DataFrame]:
    """ Цены открытия/закрытия и медианный оборот бумаг по будням (торги выходного дня не берутся)."""
    with main_engine.connect() as connection:
        frame = pd.read_sql(sa.text(
            'SELECT SECID, "DATE", "OPEN", "CLOSE", VALUE FROM candles WHERE "DATE" >= :start'
        ), connection, params={'start': start}, parse_dates=['DATE'])
    frame = frame[frame['DATE'].dt.dayofweek < 5]
    wide = {column: frame.pivot(index='DATE', columns='SECID', values=column).sort_index()
            for column in ('OPEN', 'CLOSE', 'VALUE')}
    turnover = wide['VALUE'].fillna(0).rolling(TURNOVER_DAYS, min_periods=20).median()
    return {'open': wide['OPEN'], 'close': wide['CLOSE'], 'turnover': turnover}


def load_dividends() -> pd.DataFrame:
    with main_engine.connect() as connection:
        frame = pd.read_sql(sa.text('SELECT SECID, REGISTRY_DATE, ANNOUNCE_DATE, VALUE FROM dividends'),
                            connection, parse_dates=['REGISTRY_DATE', 'ANNOUNCE_DATE'])
    return frame


def plan_events(close: pd.DataFrame, turnover: pd.DataFrame, dividends: pd.DataFrame,
                entry_days: int = DIVIDEND_ENTRY_DAYS, latest_days: int = DIVIDEND_LATEST_DAYS,
                min_yield: float = DIVIDEND_MIN_YIELD, min_turnover: float = DIVIDEND_MIN_TURNOVER) -> pd.DataFrame:
    """ События разгона: SECID, SIGNAL (день решения), LAST_CUM, YIELD — по одному на дивиденд,
    первый день окна, в который выполнены все условия (только данные по этот день)."""
    columns = ['SECID', 'SIGNAL', 'LAST_CUM', 'YIELD']
    dividends = dividends.dropna(subset=['ANNOUNCE_DATE'])
    if dividends.empty or close.empty:
        return pd.DataFrame(columns=columns)
    dividends = dividends.assign(LAST_CUM=pd.to_datetime(last_cum_date(dividends['REGISTRY_DATE'])))
    dates = close.index
    day64 = dates.to_numpy().astype('datetime64[D]')
    rows = []
    for div in dividends.itertuples():
        if div.SECID not in close:
            continue
        ahead = np.busday_count(day64, np.datetime64(div.LAST_CUM, 'D'))
        window = np.nonzero((ahead >= latest_days) & (ahead <= entry_days)
                            & np.asarray(dates >= div.ANNOUNCE_DATE))[0]
        for i in window:
            price, turn = close[div.SECID].iloc[i], turnover[div.SECID].iloc[i]
            if not (price > 0) or not (turn >= min_turnover):
                continue
            if div.VALUE / price >= min_yield:
                rows.append({'SECID': div.SECID, 'SIGNAL': dates[i], 'LAST_CUM': div.LAST_CUM,
                             'YIELD': float(div.VALUE / price)})
                break
    events = pd.DataFrame(rows, columns=columns)
    return events.sort_values(['SIGNAL', 'YIELD'], ascending=[True, False]).reset_index(drop=True)


def active_events(events: pd.DataFrame, day, slots: int = DIVIDEND_SLOTS) -> pd.DataFrame:
    """ Бумаги, которые держатся в день day: решение раньше day, day раньше LAST_CUM (утром L —
    продажа); не больше slots, по порядку решений (порядок plan_events)."""
    day = pd.Timestamp(day)
    held = events[(events['SIGNAL'] < day) & (day < events['LAST_CUM'])]
    return held.drop_duplicates('SECID').head(slots)


def overlay_daily(events: pd.DataFrame, opens: pd.DataFrame, closes: pd.DataFrame, turnover: pd.DataFrame,
                  eq_weight: pd.Series, eq_returns: pd.Series, slots: int = DIVIDEND_SLOTS) -> pd.Series:
    """ Добавка к дневной доходности капитала ротации от замены части фонда акций бумагами.

    eq_weight — доля фонда акций в капитале ротации в этот день, eq_returns — его доходность
    (MCFTR). В день входа бумага берётся по открытию (её доход — от открытия до закрытия),
    в день L продаётся по открытию (доход — ночной гэп); фонд за ночь считается неподвижным.
    Издержки: бумага — комиссия + проскальзывание по ликвидности, фонд — ROTATION_COST, на сторону."""
    dates = eq_weight.index
    extra = pd.Series(0.0, index=dates)
    if events.empty:
        return extra
    held_before: set[str] = set()
    for n, day in enumerate(dates):
        held = set(active_events(events, day, slots)['SECID'])
        weight = float(eq_weight.iloc[n]) / slots
        if weight <= 0:
            held_before = set()
            continue
        total = 0.0
        for secid in held | held_before:
            if secid not in closes:
                continue
            close, opn = closes[secid].iloc[n], opens[secid].iloc[n]
            prev = closes[secid].iloc[n - 1] if n else np.nan
            cost = COMMISSION + SLIPPAGE + liquidity_slippage(turnover[secid].iloc[n]) + ROTATION_COST
            if not np.isfinite(close) or not np.isfinite(opn):
                continue  # нет торгов — день без движения
            if secid in held and secid not in held_before:      # вход по открытию
                total += close / opn - 1 - float(eq_returns.iloc[n]) - cost
            elif secid in held:                                  # держим весь день
                total += close / prev - 1 - float(eq_returns.iloc[n]) if np.isfinite(prev) else 0.0
            else:                                                # выход по открытию
                total += (opn / prev - 1 if np.isfinite(prev) else 0.0) - cost
        extra.iloc[n] = weight * total
        held_before = held
    return extra


def liquidity_slippage(turnover: float) -> float:
    from elements.regime.realistic import liquidity_slippage as slip
    return slip(turnover)


class DividendOverlay(SQLmain):

    @classmethod
    def update(cls) -> dict:
        """ Набор бумаг на следующий торговый день по данным на последнее закрытие; пишет
        dividend_overlay. Возвращает {'date', 'held', 'added', 'removed'}."""
        market = load_market((pd.Timestamp.today() - pd.Timedelta(days=200)).strftime('%Y-%m-%d'))
        events = plan_events(market['close'], market['turnover'], load_dividends())
        data_day = market['close'].index[-1]
        next_day = pd.Timestamp(np.busday_offset(np.datetime64(data_day, 'D'), 1, roll='forward'))
        held = active_events(events, next_day)
        previous = cls.current()
        now = datetime.now()
        rows = [{'DATE': data_day.date(), 'SECID': e.SECID, 'SHARE': 1 / DIVIDEND_SLOTS,
                 'LAST_CUM': e.LAST_CUM.date(), 'YIELD': e.YIELD, 'CREATED_AT': now}
                for e in held.itertuples()]
        if not rows:
            rows = [{'DATE': data_day.date(), 'SECID': EMPTY, 'SHARE': 0.0, 'LAST_CUM': None,
                     'YIELD': None, 'CREATED_AT': now}]
        cls.upsert(tables.DividendOverlay, rows, ['DATE', 'SECID'])
        names = set(held['SECID'])
        result = {'date': data_day.date(), 'held': held, 'added': sorted(names - set(previous)),
                  'removed': sorted(set(previous) - names)}
        logger.info(f'Дивидендный разгон на {next_day.date()}: {sorted(names) or "нет бумаг"}')
        return result

    @classmethod
    def current(cls) -> dict[str, dict]:
        """ Последний записанный набор: SECID -> строка (без пустого маркера)."""
        rows = cls.run_sql_command(
            'SELECT * FROM dividend_overlay WHERE "DATE" = (SELECT MAX("DATE") FROM dividend_overlay)')
        return {r['SECID']: dict(r) for r in rows if r['SECID'] != EMPTY}


def describe(result: dict) -> str:
    """ Текст для Telegram: что добавилось и выбыло (пусто — без изменений)."""
    if not result['added'] and not result['removed']:
        return ''
    held = result['held'].set_index('SECID')
    parts = [f'{s} (дивиденд {held.at[s, "YIELD"]:.1%}, продажа {held.at[s, "LAST_CUM"]:%d.%m})'
             for s in result['added']]
    text = 'Дивидендный разгон: '
    if parts:
        text += 'покупка вместо фонда акций — ' + ', '.join(parts)
    if result['removed']:
        text += ('; ' if parts else '') + 'продажа перед отсечкой — ' + ', '.join(result['removed'])
    return text
