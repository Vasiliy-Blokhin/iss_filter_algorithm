""" Широкий набор активов для распределения капитала по режимам рынка.

Каждый актив — индекс полной доходности (сигнал и расчёт) и биржевой фонд/инструмент, которым он
покупается. Ряды хранятся в index_candles (SECID, DATE, CLOSE); валюта и золото — с валютного рынка
(закрытие CETS). Короткие ряды склеиваются со старыми аналогами по доходностям, чтобы GA учился на
истории с 2008 г. (кризисы 2008, 2014, 2020, 2022):
  EQ    — MCFTR (акции с дивидендами), до 2016-11 — IMOEX (без дивидендов, консервативно);
  SMALL — MESMTR (малая и средняя капитализация, с 2014);
  BOND  — RUGBITR3Y (ОФЗ до 3 лет, с 2011);
  LONG  — RGBITR (ОФЗ, полная доходность);
  CORP  — RUCBTRNS (корпоративные облигации), до 2019 — RUCBITR;
  GOLD  — GLDRUB_TOM (золото в рублях, с 2013-10);
  FX    — CNYRUB_TOM (юань, с 2013-04), до этого — USD000UTSTOM;
  CASH  — RUSFAR (с 2018), до этого MOEXREPO (с 2014), ещё раньше — доходность RUGBITR1Y,
          до 2011 — 6% годовых; минус комиссия фонда денежного рынка.
"""
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import sqlalchemy as sa

from source.iss.client import ISSClient
from source.settings.settings import handler, ROTATION_CASH_FEE
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

INDEXES = ['IMOEX', 'MCFTR', 'MESMTR', 'RUGBITR3Y', 'RUGBITR1Y', 'RGBITR', 'RUCBITR', 'RUCBTRNS',
           'RUSFAR', 'MOEXREPO']
CURRENCIES = ['GLDRUB_TOM', 'CNYRUB_TOM', 'USD000UTSTOM']
ASSETS = ['EQ', 'SMALL', 'BOND', 'LONG', 'CORP', 'GOLD', 'FX', 'CASH']
FUNDS = {  # чем покупается актив (все — фонды TQBR; в рабочем семействе GA только EQ/BOND/LONG/CORP/GOLD)
    'EQ': 'EQMX', 'SMALL': 'EQMX', 'BOND': 'SBRB', 'LONG': 'SBGB', 'CORP': 'AKMB',
    'GOLD': 'GOLD', 'FX': 'CNYM', 'CASH': 'LQDT',
}
TITLES = {
    'EQ': 'акции', 'SMALL': 'акции 2-го эшелона', 'BOND': 'короткие ОФЗ', 'LONG': 'длинные ОФЗ',
    'CORP': 'корп. облигации', 'GOLD': 'золото', 'FX': 'юань', 'CASH': 'фонд денег',
}
HISTORY_START = date(2007, 1, 1)
EARLY_CASH_RATE = 0.06
FULL_BEFORE = date(2019, 6, 1)


def currency_history(client: ISSClient, secid: str, start: date) -> pd.Series:
    path = f'history/engines/currency/markets/selt/boards/CETS/securities/{secid}.json'
    frames, offset = [], 0
    while True:
        frame = client.block_to_frame(client.get(path, {'from': start.isoformat(), 'start': offset})['history'])
        if frame.empty:
            break
        frames.append(frame)
        offset += len(frame)
    if not frames:
        return pd.Series(dtype=float)
    frame = pd.concat(frames)
    frame = frame[frame['CLOSE'].notna() & (frame['CLOSE'] > 0)]
    series = frame.groupby('TRADEDATE')['CLOSE'].last()
    series.index = pd.to_datetime(series.index)
    return series


def update(client: ISSClient | None = None) -> int:
    """ Докачать все ряды набора в index_candles (с последней даты, пустые — с 2007 г.)."""
    client = client or ISSClient()
    with main_engine.connect() as connection:
        last = {row[0]: (row[1], row[2]) for row in connection.execute(
            sa.select(tables.IndexCandle.SECID, sa.func.min(tables.IndexCandle.DATE),
                      sa.func.max(tables.IndexCandle.DATE)).group_by(tables.IndexCandle.SECID))}
    saved = 0
    for secid in INDEXES + CURRENCIES:
        first, end = last.get(secid, (None, None))
        # полная загрузка — только пока ряд короче прежней истории ротации (с 2019-09)
        start = end - timedelta(days=5) if first and first < FULL_BEFORE else HISTORY_START
        try:
            if secid in CURRENCIES:
                series = currency_history(client, secid, start)
            else:
                frame = client.candles(secid, start, index=True)
                series = frame.set_index('date')['close'] if len(frame) else pd.Series(dtype=float)
        except ConnectionError as error:
            logger.error(f'{secid}: {error}')
            continue
        rows = [{'SECID': secid, 'DATE': moment.date(), 'CLOSE': float(value)}
                for moment, value in series.items() if np.isfinite(value)]
        if rows:
            SQLmain.upsert(tables.IndexCandle, rows, ['SECID', 'DATE'])
            saved += len(rows)
    logger.info(f'Активы режимов обновлены: {saved} значений')
    return saved


def load_raw() -> pd.DataFrame:
    with main_engine.connect() as connection:
        frame = pd.read_sql(sa.select(tables.IndexCandle.__table__), connection)
    frame['DATE'] = pd.to_datetime(frame['DATE'])
    wide = frame.pivot(index='DATE', columns='SECID', values='CLOSE').sort_index()
    return wide[wide.index.weekday < 5]


def chain(*series: pd.Series) -> pd.Series:
    """ Склейка по доходностям: новый ряд, а до его начала — старые (в порядке приоритета)."""
    returns = None
    for item in series:
        if item is None:
            continue
        r = item.dropna().pct_change()
        returns = r if returns is None else returns.combine_first(r)
    return returns


def rate_returns(rate: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """ Ставка % годовых -> дневная доходность по календарным дням (ставка прошлого дня)."""
    rate = rate.reindex(index).ffill()
    days = index.to_series().diff().dt.days.fillna(0)
    return rate.shift(1) / 100 * days / 365


def load_prices(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """ Индексы полной доходности активов по рабочим дням (NaN — актива ещё не было)."""
    raw = load_raw() if raw is None else raw
    index = raw.index
    col = lambda name: raw[name] if name in raw else None
    returns = pd.DataFrame(index=index)
    returns['EQ'] = chain(col('MCFTR'), col('IMOEX'))
    returns['SMALL'] = chain(col('MESMTR'))
    returns['BOND'] = chain(col('RUGBITR3Y'))
    returns['LONG'] = chain(col('RGBITR'))
    returns['CORP'] = chain(col('RUCBTRNS'), col('RUCBITR'))
    returns['GOLD'] = chain(col('GLDRUB_TOM'))
    returns['FX'] = chain(col('CNYRUB_TOM'), col('USD000UTSTOM'))
    days = index.to_series().diff().dt.days.fillna(0)
    cash = rate_returns(col('RUSFAR'), index) if 'RUSFAR' in raw else None
    repo = rate_returns(col('MOEXREPO'), index) if 'MOEXREPO' in raw else None
    one_year = col('RUGBITR1Y').dropna().pct_change() if 'RUGBITR1Y' in raw else None
    early = EARLY_CASH_RATE * days / 365
    cash_ret = early
    for part in (one_year, repo, cash):
        if part is not None:
            part = part.reindex(index)
            first = part.first_valid_index()
            if first is not None:
                cash_ret = cash_ret.where(index < first, part)
    # склейки и редкие сбои котировок дают ложные скачки — деньги не меняются больше чем на 0.3% в день
    returns['CASH'] = cash_ret.fillna(0).clip(-0.001, 0.003) - ROTATION_CASH_FEE * days / 365
    # первые значения каждого ряда — начало истории актива
    prices = (1 + returns.fillna(0)).cumprod()
    for asset in ASSETS:
        first = returns[asset].first_valid_index()
        if first is None:
            prices[asset] = np.nan
        else:
            prices.loc[prices.index < first, asset] = np.nan
    # дни без торгов актива (выходные биржи) — цена прошлого дня, а не NaN
    return prices[ASSETS]
