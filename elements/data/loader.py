""" Загрузка и инкрементальное обновление дневной истории с ISS."""
import io
import logging
from datetime import date, timedelta
from time import sleep

import pandas as pd
import requests
import sqlalchemy as sa
from sqlalchemy.orm import Session

from source.iss.client import ISSClient
from source.settings.settings import (
    handler,
    HISTORY_DAYS,
    HOUR_HISTORY_DAYS,
    INDEX_SECID,
    MIN_TURNOVER,
    DIVIDENDS_URL,
    HTTP_TIMEOUT,
)
from source.settings.sectors import SECTORS
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class DataLoader(SQLmain):

    def __init__(self, client: ISSClient | None = None):
        self.client = client or ISSClient()

    def update_securities(self) -> list[str]:
        """ Справочник бумаг TQBR; возвращает список торгуемых SECID."""
        snapshot = self.client.board_snapshot()
        # В TQBR торгуются и фонды (БПИФ: LQDT, SBMX, GOLD...) — оставляем только акции.
        funds = snapshot.loc[snapshot['INSTRID'] != 'EQIN', 'SECID'].tolist()
        self.remove_securities(funds)
        snapshot = snapshot[
            (snapshot['STATUS'].fillna('A') == 'A') & (snapshot['INSTRID'] == 'EQIN')
        ]
        rows = [
            {
                'SECID': row.SECID,
                'SHORTNAME': row.SHORTNAME,
                'LOTSIZE': int(row.LOTSIZE) if row.LOTSIZE == row.LOTSIZE else None,
                'MINSTEP': float(row.MINSTEP) if row.MINSTEP == row.MINSTEP else None,
                'SECTOR': SECTORS.get(row.SECID, 'unknown'),
            }
            for row in snapshot.itertuples()
        ]
        self.upsert(tables.Security, rows, ['SECID'])
        logger.info(f'Справочник обновлён: {len(rows)} бумаг')
        return [row['SECID'] for row in rows]

    @staticmethod
    def remove_securities(secids: list[str]):
        """ Удаляет бумаги и их историю из БД (например, фонды, попавшие в выборку)."""
        if not secids:
            return
        with Session(bind=main_engine) as s:
            for table in (tables.Security, tables.Candle, tables.HourCandle, tables.Dividend):
                s.execute(sa.delete(table).where(table.SECID.in_(secids)))
            s.commit()

    def update_history(self, secids: list[str] | None = None) -> int:
        """ Докачивает свечи с последней сохранённой даты (с запасом 3 дня)."""
        secids = self.update_securities() if secids is None else secids
        last_dates = self.last_candle_dates()
        first_date = date.today() - timedelta(days=HISTORY_DAYS)
        updated = 0

        for secid in [INDEX_SECID, *secids]:
            last = last_dates.get(secid)
            start = max(first_date, last - timedelta(days=3)) if last else first_date
            try:
                if secid == INDEX_SECID:
                    frame = self.client.index_candles(start)
                else:
                    frame = self.client.candles(secid, start)
            except ConnectionError as error:
                logger.error(f'{secid}: {error}')
                continue
            if frame.empty:
                continue
            self.save_candles(secid, frame)
            updated += 1
        logger.info(f'История обновлена: {updated} инструментов')
        return updated

    def liquid_secids(self) -> list[str]:
        """ Бумаги, проходившие фильтр ликвидности хотя бы иногда за последний год."""
        candles = self.load_candles(self.securities()['SECID'].tolist())
        candles = candles[candles['DATE'] >= candles['DATE'].max() - timedelta(days=365)]
        turnover = candles.groupby('SECID')['VALUE'].median()
        return sorted(turnover[turnover >= MIN_TURNOVER / 2].index.drop(INDEX_SECID, errors='ignore'))

    def update_hour_history(self, secids: list[str] | None = None) -> int:
        """ Часовые свечи ликвидных бумаг и индекса (с последней сохранённой даты)."""
        secids = secids or self.liquid_secids()
        last_dates = self.last_hour_candle_dates()
        first_date = date.today() - timedelta(days=HOUR_HISTORY_DAYS)
        updated = 0
        for number, secid in enumerate([INDEX_SECID, *secids], 1):
            last = last_dates.get(secid)
            start = max(first_date, last.date() - timedelta(days=3)) if last else first_date
            try:
                frame = self.client.candles(secid, start, index=secid == INDEX_SECID, interval=60)
            except ConnectionError as error:
                logger.error(f'{secid}: {error}')
                continue
            if not frame.empty:
                self.save_hour_candles(secid, frame)
                updated += 1
            if number % 20 == 0:
                logger.info(f'Часовые свечи: {number}/{len(secids) + 1}')
        logger.info(f'Часовые свечи обновлены: {updated} инструментов')
        return updated

    def update_dividends(self, secids: list[str] | None = None) -> int:
        """ История дивидендов с dohod.ru (в ISS публичного источника больше нет)."""
        secids = secids or self.liquid_secids()
        rows = []
        for secid in secids:
            try:
                response = requests.get(
                    f'{DIVIDENDS_URL}/{secid.lower()}',
                    headers={'User-Agent': 'Mozilla/5.0'}, timeout=HTTP_TIMEOUT,
                )
                response.raise_for_status()
                tables_html = pd.read_html(io.StringIO(response.text))
            except requests.HTTPError as error:
                if error.response is not None and error.response.status_code == 404:
                    logger.info(f'Дивиденды {secid}: нет страницы на dohod.ru (вероятно, не платит)')
                else:
                    logger.warning(f'Дивиденды {secid}: {error}')
                continue
            except (requests.RequestException, ValueError) as error:
                logger.warning(f'Дивиденды {secid}: {error}')
                continue
            history = next(
                (t for t in tables_html if 'Дата закрытия реестра' in t.columns), None
            )
            if history is None:
                continue
            for row in history.itertuples(index=False):
                registry, value = str(row[1]), row[3]
                forecast = 'прогноз' in registry
                registry = pd.to_datetime(registry.split()[0], format='%d.%m.%Y', errors='coerce')
                announce = pd.to_datetime(str(row[0]), format='%d.%m.%Y', errors='coerce')
                value = pd.to_numeric(value, errors='coerce')
                if pd.isna(registry) or pd.isna(value) or forecast:
                    continue
                rows.append({
                    'SECID': secid,
                    'REGISTRY_DATE': registry.date(),
                    'ANNOUNCE_DATE': announce.date() if pd.notna(announce) else None,
                    'VALUE': float(value),
                    'SOURCE': 'dohod.ru',
                })
            sleep(0.5)
        self.upsert(tables.Dividend, rows, ['SECID', 'REGISTRY_DATE'])
        logger.info(f'Дивиденды обновлены: {len(rows)} выплат')
        return len(rows)
