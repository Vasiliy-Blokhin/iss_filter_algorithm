import logging
from datetime import datetime
from time import sleep

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from source.sql.tables import Base, Candle, HourCandle, Security, Activity
from source.settings.settings import DB_URL


logger = logging.getLogger(__name__)

main_engine = sa.create_engine(DB_URL, echo=False)

if main_engine.dialect.name == 'sqlite':
    @event.listens_for(main_engine, 'connect')
    def sqlite_pragmas(connection, _):
        # WAL — чтобы три сервиса могли одновременно читать и писать одну БД.
        cursor = connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('PRAGMA busy_timeout=30000')
        cursor.close()


LOCK_RETRIES = 3


def retry_locked(write):
    """ Общую SQLite пишут несколько процессов; чужая долгая транзакция может держать блокировку
    дольше busy_timeout. Тогда запись повторяется (через 5, 10 с), а не теряется."""
    def wrapper(*args, **kwargs):
        for attempt in range(LOCK_RETRIES):
            try:
                return write(*args, **kwargs)
            except sa.exc.OperationalError as error:
                if 'locked' not in str(error) or attempt == LOCK_RETRIES - 1:
                    raise
                logger.warning(f'БД занята, повтор записи через {5 * (attempt + 1)} с')
                sleep(5 * (attempt + 1))
    return wrapper


class SQLmain:

    @staticmethod
    def create_all_tables():
        Base.metadata.create_all(bind=main_engine)
        # Новые столбцы в уже существующих таблицах (create_all их не добавляет).
        inspector = sa.inspect(main_engine)
        with main_engine.begin() as connection:
            for table in Base.metadata.sorted_tables:
                existing = {column['name'] for column in inspector.get_columns(table.name)}
                for column in table.columns:
                    if column.name not in existing:
                        column_type = column.type.compile(dialect=main_engine.dialect)
                        connection.execute(sa.text(
                            f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {column_type}'
                        ))

    @staticmethod
    @retry_locked
    def upsert(table, rows: list[dict], keys: list[str]):
        """ Вставка с обновлением при конфликте уникального ключа."""
        if not rows:
            return
        with Session(bind=main_engine) as s:
            if main_engine.dialect.name == 'sqlite':
                statement = sqlite_insert(table)
            else:
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                statement = pg_insert(table)
            update_columns = {
                column.name: statement.excluded[column.name]
                for column in table.__table__.columns
                if column.name not in keys and not column.primary_key
            }
            # SQLite ограничивает число параметров в запросе — пишем пачками.
            for start in range(0, len(rows), 500):
                chunk = statement.values(rows[start:start + 500])
                s.execute(chunk.on_conflict_do_update(index_elements=keys, set_=update_columns))
            s.commit()

    @classmethod
    def save_candles(cls, secid: str, frame: pd.DataFrame):
        rows = [
            {
                'SECID': secid,
                'DATE': row.date.date(),
                'OPEN': float(row.open),
                'HIGH': float(row.high),
                'LOW': float(row.low),
                'CLOSE': float(row.close),
                'VALUE': float(row.value) if pd.notna(row.value) else None,
                'VOLUME': float(row.volume) if pd.notna(row.volume) else None,
            }
            for row in frame.itertuples()
        ]
        cls.upsert(Candle, rows, ['SECID', 'DATE'])

    @classmethod
    def save_hour_candles(cls, secid: str, frame: pd.DataFrame):
        rows = [
            {
                'SECID': secid,
                'BEGIN': row.begin.to_pydatetime(),
                'OPEN': float(row.open),
                'HIGH': float(row.high),
                'LOW': float(row.low),
                'CLOSE': float(row.close),
                'VALUE': float(row.value) if pd.notna(row.value) else None,
                'VOLUME': float(row.volume) if pd.notna(row.volume) else None,
            }
            for row in frame.itertuples()
        ]
        cls.upsert(HourCandle, rows, ['SECID', 'BEGIN'])

    @staticmethod
    def load_hour_candles(secids: list[str] | None = None) -> pd.DataFrame:
        query = sa.select(HourCandle.__table__)
        if secids:
            query = query.where(HourCandle.SECID.in_(secids))
        with main_engine.connect() as connection:
            frame = pd.read_sql(query, connection)
        frame['BEGIN'] = pd.to_datetime(frame['BEGIN'])
        return frame.drop(columns='id').sort_values(['SECID', 'BEGIN'])

    @staticmethod
    def last_hour_candle_dates() -> dict:
        with Session(bind=main_engine) as s:
            rows = s.execute(
                sa.select(HourCandle.SECID, sa.func.max(HourCandle.BEGIN)).group_by(HourCandle.SECID)
            ).all()
        return {secid: last for secid, last in rows}

    @staticmethod
    def load_candles(secids: list[str] | None = None) -> pd.DataFrame:
        query = sa.select(Candle.__table__)
        if secids:
            query = query.where(Candle.SECID.in_(secids))
        with main_engine.connect() as connection:
            frame = pd.read_sql(query, connection)
        frame['DATE'] = pd.to_datetime(frame['DATE'])
        return frame.drop(columns='id').sort_values(['SECID', 'DATE'])

    @staticmethod
    def last_candle_dates() -> dict:
        with Session(bind=main_engine) as s:
            rows = s.execute(
                sa.select(Candle.SECID, sa.func.max(Candle.DATE)).group_by(Candle.SECID)
            ).all()
        return {secid: last for secid, last in rows}

    @staticmethod
    def first_candle_dates() -> dict:
        with Session(bind=main_engine) as s:
            rows = s.execute(
                sa.select(Candle.SECID, sa.func.min(Candle.DATE)).group_by(Candle.SECID)
            ).all()
        return {secid: first for secid, first in rows}

    @staticmethod
    def securities() -> pd.DataFrame:
        with main_engine.connect() as connection:
            return pd.read_sql(sa.select(Security.__table__), connection)

    @classmethod
    def save_activity(cls, service: str, message: str = ''):
        """ Отметка «сервис жив». Общую БД пишут несколько процессов, и долгая чужая транзакция
        может держать блокировку дольше busy_timeout — тогда пробуем ещё раз, а не ломаем цикл сервиса."""
        row = {'SERVICE': service, 'LAST_TIME': datetime.now(), 'MESSAGE': message[:500]}
        try:
            cls.upsert(Activity, [row], ['SERVICE'])  # upsert сам повторяет запись при блокировке
        except sa.exc.OperationalError as error:
            if 'locked' not in str(error):
                raise
            logger.warning(f'activity {service}: БД занята, отметка пропущена')

    @staticmethod
    @retry_locked
    def append_data(data, table):
        with Session(bind=main_engine) as s:
            s.execute(sa.insert(table).values(data))
            s.commit()

    @staticmethod
    def run_sql_command(command, params=None):
        with Session(bind=main_engine) as s:
            return s.execute(sa.text(command), params or {}).mappings().fetchall()
