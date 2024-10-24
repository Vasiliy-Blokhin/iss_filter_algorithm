from dotenv import load_dotenv

import sqlalchemy as sa
from sqlalchemy.orm import Session

from source.sql.tables import Base
from source.settings.settings import DB_URL
load_dotenv()


main_engine = sa.create_engine(
    DB_URL,
    echo=False
)


class SQLmain:

    @staticmethod
    def create_all_tables():
        Base.metadata.create_all(bind=main_engine)

    @staticmethod
    def insert_data(data, table):
        with Session(bind=main_engine) as s:
            Base.metadata.drop_all(
                    bind=main_engine,
                    tables=[table.__table__]
                )
            Base.metadata.create_all(
                bind=main_engine,
                tables=[table.__table__]
            )

            s.execute(sa.insert(table).values(data))
            s.commit()

    @staticmethod
    def append_data(data, table):
        with Session(bind=main_engine) as s:
            s.execute(sa.insert(table).values(data))
            s.commit()

    @staticmethod
    def get_all_data(table):
        with Session(bind=main_engine) as s:
            result = s.execute(
                sa.select('*').select_from(table)
            ).all()

            data = []
            for el in result:
                data.append(el._asdict())
            return data

    @staticmethod
    def get_share_on_secid(table, secid):
        with Session(bind=main_engine) as s:
            result = s.execute(
                sa.select('*').select_from(table).where(table.SECID == secid)
            )

            data = []
            for el in result:
                data.append(el._asdict())
            return data
