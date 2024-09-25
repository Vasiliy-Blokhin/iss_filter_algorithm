from dotenv import load_dotenv
from typing import Annotated, Optional

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

load_dotenv()

intpk = Annotated[int, mapped_column(primary_key=True)]


class Base(DeclarativeBase):
    pass


# Создание модели (таблицы в БД).
class FilterData(Base):
    __tablename__ = 'filter_data'

    id: Mapped[intpk]
    SECID: Mapped[Optional[str]]
    SHORTNAME: Mapped[Optional[str]]
    PREVPRICE: Mapped[Optional[float]]
    PREVWAPRICE: Mapped[Optional[float]]
    PREVDATE: Mapped[Optional[str]]
    STATUS: Mapped[Optional[str]]
    WAPTOPREVWAPRICE: Mapped[Optional[float]]
    UPDATETIME: Mapped[Optional[str]]
    LCURRENTPRICE: Mapped[Optional[float]]
    LAST: Mapped[Optional[float]]
    PRICEMINUSPREVWAPRICE: Mapped[Optional[float]]
    DATAUPDATE: Mapped[Optional[str]]
    CURRENCYID: Mapped[Optional[str]]
    TRADINGSESSION: Mapped[Optional[str]]
    LASTCHANGEPRCNT: Mapped[Optional[float]]
    WAPRICE: Mapped[Optional[float]]
    LASTCNGTOLASTWAPRICE: Mapped[Optional[float]]
    WAPTOPREVWAPRICEPRCNT: Mapped[Optional[float]]
    MARKETPRICE: Mapped[Optional[float]]
    ISSUECAPITALIZATION: Mapped[Optional[float]]
    TRENDISSUECAPITALIZATION: Mapped[Optional[float]]
    BOARDID: Mapped[Optional[str]]
    VALUE: Mapped[Optional[str]]
    FILTER_SCORE: Mapped[Optional[float]]
    STATUS_FILTER: Mapped[Optional[str]]


class MaxScore(Base):
    __tablename__ = 'max_score'

    id: Mapped[intpk]
    SECID: Mapped[Optional[str]]
    SHORTNAME: Mapped[Optional[str]]
    PREVPRICE: Mapped[Optional[float]]
    PREVWAPRICE: Mapped[Optional[float]]
    PREVDATE: Mapped[Optional[str]]
    STATUS: Mapped[Optional[str]]
    WAPTOPREVWAPRICE: Mapped[Optional[float]]
    UPDATETIME: Mapped[Optional[str]]
    LCURRENTPRICE: Mapped[Optional[float]]
    LAST: Mapped[Optional[float]]
    PRICEMINUSPREVWAPRICE: Mapped[Optional[float]]
    DATAUPDATE: Mapped[Optional[str]]
    CURRENCYID: Mapped[Optional[str]]
    TRADINGSESSION: Mapped[Optional[str]]
    LASTCHANGEPRCNT: Mapped[Optional[float]]
    WAPRICE: Mapped[Optional[float]]
    LASTCNGTOLASTWAPRICE: Mapped[Optional[float]]
    WAPTOPREVWAPRICEPRCNT: Mapped[Optional[float]]
    MARKETPRICE: Mapped[Optional[float]]
    ISSUECAPITALIZATION: Mapped[Optional[float]]
    TRENDISSUECAPITALIZATION: Mapped[Optional[float]]
    BOARDID: Mapped[Optional[str]]
    VALUE: Mapped[Optional[float]]
    FILTER_SCORE: Mapped[Optional[float]]
    STATUS_FILTER: Mapped[Optional[str]]


class ParamScore(Base):
    __tablename__ = 'params_score'

    id: Mapped[intpk]
    SECID: Mapped[Optional[str]]
    WPTPWP: Mapped[Optional[str]]
    LCP: Mapped[Optional[str]]
    PMPWP_WP: Mapped[Optional[str]]
    TIC_IC: Mapped[Optional[str]]
    LCTLWP_WP: Mapped[Optional[str]]
    LCPRCNT: Mapped[Optional[str]]
    LMP: Mapped[Optional[str]]
    LAST: Mapped[Optional[float]]


class Weights(Base):
    __tablename__ = 'weights'

    id: Mapped[intpk]
    WPTPWP: Mapped[Optional[float]]
    LCP: Mapped[Optional[float]]
    PMPWP_WP: Mapped[Optional[float]]
    TIC_IC: Mapped[Optional[float]]
    LCTLWP_WP: Mapped[Optional[float]]
    LCPRCNT: Mapped[Optional[float]]
    LMP: Mapped[Optional[float]]


class PrepareData(Base):
    __tablename__ = 'prepare_data'

    id: Mapped[intpk]
    SECID: Mapped[Optional[str]]
    SHORTNAME: Mapped[Optional[str]]
    PREVPRICE: Mapped[Optional[float]]
    PREVWAPRICE: Mapped[Optional[float]]
    PREVDATE: Mapped[Optional[str]]
    STATUS: Mapped[Optional[str]]
    WAPTOPREVWAPRICE: Mapped[Optional[float]]
    UPDATETIME: Mapped[Optional[str]]
    LCURRENTPRICE: Mapped[Optional[float]]
    LAST: Mapped[Optional[float]]
    PRICEMINUSPREVWAPRICE: Mapped[Optional[float]]
    DATAUPDATE: Mapped[Optional[str]]
    CURRENCYID: Mapped[Optional[str]]
    TRADINGSESSION: Mapped[Optional[str]]
    LASTCHANGEPRCNT: Mapped[Optional[float]]
    WAPRICE: Mapped[Optional[float]]
    LASTCNGTOLASTWAPRICE: Mapped[Optional[float]]
    WAPTOPREVWAPRICEPRCNT: Mapped[Optional[float]]
    MARKETPRICE: Mapped[Optional[float]]
    ISSUECAPITALIZATION: Mapped[Optional[float]]
    TRENDISSUECAPITALIZATION: Mapped[Optional[float]]
    BOARDID: Mapped[Optional[str]]
    VALUE: Mapped[Optional[float]]
