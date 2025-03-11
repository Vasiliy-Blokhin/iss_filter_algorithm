from dotenv import load_dotenv
from typing import Annotated, Optional

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

load_dotenv()

intpk = Annotated[int, mapped_column(primary_key=True)]


class Base(DeclarativeBase):
    pass


# Создание модели (таблицы в БД).
class PrepareData(Base):
    """ Сохранение данных, полученных с Мосбиржи."""
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
    LOTSIZE: Mapped[Optional[float]]
    HIGH: Mapped[Optional[float]]
    LOW: Mapped[Optional[float]]


class FilterData(Base):
    """ Оцененные алгоритмом данные с Мосбиржи."""
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
    LOTSIZE: Mapped[Optional[str]]
    FILTER_SCORE: Mapped[Optional[float]]
    STATUS_FILTER: Mapped[Optional[str]]
    HIGH: Mapped[Optional[float]]
    LOW: Mapped[Optional[float]]


class CurrentScore(Base):
    """ Текущие данные для работы статистики и весов."""
    __tablename__ = 'current_score'

    id: Mapped[intpk]
    SECID: Mapped[Optional[str]]
    WPTPWP_MAX: Mapped[Optional[float]]
    LCP_MAX: Mapped[Optional[float]]
    PMPWP_WP_MAX: Mapped[Optional[float]]
    LCTLWP_WP_MAX: Mapped[Optional[float]]
    LCPRCNT_MAX: Mapped[Optional[float]]
    LMP_MAX: Mapped[Optional[float]]
    TIC_IC_MAX: Mapped[Optional[float]]
    WPTPWP_CUR: Mapped[Optional[float]]
    LCP_CUR: Mapped[Optional[float]]
    PMPWP_WP_CUR: Mapped[Optional[float]]
    LCTLWP_WP_CUR: Mapped[Optional[float]]
    LCPRCNT_CUR: Mapped[Optional[float]]
    LMP_CUR: Mapped[Optional[float]]
    TIC_IC_CUR: Mapped[Optional[float]]
    LAST: Mapped[Optional[float]]
    LOTSIZE: Mapped[Optional[float]]
    FILTER_SCORE: Mapped[Optional[float]]
    STATUS_FILTER: Mapped[Optional[str]]


class StartScore(Base):
    """ Начальные данные для работы статистики и весов."""
    __tablename__ = 'start_score'

    id: Mapped[intpk]
    SECID: Mapped[Optional[str]]
    WPTPWP_MAX: Mapped[Optional[float]]
    LCP_MAX: Mapped[Optional[float]]
    PMPWP_WP_MAX: Mapped[Optional[float]]
    LCTLWP_WP_MAX: Mapped[Optional[float]]
    LCPRCNT_MAX: Mapped[Optional[float]]
    LMP_MAX: Mapped[Optional[float]]
    WPTPWP_CUR: Mapped[Optional[float]]
    LCP_CUR: Mapped[Optional[float]]
    PMPWP_WP_CUR: Mapped[Optional[float]]
    LCTLWP_WP_CUR: Mapped[Optional[float]]
    LCPRCNT_CUR: Mapped[Optional[float]]
    LMP_CUR: Mapped[Optional[float]]
    LAST: Mapped[Optional[float]]
    LOTSIZE: Mapped[Optional[float]]
    FILTER_SCORE: Mapped[Optional[float]]
    STATUS_FILTER: Mapped[Optional[str]]


class Weights(Base):
    """ Весы для работы алгоритма."""
    __tablename__ = 'weights'

    id: Mapped[intpk]
    WPTPWP: Mapped[Optional[float]]
    LCP: Mapped[Optional[float]]
    PMPWP_WP: Mapped[Optional[float]]
    LCTLWP_WP: Mapped[Optional[float]]
    LCPRCNT: Mapped[Optional[float]]
    LMP: Mapped[Optional[float]]
    TIC_IC: Mapped[Optional[float]]


class AllStatistic(Base):
    """ Статистика работы алгоритма."""
    __tablename__ = 'all_statistic'

    id: Mapped[intpk]
    statistic_prcnt: Mapped[Optional[float]]
    neutral_prcnt: Mapped[Optional[float]]
    potential_profitability: Mapped[Optional[float]]
    count_price_after: Mapped[Optional[float]]


class Activity(Base):
    """ Время активности. """
    __tablename__ = 'activity'

    id: Mapped[intpk]
    last_time: Mapped[Optional[float]]
