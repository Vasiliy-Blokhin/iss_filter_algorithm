from datetime import date, datetime
from typing import Annotated, Optional

from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

intpk = Annotated[int, mapped_column(primary_key=True)]


class Base(DeclarativeBase):
    pass


class Candle(Base):
    """ Дневные свечи акций TQBR и индекса IMOEX."""
    __tablename__ = 'candles'
    __table_args__ = (UniqueConstraint('SECID', 'DATE'),)

    id: Mapped[intpk]
    SECID: Mapped[str] = mapped_column(index=True)
    DATE: Mapped[date] = mapped_column(index=True)
    OPEN: Mapped[float]
    HIGH: Mapped[float]
    LOW: Mapped[float]
    CLOSE: Mapped[float]
    VALUE: Mapped[Optional[float]]
    VOLUME: Mapped[Optional[float]]


class HourCandle(Base):
    """ Часовые свечи ликвидных акций (для внутридневных признаков)."""
    __tablename__ = 'hour_candles'
    __table_args__ = (UniqueConstraint('SECID', 'BEGIN'),)

    id: Mapped[intpk]
    SECID: Mapped[str] = mapped_column(index=True)
    BEGIN: Mapped[datetime] = mapped_column(index=True)
    OPEN: Mapped[float]
    HIGH: Mapped[float]
    LOW: Mapped[float]
    CLOSE: Mapped[float]
    VALUE: Mapped[Optional[float]]
    VOLUME: Mapped[Optional[float]]


class Dividend(Base):
    """ Дивиденды: дата закрытия реестра и размер на акцию."""
    __tablename__ = 'dividends'
    __table_args__ = (UniqueConstraint('SECID', 'REGISTRY_DATE'),)

    id: Mapped[intpk]
    SECID: Mapped[str] = mapped_column(index=True)
    REGISTRY_DATE: Mapped[date]
    ANNOUNCE_DATE: Mapped[Optional[date]]  # пусто — дата объявления неизвестна
    VALUE: Mapped[float]
    SOURCE: Mapped[Optional[str]]


class Security(Base):
    """ Справочник бумаг (лотность, шаг цены, сектор)."""
    __tablename__ = 'securities'

    SECID: Mapped[str] = mapped_column(primary_key=True)
    SHORTNAME: Mapped[Optional[str]]
    LOTSIZE: Mapped[Optional[int]]
    MINSTEP: Mapped[Optional[float]]
    SECTOR: Mapped[Optional[str]]


class Signal(Base):
    """ Сигналы на покупку с горизонтом удержания 1-2 дня.

    Читается исполнителем (ALOR_TICKETS_BOT) и Telegram-ботом.
    """
    __tablename__ = 'signals'
    __table_args__ = (UniqueConstraint('SECID', 'SIGNAL_DATE'),)

    id: Mapped[intpk]
    SECID: Mapped[str] = mapped_column(index=True)
    SHORTNAME: Mapped[Optional[str]]
    SECTOR: Mapped[Optional[str]]
    SIGNAL_DATE: Mapped[date] = mapped_column(index=True)  # дата свечи-основания
    CREATED_AT: Mapped[datetime]
    CLOSE: Mapped[float]
    ATR: Mapped[float]
    PROBABILITY: Mapped[float]  # вероятность прибыли модели
    EXPECTED_RETURN: Mapped[float]  # ожидаемая доходность после издержек
    STOP_PRICE: Mapped[float]
    TAKE_PRICE: Mapped[float]
    HOLD_DAYS: Mapped[int]
    FEATURES: Mapped[Optional[str]]  # JSON с признаками
    AI_DECISION: Mapped[Optional[str]]  # buy / skip / n/a
    AI_CONFIDENCE: Mapped[Optional[int]]
    AI_REASON: Mapped[Optional[str]]
    STATUS: Mapped[str] = mapped_column(default='new')  # new / approved / rejected / used / expired
    STRATEGY: Mapped[Optional[str]]  # стратегия(и), давшие сигнал: 'squeeze' или 'squeeze+momentum'
    WEIGHT: Mapped[Optional[float]]  # доля капитала на позицию (с учётом распределения по стратегиям)


class StrategyStat(Base):
    """ Теневая статистика стратегий: результат сделок, закрытых за день/неделю/месяц."""
    __tablename__ = 'strategy_stats'
    __table_args__ = (UniqueConstraint('DATE', 'STRATEGY', 'PERIOD'),)

    id: Mapped[intpk]
    DATE: Mapped[date] = mapped_column(index=True)
    STRATEGY: Mapped[str]
    PERIOD: Mapped[str]  # day / week / month
    RETURN: Mapped[float]  # вклад в капитал, доля
    TRADES: Mapped[int]
    WIN_RATE: Mapped[float]
    AVG_TRADE: Mapped[float]


class StrategySelection(Base):
    """ Стратегия, выбранная для работы на следующий день, и причина выбора."""
    __tablename__ = 'strategy_selection'

    DATE: Mapped[date] = mapped_column(primary_key=True)
    STRATEGY: Mapped[Optional[str]]  # пусто — торговля приостановлена; diversify — несколько
    REASON: Mapped[str]
    ALLOCATION: Mapped[Optional[str]]  # JSON {стратегия: доля капитала}
    CREATED_AT: Mapped[datetime]


class BacktestRun(Base):
    """ Результаты бэктестов (walk-forward)."""
    __tablename__ = 'backtest_runs'

    id: Mapped[intpk]
    CREATED_AT: Mapped[datetime]
    STRATEGY: Mapped[str]
    PERIOD_START: Mapped[Optional[date]]
    PERIOD_END: Mapped[Optional[date]]
    TRADES: Mapped[int]
    WIN_RATE: Mapped[float]
    AVG_TRADE: Mapped[float]  # средняя доходность сделки, доля
    TOTAL_RETURN: Mapped[float]
    MAX_DRAWDOWN: Mapped[float]
    SHARPE: Mapped[float]
    PROFIT_FACTOR: Mapped[float]
    PARAMS: Mapped[Optional[str]]


class Activity(Base):
    """ Время последней активности сервисов (для контроля зависаний)."""
    __tablename__ = 'activity'

    SERVICE: Mapped[str] = mapped_column(primary_key=True)
    LAST_TIME: Mapped[datetime]
    MESSAGE: Mapped[Optional[str]]
