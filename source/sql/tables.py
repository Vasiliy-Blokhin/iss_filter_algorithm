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
    EXIT_AT_OPEN: Mapped[Optional[bool]]  # выход по времени на открытии дня HOLD_DAYS (этапы evolve)
    PRIORITY: Mapped[Optional[float]]  # приоритет стратегии (elements/pool): исполнитель входит по нему


class EvolveRequest(Base):
    """ Запросы на итерацию геномного поиска (команда /поиск в Telegram-боте)."""
    __tablename__ = 'evolve_requests'

    id: Mapped[intpk]
    CREATED_AT: Mapped[datetime]
    MINUTES: Mapped[int]
    SOURCE: Mapped[str]  # telegram / manual
    CHAT_ID: Mapped[Optional[int]]
    STATUS: Mapped[str]  # new / running / done / failed
    STARTED_AT: Mapped[Optional[datetime]]
    FINISHED_AT: Mapped[Optional[datetime]]
    NOTE: Mapped[Optional[str]]


class StrategyPool(Base):
    """ Пул стратегий на день: состав и приоритет по бэктесту и исполненным сделкам."""
    __tablename__ = 'strategy_pool'
    __table_args__ = (UniqueConstraint('DATE', 'STRATEGY'),)

    id: Mapped[intpk]
    DATE: Mapped[date] = mapped_column(index=True)
    STRATEGY: Mapped[str]
    KIND: Mapped[str]  # rules / model / reverse / evolve
    BT_TRADES: Mapped[int]
    BT_AVG: Mapped[Optional[float]]  # средняя сделка последнего бэктеста
    BT_SHARPE: Mapped[Optional[float]]
    REAL_TRADES: Mapped[int]  # закрытые позиции исполнителя (paper и live)
    REAL_AVG: Mapped[Optional[float]]
    SCORE: Mapped[float]  # оценка средней сделки
    PRIORITY: Mapped[float]
    CREATED_AT: Mapped[datetime]


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
    STRATEGY: Mapped[Optional[str]]  # пусто — торговля приостановлена; equal/diversify — несколько
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


class AlgoParam(Base):
    """ Параметры алгоритмов (strategy / correction / tuner); активна одна запись на вид.

    BLIND_* — результат слепого бэктеста на момент DATA_END (последняя дата данных).
    """
    __tablename__ = 'algo_params'

    id: Mapped[intpk]
    KIND: Mapped[str] = mapped_column(index=True)
    PARAMS: Mapped[str]  # JSON
    ACTIVE: Mapped[bool] = mapped_column(default=False)
    CREATED_AT: Mapped[datetime]
    DATA_END: Mapped[Optional[date]]
    BLIND_CAGR: Mapped[Optional[float]]  # годовая доходность, доля
    BLIND_DD: Mapped[Optional[float]]
    BLIND_SHARPE: Mapped[Optional[float]]
    BLIND_SCORE: Mapped[Optional[float]]
    BLIND_TRADES: Mapped[Optional[int]]
    NOTE: Mapped[Optional[str]]


class SelftuneRun(Base):
    """ Журнал проверок самокоррекции: кандидат, слепой результат, принят ли."""
    __tablename__ = 'selftune_runs'

    id: Mapped[intpk]
    CREATED_AT: Mapped[datetime] = mapped_column(index=True)
    KIND: Mapped[str]  # refresh / correction / tuner
    PARAMS: Mapped[str]  # JSON {'tuner': ..., 'correction': ...}
    DATA_END: Mapped[Optional[date]]
    BLIND_START: Mapped[Optional[date]]
    CAGR: Mapped[Optional[float]]
    MAX_DRAWDOWN: Mapped[Optional[float]]
    SHARPE: Mapped[Optional[float]]
    TRADES: Mapped[Optional[int]]
    SCORE: Mapped[Optional[float]]
    H1_SCORE: Mapped[Optional[float]]  # первая половина слепого периода
    H2_SCORE: Mapped[Optional[float]]  # вторая половина
    INCUMBENT_SCORE: Mapped[Optional[float]]
    ACCEPTED: Mapped[bool] = mapped_column(default=False)
    SECONDS: Mapped[Optional[float]]
    NOTE: Mapped[Optional[str]]


class EvolveStage(Base):
    """ Зафиксированные этапы абстрактного алгоритма: геном с CAGR не ниже порога и на периоде
    поиска, и на контрольном периоде. PARENT_ID — этап, от которого он получен (цепочка)."""
    __tablename__ = 'evolve_stages'

    id: Mapped[intpk]
    CREATED_AT: Mapped[datetime]
    GENOME: Mapped[str]  # JSON
    GENOME_KEY: Mapped[str] = mapped_column(index=True)
    PARENT_ID: Mapped[Optional[int]]
    OPERATOR: Mapped[Optional[str]]
    SEED: Mapped[Optional[int]]
    DATA_END: Mapped[Optional[date]]
    HOLDOUT_START: Mapped[Optional[date]]
    CAGR: Mapped[Optional[float]]  # период поиска
    MAX_DRAWDOWN: Mapped[Optional[float]]
    SHARPE: Mapped[Optional[float]]
    TRADES: Mapped[Optional[int]]
    SCORE: Mapped[Optional[float]]
    H1_CAGR: Mapped[Optional[float]]
    H2_CAGR: Mapped[Optional[float]]
    HOLD_CAGR: Mapped[Optional[float]]  # контрольный период
    HOLD_DD: Mapped[Optional[float]]
    HOLD_TRADES: Mapped[Optional[int]]
    # Слепая проверка (последние EVOLVE_BLIND_DAYS дней): только для сведения, в отборе не участвует.
    BLIND_START: Mapped[Optional[date]]
    BLIND_CAGR: Mapped[Optional[float]]
    BLIND_RETURN: Mapped[Optional[float]]
    BLIND_DD: Mapped[Optional[float]]
    BLIND_TRADES: Mapped[Optional[int]]
    ACTIVE: Mapped[bool] = mapped_column(default=True)
    NOTE: Mapped[Optional[str]]


class EvolveRun(Base):
    """ Журнал всех проверенных геномов абстрактного алгоритма."""
    __tablename__ = 'evolve_runs'

    id: Mapped[intpk]
    CREATED_AT: Mapped[datetime] = mapped_column(index=True)
    OPERATOR: Mapped[str]
    SEED: Mapped[int]
    PARENT: Mapped[Optional[str]]  # 'stage:<id>' / 'pool' / None
    GENOME: Mapped[str]
    CAGR: Mapped[Optional[float]]
    MAX_DRAWDOWN: Mapped[Optional[float]]
    TRADES: Mapped[Optional[int]]
    SCORE: Mapped[Optional[float]]
    H1_CAGR: Mapped[Optional[float]]
    H2_CAGR: Mapped[Optional[float]]
    HOLD_CAGR: Mapped[Optional[float]]  # только у прошедших поиск
    BLIND_CAGR: Mapped[Optional[float]]
    IMPROVED: Mapped[bool] = mapped_column(default=False)  # лучше родителя
    STAGE_ID: Mapped[Optional[int]]
    SECONDS: Mapped[Optional[float]]


class SelftuneSegment(Base):
    """ Кэш порогов, подобранных слепым бэктестом на дату начала отрезка (только по прошлому)."""
    __tablename__ = 'selftune_segments'
    __table_args__ = (UniqueConstraint('TUNER_KEY', 'SEGMENT_START'),)

    id: Mapped[intpk]
    TUNER_KEY: Mapped[str] = mapped_column(index=True)
    SEGMENT_START: Mapped[date]
    PARAMS: Mapped[str]  # JSON
    CREATED_AT: Mapped[datetime]


class IndexCandle(Base):
    """ Дневные значения индексов для ротации (MCFTR, RUGBITR3Y, RUSFAR — ставка, % годовых)."""
    __tablename__ = 'index_candles'
    __table_args__ = (UniqueConstraint('SECID', 'DATE'),)

    id: Mapped[intpk]
    SECID: Mapped[str] = mapped_column(index=True)
    DATE: Mapped[date] = mapped_column(index=True)
    CLOSE: Mapped[float]


class AllocationTarget(Base):
    """ Целевые доли ротации на месяц (читает исполнитель: доля от капитала ротации)."""
    __tablename__ = 'allocation_targets'
    __table_args__ = (UniqueConstraint('DATE', 'SECID'),)

    id: Mapped[intpk]
    DATE: Mapped[date] = mapped_column(index=True)  # последний торговый день месяца-основания
    ASSET: Mapped[str]  # EQ / BOND / CASH
    SECID: Mapped[str]  # фонд
    WEIGHT: Mapped[float]
    REASON: Mapped[Optional[str]]
    CREATED_AT: Mapped[datetime]


class News(Base):
    """ Новости из лент (RSS, сайт Мосбиржи) для новостного фона."""
    __tablename__ = 'news'

    id: Mapped[intpk]
    UID: Mapped[str] = mapped_column(unique=True)  # хэш заголовка
    SOURCE: Mapped[str]
    PUBLISHED: Mapped[datetime] = mapped_column(index=True)  # МСК
    TITLE: Mapped[str]
    SUMMARY: Mapped[Optional[str]]
    URL: Mapped[Optional[str]]
    SCORED: Mapped[Optional[bool]] = mapped_column(default=False)
    CREATED_AT: Mapped[datetime]


class NewsScore(Base):
    """ Оценка новости локальной моделью: бумага (MARKET — рынок), тональность, важность."""
    __tablename__ = 'news_scores'
    __table_args__ = (UniqueConstraint('NEWS_ID', 'SECID'),)

    id: Mapped[intpk]
    NEWS_ID: Mapped[int] = mapped_column(index=True)
    SECID: Mapped[str] = mapped_column(index=True)
    SENTIMENT: Mapped[float]  # -1..1, для цены на 1-3 дня
    IMPORTANCE: Mapped[float]  # 0..1
    EVENT: Mapped[Optional[str]]
    MODEL: Mapped[Optional[str]]


class ReverseRule(Base):
    """ Правила реверсивного алгоритма и их жизненный цикл (candidate -> live -> dead)."""
    __tablename__ = 'reverse_rules'

    id: Mapped[intpk]
    KEY: Mapped[str] = mapped_column(index=True)
    ATOMS: Mapped[str]  # JSON [[признак, '<' или '>', порог], ...]
    ORIGIN: Mapped[str]  # mined — найдено статистикой; llm — гипотеза локальной модели
    STATUS: Mapped[str]
    BORN: Mapped[date]
    PROMOTED: Mapped[Optional[date]]
    DIED: Mapped[Optional[date]]
    IS_N: Mapped[Optional[int]]  # сделок в окне при рождении
    IS_MEAN: Mapped[Optional[float]]
    OOS_N: Mapped[Optional[int]]  # дней с сигналами вне выборки (после рождения)
    OOS_MEAN: Mapped[Optional[float]]
    POST: Mapped[Optional[float]]  # средняя вне выборки с усадкой к 0 (оценка правила)
    NOTE: Mapped[Optional[str]]
    UPDATED_AT: Mapped[datetime]


class ReverseJournal(Base):
    """ Журнал жизненного цикла реверсивного алгоритма: рождения, смерти, самокоррекция, разбор ИИ."""
    __tablename__ = 'reverse_journal'

    id: Mapped[intpk]
    DATE: Mapped[date] = mapped_column(index=True)  # торговый день решения
    CREATED_AT: Mapped[datetime]
    KIND: Mapped[str]  # birth / promote / death / params / mode / reflection / signal
    TEXT: Mapped[str]


class ReversePick(Base):
    """ Сигналы реверсивного алгоритма (реальные и теневые) — его собственный опыт."""
    __tablename__ = 'reverse_picks'
    __table_args__ = (UniqueConstraint('DATE', 'SECID'),)

    id: Mapped[intpk]
    DATE: Mapped[date] = mapped_column(index=True)  # торговый день сигнала (вход на следующий)
    SECID: Mapped[str]
    WEIGHT: Mapped[float]  # 0 — теневая сделка (алгоритм не уверен в себе)
    SCORE: Mapped[float]
    RULES: Mapped[Optional[str]]
    REAL: Mapped[bool] = mapped_column(default=False)
