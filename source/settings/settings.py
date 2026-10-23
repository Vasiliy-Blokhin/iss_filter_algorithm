""" Настройки движка сигналов (все значения можно переопределить в .env)."""
from os import getenv
from pathlib import Path
import logging
import sys

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / '.env')


def env_float(name: str, default: float) -> float:
    value = getenv(name)
    return float(value) if value not in (None, '') else default


def env_int(name: str, default: int) -> int:
    value = getenv(name)
    return int(value) if value not in (None, '') else default


def env_bool(name: str, default: bool) -> bool:
    value = getenv(name)
    if value in (None, ''):
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


# Описание хандлера для логгера.
handler = logging.StreamHandler(sys.stdout)
formater = logging.Formatter(
    '%(name)s, %(funcName)s, %(asctime)s, %(levelname)s - %(message)s.'
)
handler.setFormatter(formater)

# База данных. По умолчанию — общий SQLite-файл для всех трёх проектов.
DATA_DIR = Path(getenv('DATA_DIR') or BASE_DIR.parent / 'data')
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_URL = getenv('DB_URL') or f'sqlite:///{(DATA_DIR / "trading.db").as_posix()}'

# Сеть. MARKET_PROXY — прокси только для запросов к Мосбирже
# (например, если системный VPN не пускает на iss.moex.com).
ISS_BASE_URL = getenv('ISS_BASE_URL') or 'https://iss.moex.com/iss'
MARKET_PROXY = getenv('MARKET_PROXY') or None
HTTP_TIMEOUT = env_float('HTTP_TIMEOUT', 20)
DIVIDENDS_URL = getenv('DIVIDENDS_URL') or 'https://www.dohod.ru/ik/analytics/dividend'
HTTP_RETRIES = env_int('HTTP_RETRIES', 3)

BOARD = 'TQBR'
INDEX_SECID = 'IMOEX'
TIMEZONE = 'Europe/Moscow'

# История для обучения и бэктеста (дней календаря).
# ~7 лет: слепому бэктесту самокоррекции нужно несколько лет обучения и проверки.
HISTORY_DAYS = env_int('HISTORY_DAYS', 2600)
HOUR_HISTORY_DAYS = env_int('HOUR_HISTORY_DAYS', 800)

# Фильтр ликвидности: медианный оборот за 20 дней, руб.
MIN_TURNOVER = env_float('MIN_TURNOVER', 100_000_000)
MIN_PRICE = env_float('MIN_PRICE', 1.0)

# Горизонт удержания позиции (торговых дней) и выходы.
HOLD_DAYS = env_int('HOLD_DAYS', 2)
# Выход по времени по открытию последнего дня (исполнитель: TIME_EXIT сразу после 10:00),
# иначе — по закрытию. Должно совпадать с настройками ALOR_TICKETS_BOT.
EXIT_AT_OPEN = env_bool('EXIT_AT_OPEN', False)
STOP_ATR = env_float('STOP_ATR', 1.5)
TAKE_ATR = env_float('TAKE_ATR', 2.0)

# Издержки на одну сторону сделки: комиссия брокера + биржи и проскальзывание.
# Комиссия ALOR (тариф «Первый», оборот до 500 тыс. ₽/день) + биржевой сбор, на сторону сделки.
BROKER_COMMISSION = env_float('BROKER_COMMISSION', 0.0005)
EXCHANGE_FEE = env_float('EXCHANGE_FEE', 0.0001)
COMMISSION = BROKER_COMMISSION + EXCHANGE_FEE
SLIPPAGE = env_float('SLIPPAGE', 0.0003)

# Отбор сигналов.
MAX_SIGNALS = env_int('MAX_SIGNALS', 4)
MAX_PER_SECTOR = env_int('MAX_PER_SECTOR', 1)
MIN_PROBABILITY = env_float('MIN_PROBABILITY', 0.55)
MIN_EDGE = env_float('MIN_EDGE', 0.003)  # ожидаемая доходность после издержек

# Модель (walk-forward логистическая регрессия).
TRAIN_WINDOW = env_int('TRAIN_WINDOW', 500)  # торговых дней в обучении
RETRAIN_EVERY = env_int('RETRAIN_EVERY', 20)
L2_REG = env_float('L2_REG', 1.0)

# Распределение капитала между стратегиями:
#   adaptive  — коррекция по текущей статистике (параметры в БД, подбирает самокоррекция);
#   equal     — поровну между правиловыми стратегиями;
#   diversify — доли пропорционально месячному результату, стратегии с месяцем <= 0 исключаются;
#   best      — одна лучшая стратегия (неделя -> месяц -> день).
ALLOCATION_MODE = (getenv('ALLOCATION_MODE') or 'adaptive').lower()

# Выбор стратегии по теневой статистике: неделя -> месяц -> день.
# Результаты, отличающиеся меньше чем на SELECT_TIE (доля капитала), считаются равными.
SELECT_TIE = env_float('SELECT_TIE', 0.005)
# Не торговать, если у лучшей стратегии и неделя, и месяц не в плюсе.
REQUIRE_POSITIVE = env_bool('REQUIRE_POSITIVE', True)

# Как у исполнителя (ALOR_TICKETS_BOT/.env) — для симуляции портфеля самокоррекцией.
RISK_PER_TRADE = env_float('RISK_PER_TRADE', 0.01)
MAX_POSITIONS = env_int('MAX_POSITIONS', 16)

# Самокоррекция (elements/selftune): окна работы по МСК «ЧЧ:ММ-ЧЧ:ММ/мин_сек-макс_сек»
# через «;» (окно может переходить через полночь), начало слепого периода бэктеста,
# минимальный прирост оценки для принятия кандидата и допуск по каждой половине периода.
SELFTUNE_WINDOWS = getenv('SELFTUNE_WINDOWS') or '08:00-10:00/600-900;13:00-14:00/600-900;22:00-02:00/120-300'
SELFTUNE_NIGHT = getenv('SELFTUNE_NIGHT') or '22:00-02:00'  # здесь же проверяются тяжёлые кандидаты (tuner)
# Воскресенье: рекурсивный поиск по всем алгоритмам (coordinate descent от действующих
# параметров: принятый кандидат становится новой точкой отсчёта), пауза между проверками в секундах.
SELFTUNE_SUNDAY = getenv('SELFTUNE_SUNDAY') or '09:00-13:00/5-20;15:00-17:00/5-20'
# Абстрактный алгоритм (elements/evolve): дни недели (0 = пн ... 6 = вс) и окна случайного
# поиска геномов, начало периода поиска, длина контрольного периода (поиск его не видит),
# порог фиксации этапа (CAGR и на поиске, и на контроле), минимум сделок, размер пула.
EVOLVE_DAYS = getenv('EVOLVE_DAYS') or '6'
EVOLVE_WINDOWS = getenv('EVOLVE_WINDOWS') or '09:00-13:00;15:00-17:00'
# Ежедневная часовая итерация (этапы PHASES) в остальные дни: пн–сб 09:00–10:00.
EVOLVE_DAILY_DAYS = getenv('EVOLVE_DAILY_DAYS') or '0,1,2,3,4,5'
EVOLVE_DAILY_WINDOW = getenv('EVOLVE_DAILY_WINDOW') or '09:00-10:00'
EVOLVE_START = getenv('EVOLVE_START') or '2020-07-01'
EVOLVE_HOLDOUT_DAYS = env_int('EVOLVE_HOLDOUT_DAYS', 250)
# Последние EVOLVE_BLIND_DAYS из контрольного периода — слепая проверка: считается и пишется
# (BLIND_*), но не участвует ни в отборе, ни в защите, ни в снятии этапов. Контроль для защиты —
# первые EVOLVE_HOLDOUT_DAYS − EVOLVE_BLIND_DAYS дней.
EVOLVE_BLIND_DAYS = env_int('EVOLVE_BLIND_DAYS', 60)
EVOLVE_MIN_CAGR = env_float('EVOLVE_MIN_CAGR', 0.05)
EVOLVE_MIN_TRADES = env_int('EVOLVE_MIN_TRADES', 60)
# Упор на число сделок: штраф EVOLVE_ACTIVITY_PENALTY × нехватка до EVOLVE_TARGET_TRADES в год
# на периоде поиска (0 сделок — полный штраф, цель и больше — без штрафа).
EVOLVE_TARGET_TRADES = env_int('EVOLVE_TARGET_TRADES', 250)
EVOLVE_ACTIVITY_PENALTY = env_float('EVOLVE_ACTIVITY_PENALTY', 1.0)
EVOLVE_POOL = env_int('EVOLVE_POOL', 40)
EVOLVE_SIMILAR = env_float('EVOLVE_SIMILAR', 0.7)  # корреляция с этапом, выше — это его вариант
# Сигналы лучших действующих этапов (STRATEGY = 'evolve:<id>'); исполнитель берёт их только в paper.
# Доля капитала позиции = доля из гена size × EVOLVE_LIVE_SHARE / число этапов.
EVOLVE_SIGNALS = env_bool('EVOLVE_SIGNALS', True)
EVOLVE_LIVE_STAGES = env_int('EVOLVE_LIVE_STAGES', 3)
EVOLVE_LIVE_SHARE = env_float('EVOLVE_LIVE_SHARE', 1.0)

# Пул стратегий (elements/pool): правиловые, model, реверсивный и топ-EVOLVE_LIVE_STAGES этапов
# геномного; обновляется перед каждыми сигналами. Приоритет стратегии — по средней сделке:
# бэктест (с поправкой на число сделок, PRIORITY_BT_K), а при PRIORITY_MIN_REAL_TRADES и более
# исполненных сделках — смесь с ними (вес реальных n / (n + PRIORITY_REAL_K)).
# Приоритет = 1 + оценка / PRIORITY_SCALE в пределах [PRIORITY_MIN, PRIORITY_MAX]; он умножает
# долю капитала сигнала, а исполнитель входит сначала в сигналы с большим приоритетом.
PRIORITY_ENABLED = env_bool('PRIORITY_ENABLED', True)
PRIORITY_SCALE = env_float('PRIORITY_SCALE', 0.01)  # средняя сделка +1% -> приоритет x2
PRIORITY_MIN = env_float('PRIORITY_MIN', 0.5)
PRIORITY_MAX = env_float('PRIORITY_MAX', 2.0)
PRIORITY_BT_K = env_int('PRIORITY_BT_K', 30)
PRIORITY_MIN_REAL_TRADES = env_int('PRIORITY_MIN_REAL_TRADES', 20)
PRIORITY_REAL_K = env_int('PRIORITY_REAL_K', 50)
SELFTUNE_REPORT_TIME = env_float('SELFTUNE_REPORT_TIME', 19.00)  # сводка в Telegram раз в сутки (МСК)
BLIND_START = getenv('BLIND_START') or '2021-01-01'
SELFTUNE_MARGIN = env_float('SELFTUNE_MARGIN', 0.003)
SELFTUNE_HALF_TOLERANCE = env_float('SELFTUNE_HALF_TOLERANCE', 0.01)
SELFTUNE_DD_PENALTY = env_float('SELFTUNE_DD_PENALTY', 0.5)  # оценка = CAGR + штраф * просадка
TARGET_CAGR = env_float('TARGET_CAGR', 0.20)

# Ротация «акции / короткие облигации / деньги» (elements/rotation): раз в месяц по данным
# на последний торговый день месяца выбирается актив с лучшим средним ростом за
# ROTATION_LOOKBACKS торговых дней; если он не лучше денег — деньги. Доля акций не больше
# ROTATION_EQ_CAP (остаток — в фонде денежного рынка). Сигналы — по индексам, покупаются фонды.
ROTATION_ENABLED = env_bool('ROTATION_ENABLED', True)
ROTATION_LOOKBACKS = [int(x) for x in (getenv('ROTATION_LOOKBACKS') or '63,126').split(',')]
ROTATION_EQ_CAP = env_float('ROTATION_EQ_CAP', 0.7)
ROTATION_ASSETS = {  # актив: (индекс-сигнал, фонд в TQBR)
    'EQ': (getenv('ROTATION_EQ_INDEX') or 'MCFTR', getenv('ROTATION_EQ_FUND') or 'EQMX'),
    'BOND': (getenv('ROTATION_BOND_INDEX') or 'RUGBITR3Y', getenv('ROTATION_BOND_FUND') or 'SBRB'),
    'CASH': (getenv('ROTATION_CASH_INDEX') or 'RUSFAR', getenv('ROTATION_CASH_FUND') or 'LQDT'),
}
ROTATION_CASH_FEE = env_float('ROTATION_CASH_FEE', 0.004)  # комиссия фонда денежного рынка, год
ROTATION_FUND_FEE = env_float('ROTATION_FUND_FEE', 0.008)  # комиссия фондов акций/облигаций, год
ROTATION_COST = env_float('ROTATION_COST', 0.0015)  # издержки переключения на сторону

# Режим рынка: не покупать, если IMOEX ниже SMA и падает.
REGIME_FILTER = env_bool('REGIME_FILTER', True)
REGIME_SMA = env_int('REGIME_SMA', 50)

# ИИ (AI_Manager): финальная проверка кандидатов.
AI_ENABLED = env_bool('AI_ENABLED', True)
AI_URL = getenv('AI_URL') or 'http://127.0.0.1:8000'
AI_PASSWORD = getenv('AI_PASSWORD') or ''
AI_MODEL = getenv('AI_MODEL') or 'qwen36-35b-a3b'
AI_TIMEOUT = env_float('AI_TIMEOUT', 900)
AI_RETRIES = env_int('AI_RETRIES', 3)
AI_MIN_CONFIDENCE = env_int('AI_MIN_CONFIDENCE', 40)
AI_PROJECT = 'iss_filter_algorithm'

# Реверсивный алгоритм (elements/reverse): правила «от обратного» по опыту + новостной фон.
REVERSE_ENABLED = env_bool('REVERSE_ENABLED', True)
# Локальные модели AI_Manager: оценка новостей и проверка сигналов — лучшая по русскому
# тексту; разбор опыта и новые гипотезы (раз в сутки, ночью) — модель с рассуждением.
REVERSE_NEWS_MODEL = getenv('REVERSE_NEWS_MODEL') or AI_MODEL
REVERSE_MIND_MODEL = getenv('REVERSE_MIND_MODEL') or 'gpt-oss-20b'
REVERSE_REVIEW = env_bool('REVERSE_REVIEW', True)  # ИИ-проверка сигналов реверсивного по новостям
# Окна демона `main.py reverse` (МСК): сбор и оценка новостей; разбор опыта — в ночном окне.
REVERSE_NEWS_EVERY = env_int('REVERSE_NEWS_EVERY', 1200)  # секунд между сборами новостей
REVERSE_NIGHT = getenv('REVERSE_NIGHT') or '22:30-05:30'
REVERSE_SCORE_DEADLINE = env_float('REVERSE_SCORE_DEADLINE', 6.20)  # до сигналов (SIGNAL_TIME)
# Ленты новостей (RSS, через запятую) и параметры оценки.
NEWS_FEEDS = [url.strip() for url in (getenv('NEWS_FEEDS') or ','.join([
    'https://www.interfax.ru/rss.asp',
    'https://www.finam.ru/analysis/conews/rsspoint/',
    'https://1prime.ru/export/rss2/index.xml',
    'https://www.vedomosti.ru/rss/rubric/finance/markets',
    'https://www.kommersant.ru/RSS/section-economics.xml',
    'https://tass.ru/rss/v2.xml',
    'https://smart-lab.ru/rss/',
])).split(',') if url.strip()]
NEWS_BATCH = env_int('NEWS_BATCH', 30)
NEWS_SHIFT_HOURS = env_float('NEWS_SHIFT_HOURS', 6.5)  # новость до 06:30 — к сигналу прошлого дня

# Расписание (время МСК, часы.минуты как float: 18.45 = 18:45).
SIGNAL_TIME = env_float('SIGNAL_TIME', 6.30)
# Ежедневный бэктест: вечерняя сессия до 23:50, с 23:55 свеча дня считается завершённой.
BACKTEST_TIME = env_float('BACKTEST_TIME', 23.56)
LOOP_SLEEP = env_int('LOOP_SLEEP', 60)

# Telegram-уведомления (необязательно).
# Отчёты и логи идут в отдельный бот (не в бота с кнопками).
TELEGRAM_TOKEN = getenv('TLG_REPORT_TOKEN') or ''
TELEGRAM_CHAT_ID = getenv('TLG_REPORT_CHAT_ID') or ''
START_MESSAGE = '---Project: iss_filter_algorithm---\n'
