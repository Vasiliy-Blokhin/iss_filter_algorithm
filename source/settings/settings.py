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
HISTORY_DAYS = env_int('HISTORY_DAYS', 1100)
HOUR_HISTORY_DAYS = env_int('HOUR_HISTORY_DAYS', 800)

# Фильтр ликвидности: медианный оборот за 20 дней, руб.
MIN_TURNOVER = env_float('MIN_TURNOVER', 100_000_000)
MIN_PRICE = env_float('MIN_PRICE', 1.0)

# Горизонт удержания позиции (торговых дней) и выходы.
HOLD_DAYS = env_int('HOLD_DAYS', 2)
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
#   diversify — доли пропорционально месячному результату, стратегии с месяцем <= 0 исключаются;
#   best      — одна лучшая стратегия (неделя -> месяц -> день).
ALLOCATION_MODE = (getenv('ALLOCATION_MODE') or 'diversify').lower()

# Выбор стратегии по теневой статистике: неделя -> месяц -> день.
# Результаты, отличающиеся меньше чем на SELECT_TIE (доля капитала), считаются равными.
SELECT_TIE = env_float('SELECT_TIE', 0.005)
# Не торговать, если у лучшей стратегии и неделя, и месяц не в плюсе.
REQUIRE_POSITIVE = env_bool('REQUIRE_POSITIVE', True)

# Режим рынка: не покупать, если IMOEX ниже SMA и падает.
REGIME_FILTER = env_bool('REGIME_FILTER', True)
REGIME_SMA = env_int('REGIME_SMA', 50)

# ИИ (AI_Manager): финальная проверка кандидатов.
AI_ENABLED = env_bool('AI_ENABLED', True)
AI_URL = getenv('AI_URL') or 'http://127.0.0.1:8000'
AI_PASSWORD = getenv('AI_PASSWORD') or ''
AI_MODEL = getenv('AI_MODEL') or 'qwen36-35b-a3b'
AI_TIMEOUT = env_float('AI_TIMEOUT', 900)
AI_MIN_CONFIDENCE = env_int('AI_MIN_CONFIDENCE', 40)
AI_PROJECT = 'iss_filter_algorithm'

# Расписание (время МСК, часы.минуты как float: 18.45 = 18:45).
SIGNAL_TIME = env_float('SIGNAL_TIME', 6.30)
LOOP_SLEEP = env_int('LOOP_SLEEP', 60)

# Telegram-уведомления (необязательно).
TELEGRAM_TOKEN = getenv('TLG_NOTICE_TOKEN') or ''
TELEGRAM_CHAT_ID = getenv('TLG_NOTICE_CHAT_ID') or ''
START_MESSAGE = '---Project: iss_filter_algorithm---\n'
