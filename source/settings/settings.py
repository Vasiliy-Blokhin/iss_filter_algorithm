from os import getenv, getcwd
import sys

from dotenv import load_dotenv
import logging


# Описание хандлера для логгера.
handler = logging.StreamHandler(sys.stdout)
formater = logging.Formatter(
    '%(name)s, %(funcName)s, %(asctime)s, %(levelname)s - %(message)s.'
)
handler.setFormatter(formater)

load_dotenv()

BASE_DIR = getenv('BASE_DIR') if getenv('BASE_DIR') else getcwd()
REP_NAME = getenv('REP_NAME')
RESTART_BASH_NAME = getenv('RESTART_BASH_NAME')

NEEDFUL = [
    'SECID', 'SHORTNAME', 'PREVPRICE', 'PREVWAPRICE', 'PREVDATE',
    'STATUS', 'WAPTOPREVWAPRICE', 'UPDATETIME', 'LCURRENTPRICE', 'LAST',
    'PRICEMINUSPREVWAPRICE', 'DATAUPDATE', 'CURRENCYID', 'TRADINGSESSION',
    'STATUS_FILTER', 'LASTCHANGEPRCNT', 'WAPRICE', 'LASTCNGTOLASTWAPRICE',
    'WAPTOPREVWAPRICEPRCNT', 'MARKETPRICE', 'ISSUECAPITALIZATION',
    'TRENDISSUECAPITALIZATION', 'BOARDID', 'LOTSIZE', 'HIGH', 'LOW'
]
STATISTIC_NEED = [
    'SECID', 'STATUS_FILTER', 'LAST', 'FILTER_SCORE', 'LOTSIZE'
]
TYPE_DATA_IMOEX = ['securities', 'marketdata']
SHARE_GROUPS = ['EQBR', 'EQBS', 'EQCC', 'TQBR']

WEIGHTS_PARAM = [
    'WPTPWP', 'LCP', 'PMPWP_WP', 'LCTLWP_WP', 'LCPRCNT', 'LMP',
]
STOP_TRADING = 'торги приостановлены'
RUN_TRADING = 'торги идут'
# URL для получения данных Мосбиржи.
IMOEX_URL = (
    'http://iss.moex.com/iss/engines/stock/markets/shares/'
    'securities.json?iss.json=extended&iss.meta=off'
)


# Итерация работы.
SET_ITERATION = 10
TIME_UPDATE = 800
START_VALUE = 0

# Баллы по параметрам алгоритма.
WPTPWP_POINTS = 5
LCP_POINTS = 5
PMPWP_WP_POINTS = 8
LCTLWP_WP_POINTS = 5
LCPRCNT_POINTS = 10
LMP_POINTS = 10

# Коэффициенты для работы алгоритма и статистики.
# Комиссия на оборот.
COMISSION_COEFF = 0.08 / 100
INCOME_COEFF = 0.2 / 100
DELTA_COEFF = 0.15
BUY_COEFF = 0
MAX_SCORE = (
    abs(WPTPWP_POINTS) + abs(LCP_POINTS) +
    abs(PMPWP_WP_POINTS) +
    abs(LCTLWP_WP_POINTS) + abs(LCPRCNT_POINTS) +
    abs(LMP_POINTS)
)
# Абстрактное увеличение результата для выборки
ABSTRACT_COEFF = 1
# Границы допуска.
BORDERS_SCORE = 75

SIZE_F_MAX_SCORE_RESULT = 10

STATUS_UP = 'вероятность роста'
STATUS_DOWN = 'вероятность падения'
STATUS_MEDIUM = 'среднее значение'

db_connector = getenv('db_connector')
db_login = getenv('db_login')
db_password = getenv('db_password')
db_port = getenv('db_port')
db_name = getenv('db_name')
DB_URL = (
    f'{db_connector}://{db_login}:{db_password}'
    f'@localhost:{db_port}/{db_name}'
)

MIN = [-25, 10]
MAX = [5, 15]
MED = [-5, 5]

NULL_DATA_ERROR = 'null_data'
SPLYT_SYMB = '-'

TELEGRAM_TOKEN = getenv('TLG_NOTICE_TOKEN')
TELEGRAM_CHAT_ID = getenv('TLG_NOTICE_CHAT_ID')
TELEGRAM_URL = (
    'https://api.telegram.org/bot'
    f'{TELEGRAM_TOKEN}/sendMessage?chat_id='
    f'{TELEGRAM_CHAT_ID}&text='
)

END_INTERATION_MESSAGE = 'Итерация выполнена успешно.'
ERROR_MESSAGE = 'Получена ошибка - '
EMPTY_STATISTIC_MESSAGE = 'Статистика не собрана.'
DAILY_EXP_MOV_MESSAGE = 'Дневной сбор "LASTWAPRICE" произведен.'
START_MESSAGE = '---Project: iss_filter_algorythm---\n'
RESTART_MESSAGE = 'Перезагрузка сервиса.'
FIRST_MESSAGE = 'Сервис стартовал.'

SIZE_STAT_BASE = 120
