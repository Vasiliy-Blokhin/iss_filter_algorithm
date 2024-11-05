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

BASE_DIR: str = getenv('BASE_DIR') if getenv('BASE_DIR') else getcwd()

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
    'WPTPWP', 'LCP', 'PMPWP_WP', 'LCTLWP_WP', 'LCPRCNT', 'LMP'
]
STOP_TRADING = 'торги приостановлены'
RUN_TRADING = 'торги идут'
# URL для получения данных Мосбиржи.
IMOEX_URL = (
    'http://iss.moex.com/iss/engines/stock/markets/shares/'
    'securities.json?iss.json=extended&iss.meta=off'
)


# Итерация работы.
SET_ITERATION = 3
TIME_UPDATE = 6
START_VALUE = 0

# Баллы по параметрам алгоритма.
WPTPWP_POINTS = 5
LCP_POINTS = 5
PMPWP_WP_POINTS = 8
LCTLWP_WP_POINTS = 5
LCPRCNT_POINTS = 5
LMP_POINTS = 15

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
BORDERS_SCORE = 78

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
