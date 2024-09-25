from os import getenv
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

WEIGHTS_PARAM = ['WPTPWP', 'LCP', 'PMPWP-WP', 'TIC-IC', 'LCTLWP-WP', 'LCPRCNT', 'LMP']
STOP_TRADING = 'торги приостановлены'

# URL для получения данных Мосбиржи.
IMOEX_URL = (
    'http://iss.moex.com/iss/engines/stock/markets/shares/'
    'securities.json?iss.json=extended&iss.meta=off'
)


# Итерация работы.
TIME_UPDATE = 100

# Баллы по параметрам алгоритма.
WPTPWP_POINTS = 5
LCP_POINTS = 5
PMPWP_WP_POINTS = 8
TIC_IC_POINTS = 10
LCTLWP_WP_POINTS = 5
LCPRCNT_POINTS = 5
LMP_POINTS = 15

# Коэффициенты для работы алгоритма и статистики.
# Комиссия на оборот.
COMISSION_COEFF = 0.08 / 100
INCOME_COEFF = 0.2 / 100
BUY_COEFF = 0
MAX_SCORE = (
    abs(WPTPWP_POINTS) + abs(LCP_POINTS) +
    abs(PMPWP_WP_POINTS) + abs(TIC_IC_POINTS) +
    abs(LCTLWP_WP_POINTS) + abs(LCPRCNT_POINTS) +
    abs(LMP_POINTS)
)
# Абстрактное увеличение результата для выборки
ABSTRACT_COEFF = 2.5
# Границы допуска.
BORDERS_SCORE = MAX_SCORE * 0.5

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
