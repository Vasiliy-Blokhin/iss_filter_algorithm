from elements.algorithm.worker import Algorithm as A
from elements.statistic.worker import Statistic as s
import source.sql.tables as tables

A.insert_data(
    data=[
        {'statistic_prcnt': 7.142857142857143, 'neutral_prcnt': 85.71428571428571, 'potential_profitability': -28.371519999995634, 'count_price_after': 46651.9}
    ],
    table=tables.AllStatistic
)
