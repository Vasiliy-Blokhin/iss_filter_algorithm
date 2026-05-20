from elements.algorithm.worker import Algorithm as A
from elements.statistic.worker import Statistic as s
import source.sql.tables as tables

A.insert_data(table=tables.AllStatistic, data={'statistic_prcnt': 10, 'neutral_prcnt': 10, 'potential_profitability': 10, 'count_price_after':10})
