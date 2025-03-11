from elements.algorithm.worker import Algorithm as A
from elements.statistic.worker import Statistic as s
import source.sql.tables as tables

A.reload_db(table=tables.CurrentScore)
A.reload_db(table=tables.StartScore)
