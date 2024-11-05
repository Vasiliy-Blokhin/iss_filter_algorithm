from elements.algorithm.worker import Algorithm as A
from elements.statistic.worker import Statistic as s
import source.sql.tables as tables

A.insert_data(
    data=[
        {
            'WPTPWP': 1.05,
            'LCP': 1.1,
            'PMPWP_WP': 1.14,
            'LCTLWP_WP': 0.97,
            'LCPRCNT': 1.15,
            'LMP': 1.2,
            'EMA': 1
        },
    ],
    table=tables.Weights
)
