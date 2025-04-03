from elements.algorithm.worker import Algorithm as A
from elements.statistic.worker import Statistic as s
import source.sql.tables as tables

A.insert_data(
    data=[
        {
            'WPTPWP': -0.8,
            'LCP': -1,
            'PMPWP_WP': 1.7,
            'LCTLWP_WP': 1.6,
            'LCPRCNT': 1.1,
            'LMP': 1,
            'TIC_IC': 1
        },
    ],
    table=tables.Weights
)
