from elements.algorithm.worker import Algorithm as A
from elements.statistic.worker import Statistic as s
import source.sql.tables as tables

A.insert_data(
    data=[
        {
            'WPTPWP': 1.06,
            'LCP': 1.2,
            'PMPWP_WP': 1.23,
            'LCTLWP_WP': 1.06,
            'LCPRCNT': 1.15,
            'LMP': 1.3,
            'TIC_IC': 1
        },
    ],
    table=tables.Weights
)
