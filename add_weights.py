from elements.algorithm.worker import Algorithm as A
from elements.statistic.worker import Statistic as s
import source.sql.tables as tables

A.insert_data(
    data=[
        {
            'WPTPWP': 1,
            'LCP': 1,
            'PMPWP_WP': 1.1,
            'TIC_IC': 0.8,
            'LCTLWP_WP': 1,
            'LCPRCNT': 1.2,
            'LMP': 1.1,
            'HL': 1
        },
    ],
    table=tables.Weights
)
