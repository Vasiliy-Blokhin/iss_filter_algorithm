""" Дата последнего дня с правом на дивиденд (T+2 → T+1)."""

import numpy as np
import pandas as pd
import pytest
from elements.features.intraday import last_cum_date, ex_dividend_cash

def test_last_cum_date_t2_t1():
    # T+2 до 31.07.2023
    # последний день T+2
    # первый день T+1
    # торги выходного дня не считаются
    dates = ['2023-07-11', '2023-07-28', '2023-07-31', '2026-07-20']
    result = last_cum_date(dates)
    expected = np.array(['2023-07-07', '2023-07-26', '2023-07-28', '2026-07-17'], dtype='datetime64[D]')
    np.testing.assert_array_equal(result, expected)

def test_ex_dividend_cash_keys():
    dividends = pd.DataFrame({
        'SECID': ['SBER', 'TATN'],
        'REGISTRY_DATE': ['2026-07-20', '2023-07-11'],
        'VALUE': [37.64, 27.71]
    })
    cash = ex_dividend_cash(pd.DataFrame(), dividends)
    assert cash[('SBER', pd.Timestamp('2026-07-17'))] == pytest.approx(37.64 * 0.87)
    assert cash[('TATN', pd.Timestamp('2023-07-07'))] == pytest.approx(27.71 * 0.87)
