""" Дивидендный разгон: отбор событий, места и добавка к доходности."""
import numpy as np
import pandas as pd
import pytest
from elements.dividend.worker import plan_events, active_events, overlay_daily

def frames():
    dates = pd.bdate_range('2026-05-01', periods=40)
    closes = pd.DataFrame({'AAA': 100.0, 'BBB': 100.0}, index=dates)
    opens = closes.copy()
    turnover = pd.DataFrame({'AAA': 5e8, 'BBB': 5e7}, index=dates)
    dividends = pd.DataFrame({
        'SECID': ['AAA', 'BBB', 'AAA'],
        'REGISTRY_DATE': pd.to_datetime([dates[30], dates[30], dates[35]]),
        'ANNOUNCE_DATE': pd.to_datetime([dates[0], dates[0], dates[25]]),
        'VALUE': [6.0, 6.0, 1.0]
    })
    return opens, closes, turnover, dividends

def test_plan_events_filters():
    opens, closes, turnover, dividends = frames()
    events = plan_events(closes, turnover, dividends)
    assert list(events['SECID']) == ['AAA']
    dates = pd.bdate_range('2026-05-01', periods=40)
    assert events['SIGNAL'].iloc[0] == dates[8]
    assert events['LAST_CUM'].iloc[0] == dates[29]

def test_active_events_window_and_slots():
    dates = pd.bdate_range('2026-05-01', periods=10)
    events = pd.DataFrame({
        'SECID': ['A', 'B', 'C'],
        'SIGNAL': pd.to_datetime(['2026-05-01', '2026-05-01', '2026-05-04']),
        'LAST_CUM': pd.to_datetime(['2026-05-10', '2026-05-20', '2026-05-20']),
        'YIELD': [0.05, 0.06, 0.07]
    })
    assert list(active_events(events, '2026-05-01', 2)['SECID']) == []
    assert list(active_events(events, '2026-05-05', 2)['SECID']) == ['A', 'B']
    assert list(active_events(events, '2026-05-10', 2)['SECID']) == ['B', 'C']

def test_overlay_daily_zero_without_equity():
    opens, closes, turnover, dividends = frames()
    events = plan_events(closes, turnover, dividends)
    extra = overlay_daily(
        events, opens, closes, turnover,
        pd.Series(0.0, index=closes.index),
        pd.Series(0.0, index=closes.index)
    )
    assert (extra == 0).all()

def test_overlay_daily_gain_vs_fund():
    opens, closes, turnover, dividends = frames()
    # compute events before price change
    events = plan_events(closes.copy(), turnover, dividends)
    # modify prices
    closes.loc[closes.index[10]:, 'AAA'] = 110.0
    opens.loc[opens.index[10]:, 'AAA'] = 110.0
    extra = overlay_daily(
        events, opens, closes, turnover,
        pd.Series(0.5, index=closes.index),
        pd.Series(0.0, index=closes.index),
        slots=5
    )
    expected = pytest.approx(0.5 / 5 * 0.10 - 2 * 0.5 / 5 * 0.0, abs=0.002)
    assert extra.sum() == expected
    assert extra.sum() < 0.01
