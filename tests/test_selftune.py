""" Самокоррекция: быстрый симулятор, коррекция, слепой бэктест без заглядывания вперёд, демон."""
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('DB_URL', f'sqlite:///{Path(tempfile.mkdtemp()).as_posix()}/test.db')
os.environ.setdefault('AI_ENABLED', 'false')
os.environ.setdefault('MIN_TURNOVER', '1000000')
os.environ['BLIND_START'] = '2023-11-01'

from test_pipeline import candles  # noqa: E402,F401 — общий синтетический рынок

SMALL_TUNER = {'train_days': 250, 'reopt_days': 63, 'block_days': 63, 'lam': 1.0,
               'passes': 1, 'min_trades': 5}


@pytest.fixture(scope='module')
def panel(candles):
    from elements.features.worker import build_dataset
    return build_dataset(candles)


def test_fast_trades_match_backtest(panel):
    """ Быстрые сделки симулятора совпадают с Backtest.trades для всех правиловых стратегий."""
    from elements.backtest.worker import Backtest
    from elements.selftune.simulator import Market, strategy_trades
    from elements.strategy.worker import STRATEGIES, DEFAULT_PARAMS

    market = Market.build(panel)
    for name, scorer in STRATEGIES.items():
        expected = Backtest.trades(scorer(panel, DEFAULT_PARAMS[name]))
        expected = set() if expected.empty else set(zip(expected['DATE'], expected['SECID']))
        rows = strategy_trades(market, name, DEFAULT_PARAMS[name])
        actual = set(zip(market.panel['DATE'].iloc[rows], market.panel['SECID'].iloc[rows]))
        assert actual == expected, name


def test_correction_weight():
    from elements.selftune.simulator import correction_weight, trade_history
    correction = {'lookback': 10, 'min_trades': 2, 'cut': 0.0, 'w_min': 0.0,
                  'boost': 0.01, 'w_max': 2.0, 'slot': 0.1}
    history = trade_history(np.array([5, 6, 7]), np.array([-0.02, -0.01, 0.005]))
    assert correction_weight(*history, 4, correction) == 1.0  # ещё нет закрытых сделок
    assert correction_weight(*history, 6, correction) == 0.0  # средняя -1.5% -> остыла
    history = trade_history(np.array([5, 6]), np.array([0.02, 0.01]))
    assert correction_weight(*history, 6, correction) == 2.0  # средняя +1.5% -> в форме
    assert correction_weight(*history, 30, correction) == 1.0  # сделки вышли из окна


def test_blind_has_no_lookahead(panel):
    """ Пороги на начало отрезка не зависят от данных после него."""
    from elements.selftune.blind import BlindBacktest, SegmentStore
    full = BlindBacktest(panel, SegmentStore(), persist=False)
    starts = full.segment_params(SMALL_TUNER)
    assert len(starts) >= 3
    cut_day = starts[2][0]
    cut_date = full.market.dates[cut_day + 5]
    truncated = BlindBacktest(panel[panel['DATE'] <= cut_date], SegmentStore(), persist=False)
    early = truncated.segment_params(SMALL_TUNER)
    for (day_a, params_a), (day_b, params_b) in zip(starts[:3], early[:3]):
        assert day_a == day_b and params_a == params_b
    result = full.run(SMALL_TUNER, {'lookback': 21, 'min_trades': 3, 'cut': 0.0, 'w_min': 0.0,
                                    'boost': 0.01, 'w_max': 1.5, 'slot': 0.1})
    assert np.isfinite(result['stats']['CAGR'])


def test_windows():
    from elements.selftune.worker import parse_windows, current_window, seconds_to_next_window
    windows = parse_windows('08:00-10:00/600-900;13:00-14:00/600-900;22:00-02:00/120-300')
    assert current_window(datetime(2026, 1, 5, 9, 59), windows)[2:] == (600, 900)
    assert current_window(datetime(2026, 1, 5, 10, 0), windows) is None
    assert current_window(datetime(2026, 1, 5, 23, 30), windows)[2:] == (120, 300)
    assert current_window(datetime(2026, 1, 6, 1, 59), windows)[2:] == (120, 300)
    assert current_window(datetime(2026, 1, 6, 2, 0), windows) is None
    assert seconds_to_next_window(datetime(2026, 1, 5, 12, 0, 0), windows) == 3600


def test_selftune_tick_updates_db(candles):
    from source.sql.main import SQLmain
    from elements.selftune.params import load_active, load_active_record, DEFAULT_PARAMS
    from elements.selftune.worker import SelfTuner

    SQLmain.create_all_tables()
    if not SQLmain.run_sql_command('SELECT COUNT(*) AS n FROM candles')[0]['n']:
        for secid, frame in candles.groupby('SECID'):
            SQLmain.save_candles(secid, frame.rename(columns=str.lower))
    tuner = SelfTuner(seed=1)
    first = tuner.tick()  # refresh: слепой результат + живые пороги
    assert load_active_record('strategy') is not None
    assert set(load_active('strategy')) == set(DEFAULT_PARAMS)
    assert load_active_record('correction')['BLIND_CAGR'] == pytest.approx(first['stats']['CAGR'])
    tuner.tick(heavy=True)  # кандидат
    runs = SQLmain.run_sql_command('SELECT KIND, ACCEPTED FROM selftune_runs ORDER BY id')
    assert [row['KIND'] for row in runs][0] == 'refresh' and len(runs) == 2
