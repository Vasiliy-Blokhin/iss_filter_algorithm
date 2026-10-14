""" Ротация фондов: решение только по данным до начала месяца, ограничение доли акций."""
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('DB_URL', f'sqlite:///{Path(tempfile.mkdtemp()).as_posix()}/test.db')


def synthetic_indexes(equity_trend: float) -> list[dict]:
    dates = pd.bdate_range('2025-01-01', '2026-03-20')
    steps = np.arange(len(dates))
    rows = []
    for secid, values in (
        ('MCFTR', 1000 * (1 + equity_trend) ** steps),
        ('RUGBITR3Y', 500 * 1.0002 ** steps),
        ('RUSFAR', np.full(len(dates), 15.0)),
    ):
        rows += [{'SECID': secid, 'DATE': d.date(), 'CLOSE': float(v)} for d, v in zip(dates, values)]
    return rows


def test_rotation_target_uses_previous_month_only():
    from source.sql.main import SQLmain
    from source.sql import tables
    from elements.rotation.worker import Rotation, target_weights, backtest
    from source.settings.settings import ROTATION_EQ_CAP

    SQLmain.create_all_tables()
    SQLmain.upsert(tables.IndexCandle, synthetic_indexes(0.002), ['SECID', 'DATE'])
    result = Rotation.update_target(date(2026, 3, 16))
    # Решение по последнему дню февраля, даже если в базе есть мартовские данные.
    assert result['date'] == date(2026, 2, 27)
    assert set(result['target']) == {'EQMX', 'LQDT'}
    assert abs(result['target']['EQMX'] - ROTATION_EQ_CAP) < 1e-9
    assert Rotation.update_target(date(2026, 3, 17)) is None  # уже рассчитано в этом месяце

    prices = Rotation.load_prices()
    falling = prices.copy()
    falling['EQ'] = 1000 * 0.997 ** np.arange(len(prices))
    weights, _ = target_weights(falling)
    assert weights['EQ'] == 0 and weights['BOND'] + weights['CASH'] == 1
    stats = backtest(prices, start='2025-08-01')
    assert np.isfinite(stats['CAGR']) and stats['MAX_DRAWDOWN'] <= 0
