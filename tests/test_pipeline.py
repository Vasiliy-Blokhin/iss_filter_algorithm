""" Сквозной тест на синтетических данных (без сети).

В данные заложен краткосрочный откат: после сильного падения следующий день
в среднем растёт. Модель должна это найти, а бэктест — не заглядывать в будущее.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Настройки читаются при импорте — подменяем БД и параметры до импорта модулей проекта.
os.environ['DB_URL'] = f'sqlite:///{Path(tempfile.mkdtemp()).as_posix()}/test.db'
os.environ['AI_ENABLED'] = 'false'
os.environ['MIN_TURNOVER'] = '1000000'


@pytest.fixture(scope='module')
def candles():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2023-01-02', periods=700)
    index_ret = rng.normal(0.0003, 0.01, len(dates))
    frames = [pd.DataFrame({
        'SECID': 'IMOEX', 'DATE': dates,
        'CLOSE': 3000 * np.cumprod(1 + index_ret),
    })]
    for number in range(40):
        noise = rng.normal(0, 0.018, len(dates))
        returns = 0.6 * index_ret + noise
        returns[1:] += -0.35 * np.minimum(noise[:-1], 0)  # откат после падений
        close = 100 * np.cumprod(1 + returns)
        open_ = close / (1 + returns) * (1 + rng.normal(0, 0.002, len(dates)))
        frames.append(pd.DataFrame({
            'SECID': f'S{number:02d}', 'DATE': dates, 'CLOSE': close, 'OPEN': open_,
            'HIGH': np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, len(dates)))),
            'LOW': np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, len(dates)))),
            'VALUE': rng.lognormal(19, 0.4, len(dates)), 'VOLUME': 1e6,
        }))
    data = pd.concat(frames, ignore_index=True)
    for column in ('OPEN', 'HIGH', 'LOW'):
        data[column] = data[column].fillna(data['CLOSE'])
    return data


def test_trade_outcome_stop_and_take():
    from elements.features.worker import trade_outcomes
    from source.settings.settings import COMMISSION, SLIPPAGE
    costs = 2 * (COMMISSION + SLIPPAGE)
    frame = pd.DataFrame({
        'OPEN': [100, 100, 100, 100], 'HIGH': [101, 101, 110, 100],
        'LOW': [99, 99, 99, 90], 'CLOSE': [100, 100, 105, 95], 'ATR': [2, 2, 2, 2],
    })
    result = trade_outcomes(frame, hold=2, stop_atr=1.5, take_atr=2.0)
    assert result.loc[0, 'EXIT_KIND'] == 'take'  # вход 100, тейк 104 на второй день
    assert result.loc[0, 'TRADE_RET'] == pytest.approx(0.04 - costs)
    assert result.loc[1, 'EXIT_KIND'] == 'take'  # вход 100, high 110 в день входа
    frame['HIGH'] = [101, 101, 101, 100]
    result = trade_outcomes(frame, hold=2, stop_atr=1.5, take_atr=2.0)
    assert result.loc[1, 'EXIT_KIND'] == 'stop'  # вход 100, стоп 97: low 90 на 2-й день
    assert result.loc[1, 'TRADE_RET'] == pytest.approx(-0.03 - costs)
    assert np.isnan(result.loc[2, 'TRADE_RET'])  # нет данных для выхода


def test_no_lookahead_in_training(candles):
    from elements.features.worker import build_dataset
    from elements.strategy.worker import train_model
    panel = build_dataset(candles)
    as_of = panel['DATE'].sort_values().unique()[400]
    model_full = train_model(panel, pd.Timestamp(as_of))
    model_cut = train_model(panel[panel['DATE'] < as_of], pd.Timestamp(as_of))
    assert np.allclose(model_full.weights, model_cut.weights)


def test_backtest_and_signals(candles):
    from source.sql.main import SQLmain
    from source.sql import tables
    from elements.features.worker import build_dataset
    from elements.backtest.worker import Backtest
    from elements.algorithm.worker import Algorithm

    SQLmain.create_all_tables()
    for secid, frame in candles.groupby('SECID'):
        SQLmain.save_candles(secid, frame.rename(columns=str.lower))
    SQLmain.upsert(tables.Security, [
        {'SECID': f'S{n:02d}', 'SHORTNAME': f'Synthetic {n}', 'LOTSIZE': 10,
         'MINSTEP': 0.01, 'SECTOR': 'unknown'} for n in range(40)
    ], ['SECID'])

    panel = build_dataset(SQLmain.load_candles())
    results = Backtest.run(panel, test_days=200)
    model = results['model']['stats']
    assert model['TRADES'] > 50
    assert model['AVG_TRADE'] > 0, 'модель должна найти заложенный откат'

    signals = Algorithm.generate_signals(use_ai=False)
    assert all(signal['STOP_PRICE'] < signal['CLOSE'] < signal['TAKE_PRICE'] for signal in signals)
    stored = SQLmain.run_sql_command('SELECT COUNT(*) AS n FROM signals')[0]['n']
    assert stored == len(signals)


def test_ai_parse():
    from elements.ai.worker import AIReviewer
    verdict = AIReviewer.parse('ответ: {"decision": "SKIP", "confidence": 140, "reason": "отсечка"}')
    assert verdict == {'decision': 'skip', 'confidence': 100, 'reason': 'отсечка'}


def test_strategy_choice_order():
    """ Неделя решает; при близкой неделе — месяц; при близком месяце — день."""
    import pandas as pd
    from elements.selector.worker import StrategySelector

    def stats(values):
        return pd.DataFrame([
            {'STRATEGY': name, 'PERIOD': period, 'RETURN': value, 'TRADES': 5,
             'WIN_RATE': 0.5, 'AVG_TRADE': 0.001}
            for name, row in values.items() for period, value in zip(('day', 'week', 'month'), row)
        ])

    # Явный лидер недели.
    chosen, _ = StrategySelector.choose(stats({'a': (0, 0.03, 0.01), 'b': (0.01, 0.01, 0.05)}))
    assert chosen == 'a'
    # Недели близки (разница < 0.5%) — решает месяц.
    chosen, _ = StrategySelector.choose(stats({'a': (0, 0.020, 0.01), 'b': (0, 0.018, 0.04)}))
    assert chosen == 'b'
    # Недели и месяцы близки — решает день.
    chosen, _ = StrategySelector.choose(stats({'a': (0.002, 0.02, 0.03), 'b': (0.009, 0.02, 0.031)}))
    assert chosen == 'b'
    # У лучшей и неделя, и месяц в минусе — торговля приостановлена.
    chosen, reason = StrategySelector.choose(stats({'a': (0, -0.01, -0.02), 'b': (0, -0.02, -0.01)}))
    assert chosen is None and 'приостановлена' in reason


def test_diversify_allocation():
    """ Доли пропорциональны месячному результату, минусовые и нулевые исключены."""
    import pandas as pd
    from elements.selector.worker import StrategySelector

    stats = pd.DataFrame([
        {'STRATEGY': name, 'PERIOD': 'month', 'RETURN': value, 'TRADES': 5,
         'WIN_RATE': 0.5, 'AVG_TRADE': 0.001}
        for name, value in {'a': 0.03, 'b': 0.01, 'c': -0.02, 'd': 0.0}.items()
    ])
    allocation, reason = StrategySelector.diversify(stats)
    assert allocation == {'a': 0.75, 'b': 0.25}
    assert 'исключены: c, d' in reason

    negative = stats.assign(RETURN=-0.01)
    allocation, reason = StrategySelector.diversify(negative)
    assert allocation == {} and 'приостановлена' in reason
