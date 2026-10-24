""" Режимы рынка (детектор, гистерезис) и реалистичный счёт слепого теста."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from elements.regime.detector import apply_hysteresis, classify, regime_series, breadth_from_panel
from elements.regime.realistic import simulate_account, period_stats, after_tax, parse_periods, liquidity_slippage

def trades(**kw):
    base = dict(
        day=[0],
        exit=[3],
        ret=[0.05],
        weight=[0.3],
        entry=[100.0],
        close=[100.0],
        lot=[1],
        atr_pct=[0.02],
        turnover=[1e9],
        secid=[1],
        tag=[0]
    )
    base.update(kw)
    return {k: np.array(v) for k, v in base.items()}

KW = dict(max_positions=5, risk_per_trade=0.01, stop_atr=1.5, cost_per_side=0.0009, max_gap=0.02)

def test_classify():
    assert classify(-0.9) == 'crisis'
    assert classify(-0.3) == 'bear'
    assert classify(0.0) == 'neutral'
    assert classify(0.3) == 'bull'
    assert classify(0.8) == 'boom'
    assert classify(float('nan')) == 'neutral'

def test_hysteresis():
    raw = pd.Series(['neutral','bull','bull','bull','crisis','bull','bull','bull','boom','boom','boom'])
    expected = ['neutral','neutral','neutral','bull','crisis','crisis','crisis','bear','bear','bear','boom']
    assert list(apply_hysteresis(raw,3)) == expected

def test_hysteresis_interrupted():
    raw = pd.Series(['neutral','bull','bull','neutral','bull','bull','bull'])
    expected = ['neutral']*6 + ['bull']
    assert list(apply_hysteresis(raw,3)) == expected

def test_causal():
    rng = np.random.default_rng(0)
    idx = pd.DataFrame({
        'DATE': pd.bdate_range('2020-01-01', periods=600),
        'CLOSE': 100*np.exp(np.cumsum(rng.normal(0,0.01,600)))
    })
    full = regime_series(idx)
    part = regime_series(idx.iloc[:400])
    assert np.allclose(full['score'].iloc[:400].to_numpy(), part['score'].to_numpy(), equal_nan=True)
    assert full['regime'].iloc[:400].tolist() == part['regime'].tolist()

def test_crash_is_crisis():
    up = np.full(400, 1.0005)
    down = np.full(30, 0.985)
    close = np.r_[100*np.cumprod(up), 100*np.cumprod(down)]
    df = pd.DataFrame({
        'DATE': pd.bdate_range('2020-01-01', periods=430),
        'CLOSE': close
    })
    assert regime_series(df)['regime'].iloc[-1] == 'crisis'

def test_breadth():
    d1 = pd.Timestamp('2024-01-01')
    d2 = pd.Timestamp('2024-01-02')
    panel = pd.DataFrame({
        'DATE':[d1,d1,d1,d2,d2,d2],
        'dist_sma50':[0.1,-0.1,np.nan,0.1,0.2,0.3]
    })
    assert breadth_from_panel(panel).tolist() == [0.5,1.0]

def test_account_lots_fees():
    t = trades(
        day=[0,0,2],
        exit=[3,3,5],
        ret=[0.05,-0.02,0.01],
        weight=[0.3]*3,
        entry=[100.,10.,50.],
        close=[100.,10.,50.],
        lot=[1,10,1],
        atr_pct=[0.02]*3,
        turnover=[1e9,1e9,1e8],
        secid=[1,2,1],
        tag=[0,1,0]
    )
    r = simulate_account(t,0,6,7,10000,**KW)
    assert r['trades'] == 3
    assert abs(r['equity'][3]-10087) < 1
    assert abs(r['equity'][-1]-10113.55) < 1
    assert r['by_tag'][1][0] == 1

def test_gap_skip():
    r = simulate_account(trades(entry=[103.0]),0,5,6,10000,**KW)
    assert r['trades'] == 0
    assert np.allclose(r['equity'],10000)

def test_rotation():
    empty = {k: np.array([]) for k in trades()}
    r = simulate_account(empty,0,6,7,10000,short_share=0.0,rotation_daily=np.full(7,0.001),**KW)
    assert np.isclose(r['equity'][-1],10000*1.001**7,rtol=1e-6)

def test_stats_tax():
    assert period_stats(np.array([0.1,-0.5]))['MAX_DRAWDOWN'] == pytest.approx(-0.5)
    assert after_tax(np.array([0.1,0.1]),np.array([2020,2021]),0.13) == pytest.approx(1.087**2-1)

def test_periods():
    assert parse_periods('1m,6m,1y,5y') == [('1m',21),('6m',126),('1y',252),('5y',1260)]
    assert liquidity_slippage(2e8) == 0.0002
