""" Режимное распределение капитала: доли, задержка исполнения, счёт, генетика, причинность."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from elements.regime.allocation import (
    Universe, AllocGenome, Policy, Method, search, walk_forward, yearly, fitness, EXEC_LAG, ASSETS,
    ROBUST_ASSETS, random_genome, mutate, crossover,
)
from elements.regime.assets import chain
from elements.regime.detector import REGIME_KEYS


def universe(days=900, seed=0, trend=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2015-01-01', periods=days)
    rets = rng.normal(0.0003, 0.01, size=(days, len(ASSETS)))
    rets[:, ASSETS.index('CASH')] = 0.0003
    if trend:
        rets[:, ASSETS.index(trend)] += 0.002
    prices = pd.DataFrame(np.cumprod(1 + rets, axis=0), index=dates, columns=ASSETS)
    regimes = pd.Series([REGIME_KEYS[(i // 100) % 5] for i in range(days)], index=dates)
    return Universe(prices, regimes)


def test_weights_sum_to_one_and_respect_assets():
    u = universe()
    genome = AllocGenome(tuple(Policy(lbs=(1, 3), k=2, assets=('EQ', 'GOLD'), weighting='invvol') for _ in REGIME_KEYS))
    weights, days = u.targets(genome, 300, 899)
    assert np.allclose(weights.sum(axis=1), 1.0)
    allowed = [ASSETS.index(a) for a in ('EQ', 'GOLD', 'CASH')]
    other = [i for i in range(len(ASSETS)) if i not in allowed]
    assert np.all(weights[:, other] == 0)


def test_cash_policy_earns_cash():
    u = universe()
    genome = AllocGenome(tuple(Policy(exposure=0.0) for _ in REGIME_KEYS))
    acc = u.account(genome, 300, 899)
    assert np.allclose(acc['daily'], 0.0003)


def test_trend_asset_is_found():
    u = universe(trend='GOLD')
    genome = AllocGenome(tuple(Policy(lbs=(3, 4), k=1) for _ in REGIME_KEYS))
    weights, _ = u.targets(genome, 400, 899)
    assert weights[:, ASSETS.index('GOLD')].mean() > 0.8


def test_execution_lag_ignores_signal_day_return():
    """ Доля, решённая по закрытию дня t, не получает доходность дней t и t+1."""
    u = universe()
    genome = AllocGenome(tuple(Policy(lbs=(3,), k=1, beat_cash=False) for _ in REGIME_KEYS), 5, False)
    base = u.account(genome, 300, 400)['daily']
    u.returns[300 + 1] *= 0  # доходность первого дня после решения на старте не влияет: ещё деньги
    u.returns[300 + 1, ASSETS.index('CASH')] = 0.0003
    again = u.account(genome, 300, 400)['daily']
    assert np.isclose(base[1], again[1]) and EXEC_LAG == 2


def test_causal_targets_do_not_change_with_future():
    u = universe()
    genome = AllocGenome(tuple(Policy(lbs=(0, 2), k=2) for _ in REGIME_KEYS), 5, True)
    early, days = u.targets(genome, 300, 600)
    u.mom = u.mom.copy()
    u.mom[:, 700:] *= -1
    u.vol = u.vol.copy()
    u.vol[700:] *= 5
    late, days2 = u.targets(genome, 300, 600)
    assert np.allclose(early, late) and np.array_equal(days, days2)


def test_yearly_and_fitness_shortfall():
    years = np.repeat([2020, 2021], 252)
    daily = np.r_[np.full(252, 0.001), np.full(252, -0.0002)]
    values, labels = yearly(daily, years)
    assert list(labels) == [2020, 2021] and values[0] > 0.2 and values[1] < 0
    loose = fitness(daily, years, Method('a', shortfall=0))
    strict = fitness(daily, years, Method('b', shortfall=5))
    assert strict < loose


def test_robust_family_and_operators():
    rng = random.Random(1)
    for _ in range(50):
        genome = random_genome(rng, 'robust')
        assert all(p.assets == ROBUST_ASSETS and p.k >= 3 and p.weighting == 'invvol' for p in genome.policies)
        child = mutate(rng, crossover(rng, genome, random_genome(rng, 'robust')), 'robust')
        assert all(p.assets == ROBUST_ASSETS for p in child.policies)
        assert AllocGenome.from_dict(child.to_dict()) == child


def test_search_and_walk_forward_run():
    u = universe(days=1100, trend='EQ')
    ranked = search(u, 300, 800, Method('t', population=12, generations=3), random.Random(0))
    assert ranked[0][1] >= ranked[-1][1]
    wf = walk_forward(u, u.dates[900], u.dates[-1], Method('t', population=8, generations=2, runs=2,
                                                            family='robust'), train_start=u.dates[300])
    weights, days = wf['targets']
    assert days[0] == 900 and np.allclose(weights.sum(axis=1), 1.0)


def test_chain_prefers_new_series():
    index = pd.bdate_range('2020-01-01', periods=6)
    old = pd.Series([1, 2, 4, 8, 16, 32.0], index=index)
    new = pd.Series([np.nan, np.nan, np.nan, 10, 11, 12.1], index=index)
    returns = chain(new, old)
    assert np.isclose(returns.iloc[1], 1.0) and np.isclose(returns.iloc[4], 0.1)
