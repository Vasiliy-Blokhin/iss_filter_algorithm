""" Абстрактный алгоритм: гены, геном, операторы, слепая оценка и фиксация этапов."""
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('DB_URL', f'sqlite:///{Path(tempfile.mkdtemp()).as_posix()}/test.db')
os.environ.setdefault('AI_ENABLED', 'false')
os.environ.setdefault('MIN_TURNOVER', '1000000')

from test_pipeline import candles  # noqa: E402,F401 — общий синтетический рынок


@pytest.fixture(scope='module')
def market(candles):
    from elements.features.worker import build_dataset
    from elements.evolve.market import EMarket
    return EMarket(build_dataset(candles), start=str(candles['DATE'].min().date()), holdout_days=120)


def test_genome_roundtrip_and_operators(market):
    from elements.evolve.genome import Genome, OPERATORS, SLOTS
    rng = random.Random(7)
    parent, donor = Genome.random(rng, market), Genome.random(rng, market)
    assert Genome.from_dict(json.loads(json.dumps(parent.to_dict()))).key() == parent.key()
    for name, (function, _) in OPERATORS.items():
        child = function(rng, market, parent, [donor]).repair(rng, market)
        for slot, (low, high) in SLOTS.items():
            assert low <= len(child.genes[slot]) <= high, (name, slot)


def test_extensible_gene_is_used(market):
    """ Новый ген, помеченный @register, сразу попадает в случайные геномы."""
    from elements.evolve.genes import Gene, Real, REGISTRY, register
    from elements.evolve.genome import Genome

    @register
    class AlwaysTrue(Gene):
        name, slot, weight = 'test_always', 'filter', 1000.0
        space = {'x': Real(0, 1, 0.1)}

        def evaluate(self, market):
            return np.ones(len(market.didx), dtype=bool)

    try:
        genome = Genome.random(random.Random(1), market)
        assert any(g.name == 'test_always' for g in genome.genes['filter'])
        assert market.evaluate(genome)['signals'] > 0
    finally:
        REGISTRY.pop('test_always')


def test_quantiles_ignore_holdout(market):
    """ Пороги генов считаются только по периоду поиска."""
    feature = market.features['stock'][0]
    original = market.values[feature]
    try:
        spoiled = original.copy()
        spoiled[market.didx > market.search_end] = 1e9
        market.values[feature] = spoiled
        market._quantiles.clear()
        assert market.quantile(feature, 0.99) < 1e9
    finally:
        market.values[feature] = original
        market._quantiles.clear()


def test_evolver_steps_and_stage(candles, market, monkeypatch):
    from source.sql.main import SQLmain
    import elements.evolve.worker as worker
    SQLmain.create_all_tables()
    monkeypatch.setattr(worker, 'EVOLVE_MIN_CAGR', -1.0)  # синтетика: фиксируем всё, что торгует
    monkeypatch.setattr(worker, 'EVOLVE_MIN_TRADES', 1)
    monkeypatch.setattr(worker.tlg, 'send_message', lambda *a, **k: None)
    monkeypatch.setattr(worker, 'search_ok', lambda r: r['stats']['TRADES'] >= 1)
    evolver = worker.Evolver(seed=3)
    evolver.market, evolver.loaded_end = market, 'synthetic'
    for _ in range(12):
        evolver.step()
    runs = SQLmain.run_sql_command('SELECT * FROM evolve_runs')
    assert len(runs) == 12 and len({r['SEED'] for r in runs}) == 12
    stages = evolver.stages()
    assert stages, 'этапы с положительным (по ослабленному порогу) результатом фиксируются'
    evolver.revalidate()  # перепроверка не ломает этапы
    assert evolver.session_report()


def test_stage_signals_carry_own_exit(candles, market, monkeypatch):
    """ Сигналы этапа: только последний день, правило выхода и вес из генома."""
    import pandas as pd
    from elements.algorithm.worker import Algorithm
    from elements.evolve import live
    from elements.evolve.genes import Gene
    from elements.evolve.genome import Genome
    from elements.evolve.market import EMarket

    genome = Genome({
        'filter': [Gene.from_dict({'gene': 'xrank', 'feature': 'r1', 'op': '>', 'q': 0.0})],
        'logic': [Gene.from_dict({'gene': 'vote', 'need': 1.0})],
        'score': [Gene.from_dict({'gene': 'rank_score', 'feature': 'r5', 'sign': 1, 'w': 1.0})],
        'exit': [Gene.from_dict({'gene': 'exit', 'hold': 4, 'stop': 2.5, 'take': 99.0, 'at_open': 1})],
        'size': [Gene.from_dict({'gene': 'size', 'slot': 0.1, 'top': 2})],
    })
    stage = {'id': 5, 'GENOME': json.dumps(genome.to_dict())}
    monkeypatch.setattr(live, 'best_stages', lambda limit=3: [stage])
    monkeypatch.setattr(live, 'EMarket', lambda panel: EMarket(panel, start=str(candles['DATE'].min().date()),
                                                              holdout_days=120))
    panel = market.panel
    picks = live.stage_signals(panel)
    assert 1 <= len(picks) <= 2
    assert all(p['ROW']['DATE'] == panel['DATE'].max() for p in picks)
    assert all(p['HOLD'] == 4 and p['AT_OPEN'] and p['STOP'] == 2.5 for p in picks)

    monkeypatch.setattr('elements.evolve.live.stage_signals', lambda panel, **kwargs: picks)
    merged = {}
    securities = pd.DataFrame({'SECID': [], 'SHORTNAME': []}).set_index('SECID')
    Algorithm.add_evolve(panel, merged, securities, {}, panel['DATE'].max())
    signal = next(iter(merged.values()))
    assert signal['STRATEGY'] == 'evolve:5' and signal['HOLD_DAYS'] == 4 and signal['EXIT_AT_OPEN']
    assert signal['STOP_PRICE'] == pytest.approx(signal['CLOSE'] - 2.5 * signal['ATR'])
    assert signal['WEIGHT'] == pytest.approx(0.1)
