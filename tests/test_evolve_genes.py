""" Свод правил «динамика бумаги»: изменение, z-оценка, серии, режимные условия."""
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


def test_derived_series(market):
    lag = 3
    change = market.change('r1', lag)
    rows = market.groups[0]
    values = market.values['r1'][rows]
    assert np.allclose(change[rows][lag:], values[lag:] - values[:-lag], equal_nan=True)
    assert np.isnan(change[rows][:lag]).all()
    streak = market.streak()
    sign = np.sign(np.nan_to_num(values))
    for i in range(1, len(rows)):
        if sign[i] != 0 and sign[i] == sign[i - 1]:
            assert abs(streak[rows][i]) == abs(streak[rows][i - 1]) + 1
    z = market.zscore('r1', 20)
    assert np.nanmean(np.abs(z)) < 3


def test_new_genes_evaluate_and_mutate(market):
    from elements.evolve.genes import REGISTRY, NEW_GENES
    rng = random.Random(3)
    for name in NEW_GENES:
        cls = REGISTRY[name]
        for _ in range(5):
            gene = cls.random(rng, market).mutate(rng, market)
            value = gene.evaluate(market)
            assert len(value) == len(market.didx)
            if cls.slot == 'filter':
                assert value.dtype == bool


def test_phases_run_briefly(market, monkeypatch):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from elements.evolve import worker
    from elements.evolve.genes import REGISTRY, NEW_GENES
    from source.sql.main import SQLmain
    SQLmain.create_all_tables()
    monkeypatch.setattr(worker.Evolver, 'ensure_market', lambda self, panel=None: setattr(self, 'market', market))
    monkeypatch.setattr(worker.tlg, 'send_message', staticmethod(lambda text: None))
    import sqlalchemy as sa
    from source.sql.main import main_engine
    last = {table: SQLmain.run_sql_command(f'SELECT COALESCE(MAX(id), 0) AS n FROM {table}')[0]['n']
            for table in ('evolve_runs', 'evolve_stages')}
    before = {name: REGISTRY[name].weight for name in NEW_GENES}
    reports = worker.run_phases(datetime.now(ZoneInfo('Europe/Moscow')) + timedelta(seconds=6))
    assert len(reports) == 3 and 'новые своды правил' in reports[0]
    assert {name: REGISTRY[name].weight for name in NEW_GENES} == before
    with main_engine.begin() as connection:  # общая БД тестов: убрать свои шаги и этапы
        for table, n in last.items():
            connection.execute(sa.text(f'DELETE FROM {table} WHERE id > :n'), {'n': n})


def test_seeded_genome_inherits_few_successful_genes(market):
    from elements.evolve.genome import Genome, OPERATORS, SLOTS, SEED_GENES, inherit
    rng = random.Random(11)
    donors = [Genome.random(rng, market) for _ in range(3)]
    donor_params = {json.dumps(g.to_dict(), sort_keys=True) for d in donors for genes in d.genes.values() for g in genes}
    donor_values = {(g.name, k, json.dumps(v)) for d in donors for genes in d.genes.values() for g in genes
                    for k, v in g.p.items()}
    inherited_whole = inherited_part = 0
    for _ in range(40):
        child = OPERATORS['seeded'][0](rng, market, None, donors).repair(rng, market)
        for slot, (low, high) in SLOTS.items():
            assert low <= len(child.genes[slot]) <= high
        genes = [g for gs in child.genes.values() for g in gs]
        inherited_whole += any(json.dumps(g.to_dict(), sort_keys=True) in donor_params for g in genes)
        inherited_part += any((g.name, k, json.dumps(v)) in donor_values for g in genes for k, v in g.p.items())
    assert inherited_whole > 0 and inherited_part >= inherited_whole
    # Частичное наследование: класс тот же, хотя бы один параметр донора сохранён.
    gene = donors[0].genes['exit'][0]
    for _ in range(10):
        child_gene = inherit(rng, market, gene)
        assert type(child_gene) is type(gene)
        assert any(child_gene.p[k] == v for k, v in gene.p.items())
    assert 1 <= SEED_GENES <= 3


def test_seeded_used_for_new_genomes(market, monkeypatch):
    from elements.evolve import worker
    from source.sql.main import SQLmain
    SQLmain.create_all_tables()
    evolver = worker.Evolver(seed=5)
    evolver.market = market
    evolver.stage_share = 0.0
    monkeypatch.setattr(worker, 'POOL_PARENT_SHARE', 0.0)  # только новые геномы
    last = SQLmain.run_sql_command('SELECT COALESCE(MAX(id), 0) AS n FROM evolve_runs')[0]['n']
    for _ in range(12):
        evolver.step()
    operators = [row['OPERATOR'] for row in SQLmain.run_sql_command(
        'SELECT OPERATOR FROM evolve_runs WHERE id > :n', {'n': last})]
    assert set(operators) <= {'fresh', 'seeded'} and 'seeded' in operators
    import sqlalchemy as sa
    from source.sql.main import main_engine
    with main_engine.begin() as connection:  # общая БД тестов: убрать свои шаги и этапы
        connection.execute(sa.text('DELETE FROM evolve_runs WHERE id > :n'), {'n': last})
        connection.execute(sa.text('UPDATE evolve_stages SET ACTIVE = 0 WHERE CREATED_AT > :t'),
                           {'t': '2000-01-01'})


def test_fitness_rewards_more_trades():
    from datetime import date
    from elements.evolve import worker
    def result(trades):
        stats = {'SCORE': 0.3, 'CAGR': 0.3, 'TRADES': trades}
        return {'stats': stats, 'halves': [{'CAGR': 0.1}, {'CAGR': 0.1}],
                'search_start': date(2020, 1, 1), 'search_end': date(2024, 1, 1)}
    target = worker.target_trades(result(0))
    assert abs(target - worker.EVOLVE_TARGET_TRADES * 4) < 5
    few, half, enough = (worker.fitness(result(n)) for n in (0, int(target / 2), int(target) + 1))
    assert abs(few - (0.3 - worker.EVOLVE_ACTIVITY_PENALTY)) < 1e-9
    assert few < half < enough == 0.3


def test_holdout_guard_blocks_overfit_replacement():
    from elements.evolve.worker import holdout_guard
    result = lambda cagr: {'holdout': {'CAGR': cagr}}
    assert not holdout_guard(result(0.04), None, 0.05)  # ниже порога
    assert holdout_guard(result(0.06), None, 0.05)
    assert not holdout_guard(result(0.06), {'HOLD_CAGR': 0.08}, 0.05)  # хуже вытесняемого
    assert holdout_guard(result(0.08), {'HOLD_CAGR': 0.08}, 0.05)
    assert holdout_guard(result(0.06), {'HOLD_CAGR': None}, 0.05)


def test_search_lock_and_requests(monkeypatch):
    from datetime import datetime
    from elements.evolve import worker
    from source.sql.main import SQLmain
    SQLmain.create_all_tables()
    monkeypatch.setattr(worker, "LOCK", Path(tempfile.mkdtemp()) / "evolve.lock")
    assert not worker.search_busy()
    worker.touch_lock()
    assert worker.search_busy()
    worker.release_lock()
    assert not worker.search_busy()

    import source.sql.tables as tables
    SQLmain.append_data({'CREATED_AT': datetime.now(), 'MINUTES': 1, 'SOURCE': 'telegram', 'STATUS': 'new'},
                        tables.EvolveRequest)
    request = worker.pending_request()
    assert request and request['MINUTES'] == 1
    calls = []
    monkeypatch.setattr(worker, 'run_phases', lambda until: calls.append(until))
    monkeypatch.setattr(worker.tlg, 'send_message', staticmethod(lambda text: None))
    worker.run_request(request)
    assert calls and worker.pending_request() is None
    status = SQLmain.run_sql_command('SELECT STATUS FROM evolve_requests WHERE id = :id', {'id': request['id']})
    assert status[0]['STATUS'] == 'done'


def test_holdout_split_into_guard_and_blind(market):
    import random as _random
    from elements.evolve.genome import Genome
    from elements.evolve.worker import holdout_guard
    assert market.hold_start <= market.guard_end < market.blind_start <= market.last
    assert market.last - market.blind_start + 1 <= (market.last - market.hold_start + 1) // 2 + 1
    result = market.evaluate(Genome.random(_random.Random(1), market), holdout=True)
    assert {'holdout', 'blind'} <= set(result) and result['blind_start'] == market.date(market.blind_start)
    # Слепая часть не участвует в защите: её значение не меняет решение.
    for blind in (-0.9, 0.9):
        probe = {'holdout': {'CAGR': 0.06}, 'blind': {'CAGR': blind}}
        assert holdout_guard(probe, {'HOLD_CAGR': 0.05}, 0.05)
