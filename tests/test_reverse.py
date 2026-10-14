""" Реверсивный алгоритм: поиск «от обратного», жизненный цикл, новости, сохранение в БД."""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('DB_URL', f'sqlite:///{Path(tempfile.mkdtemp()).as_posix()}/test.db')
os.environ.setdefault('AI_ENABLED', 'false')
os.environ.setdefault('MIN_TURNOVER', '1000000')

from test_pipeline import candles  # noqa: E402,F401 — синтетический рынок с откатом после падений

SMALL = {'lookback': 200, 'reopt': 10, 'min_support': 40, 'promote_n': 15, 'max_age': 120,
         'adapt_min': 5}


@pytest.fixture(scope='module')
def panel(candles):
    from elements.features.worker import build_dataset
    return build_dataset(candles)


def run_engine(panel, last_date=None, params=SMALL):
    from elements.reverse.engine import RMarket, ReverseEngine
    market = RMarket(panel)
    last = len(market.dates) - 1 if last_date is None else market.day(last_date)
    return market, ReverseEngine(market, params).run(params['lookback'] + 5, last)


def test_finds_planted_pattern(panel):
    """ В данных заложен отскок после падения: найденные правила и сигналы его используют."""
    market, engine = run_engine(panel)
    features = {atom[0] for rule in engine.rules for atom in rule.atoms}
    assert {'r1', 'r1_cs', 'clv', 'rsi2', 'gap'} & features
    assert any(rule.status == 'live' for rule in engine.rules) or any(
        rule.promoted is not None for rule in engine.rules)
    known = [row for _, row, _, _ in engine.picks if np.isfinite(market.ret[row])]
    assert len(known) > 30
    assert market.ret[known].mean() > 0  # отбор лучше нуля после издержек


def test_no_lookahead(panel, candles):
    """ Решения до дня X не меняются, если удалить данные после X."""
    from elements.features.worker import build_dataset
    cutoff = pd.Timestamp(np.sort(panel['DATE'].unique())[500])
    _, full = run_engine(panel, cutoff)
    truncated = build_dataset(candles[candles['DATE'] <= cutoff + pd.Timedelta(days=10)])
    market, short = run_engine(truncated, cutoff)
    key = lambda engine: [(engine.market.date(d), engine.market.panel['SECID'].iat[row], round(w, 6))
                          for d, row, w, _ in engine.picks]
    assert key(full) == key(short)
    assert [r.key for r in full.rules] == [r.key for r in short.rules]
    assert [r.status for r in full.rules] == [r.status for r in short.rules]


def test_rule_lifecycle_uses_only_post_birth_trades(panel):
    from elements.reverse.engine import RMarket, Rule
    market = RMarket(panel)
    born = 400
    rule = Rule([['r1', '<', -0.02]], born=born)
    rule.attach(market)
    n_before, _ = rule.oos(born - 1)
    assert n_before == 0  # до рождения у правила нет «внешней» истории
    n, total = rule.oos(len(market.dates) - 1)
    assert n > 0 and np.isfinite(total)


def test_gate_keeps_shadow_until_confident(panel):
    from elements.reverse.engine import RMarket, ReverseEngine
    market = RMarket(panel)
    engine = ReverseEngine(market, SMALL)
    assert not engine.confident(300)  # своей истории ещё нет — только тень
    engine.run(SMALL['lookback'] + 5, 330)
    shadow = [w for d, _, w, _ in engine.picks if d <= 330 and not engine.trading]
    assert all(w == 0 for w in shadow[:3])


def test_news_signal_day_mapping():
    from elements.reverse.news import news_features, MARKET
    dates = pd.to_datetime(['2026-10-09', '2026-10-12', '2026-10-13'])  # пт, пн, вт
    panel = pd.DataFrame({'DATE': np.repeat(dates, 2), 'SECID': ['AAA', 'BBB'] * 3})
    scores = pd.DataFrame({
        'SECID': ['AAA', 'AAA', 'AAA', MARKET],
        'SENTIMENT': [0.5, -1.0, 0.8, 0.4], 'IMPORTANCE': [1.0, 0.5, 1.0, 0.5],
        'EVENT': 'other', 'TITLE': 't',
        'PUBLISHED': pd.to_datetime(['2026-10-11 12:00',   # воскресенье -> сигнал пятницы
                                     '2026-10-13 05:00',   # до 06:30 вт -> сигнал понедельника
                                     '2026-10-13 09:00',   # вторник днём -> сигнал вторника
                                     '2026-10-12 15:00']),
    })
    result = news_features(panel, scores).set_index(['DATE', 'SECID'])
    assert result.loc[(dates[0], 'AAA'), 'news_score'] == pytest.approx(0.5)
    assert result.loc[(dates[1], 'AAA'), 'news_score'] == pytest.approx(-0.5)
    assert result.loc[(dates[2], 'AAA'), 'news_score'] == pytest.approx(0.8)
    assert result.loc[(dates[1], 'BBB'), 'news_score'] == 0  # новости собирались, о бумаге нет
    assert result.loc[(dates[1], 'BBB'), 'mkt_news'] == pytest.approx(0.2)


def test_news_parse_per_ticker_sentiment():
    from elements.reverse.news import NewsScorer
    scorer = NewsScorer.__new__(NewsScorer)
    scorer.tickers, scorer.model = {'SBER': 'Сбербанк', 'EUTR': 'ЕвроТранс'}, 'test'
    batch = [{'id': 10}, {'id': 11}]
    rows = scorer.parse('{"items": [{"i": 0, "t": {"SBER": 0.1, "EUTR": -0.8, "XXXX": 1}, '
                        '"w": 0.7, "e": "legal"}, {"i": 5, "t": {"SBER": 1}}]}', batch)
    assert {(r['SECID'], r['SENTIMENT']) for r in rows} == {('SBER', 0.1), ('EUTR', -0.8)}
    assert all(r['NEWS_ID'] == 10 and r['IMPORTANCE'] == 0.7 for r in rows)


def test_mind_validates_model_output():
    from elements.reverse.mind import valid_atoms, snap_params
    assert valid_atoms([['rsi2', '<', '12'], ['news_score', '>', 0.2]]) == [
        ['rsi2', '<', 12.0], ['news_score', '>', 0.2]]
    assert valid_atoms([['unknown', '<', 1]]) is None
    assert valid_atoms([['rsi2', '=', 1]]) is None
    assert valid_atoms([['rsi2', '<', 1], ['rsi2', '<', 2]]) is None
    assert valid_atoms([]) is None
    assert snap_params({'slot': 0.12, 'evil': 1, 'news_veto': 'x'}, {'slot': 0.05}) == {'slot': 0.1}


def test_live_step_persists_lifecycle(panel):
    from elements.reverse.worker import ReverseAlgorithm
    from elements.selftune.params import save_active, load_active
    from source.sql.main import SQLmain
    SQLmain.create_all_tables()
    import sqlalchemy as sa
    from source.sql.main import main_engine
    with main_engine.begin() as connection:  # общая тестовая БД: сигналы других тестов уже запускали алгоритм
        for table in ('reverse_rules', 'reverse_picks', 'reverse_journal'):
            connection.execute(sa.text(f'DELETE FROM {table}'))
        connection.execute(sa.text("DELETE FROM algo_params WHERE KIND = 'reverse_state'"))
    save_active('reverse', SMALL)
    dates = np.sort(panel['DATE'].unique())
    first = panel[panel['DATE'] <= dates[-30]]
    result = ReverseAlgorithm.step(first)
    assert load_active('reverse_state')['last_day'] == str(pd.Timestamp(dates[-30]).date())
    stored = SQLmain.run_sql_command('SELECT COUNT(*) AS N FROM reverse_rules')[0]['N']
    assert stored == len(result['engine'].rules) > 0
    # Следующие дни догоняются шагами, правила и сигналы продолжают жизнь, а не начинаются заново.
    result = ReverseAlgorithm.step(panel)
    assert load_active('reverse_state')['last_day'] == str(pd.Timestamp(dates[-1]).date())
    assert SQLmain.run_sql_command('SELECT COUNT(*) AS N FROM reverse_rules')[0]['N'] == len(result['engine'].rules)
    again = ReverseAlgorithm.step(panel)  # тот же день — без повторного шага
    assert [r for r, _, _ in again['chosen']] == [r for r, _, _ in result['chosen']]
    journal = SQLmain.run_sql_command('SELECT KIND FROM reverse_journal')
    assert {'birth', 'mode'} <= {row['KIND'] for row in journal}
