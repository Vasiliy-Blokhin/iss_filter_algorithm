""" Пул стратегий: состав (с реверсивным и этапами геномного) и приоритет по бэктесту и сделкам."""
import os
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('DB_URL', f'sqlite:///{Path(tempfile.mkdtemp()).as_posix()}/test.db')
os.environ.setdefault('AI_ENABLED', 'false')


def add_backtest(strategy: str, trades: int, avg: float):
    from source.sql.main import SQLmain
    import source.sql.tables as tables
    SQLmain.append_data({
        'CREATED_AT': datetime(2026, 10, 20), 'STRATEGY': strategy, 'PERIOD_START': date(2025, 10, 20),
        'PERIOD_END': date(2026, 10, 20), 'TRADES': trades, 'WIN_RATE': 0.5, 'AVG_TRADE': avg,
        'TOTAL_RETURN': 0.0, 'MAX_DRAWDOWN': 0.0, 'SHARPE': 1.0, 'PROFIT_FACTOR': 1.0,
    }, tables.BacktestRun)


def add_stage(stage_id: int, score: float):
    from sqlalchemy.orm import Session
    from source.sql.main import main_engine
    import source.sql.tables as tables
    with Session(bind=main_engine) as s:
        s.add(tables.EvolveStage(
            id=stage_id, CREATED_AT=datetime(2026, 10, 18), GENOME='{}', GENOME_KEY=f'k{stage_id}',
            OPERATOR='fresh', SEED=1, ACTIVE=True, SCORE=score, CAGR=0.1, MAX_DRAWDOWN=0.1, SHARPE=1.0,
            TRADES=100, H1_CAGR=0.1, H2_CAGR=0.1, HOLD_CAGR=0.1, HOLD_DD=0.1, HOLD_TRADES=50,
        ))
        s.commit()


def test_score_and_priority():
    from elements.pool.worker import score, priority_of
    assert score(0, None, 0, None) == 0.0
    assert abs(score(30, 0.01, 0, None) - 0.005) < 1e-12  # малая выборка — оценка вдвое меньше
    assert score(30, 0.01, 19, -0.02) == score(30, 0.01, 0, None)  # реальных сделок мало
    blended = score(30, 0.01, 50, -0.02)  # 50 реальных: вес 0.5
    assert abs(blended - (0.5 * 0.005 + 0.5 * -0.02)) < 1e-12
    assert priority_of(0.0) == 1.0 and priority_of(0.005) == 1.5
    assert priority_of(0.5) == 2.0 and priority_of(-0.5) == 0.5


def test_strategy_parts_and_signal_priority():
    from elements.pool.worker import StrategyPool, strategy_parts
    assert strategy_parts('evolve:12+15') == ['evolve:12', 'evolve:15']
    assert strategy_parts('momentum+reverse') == ['momentum', 'reverse']
    priorities = {'momentum': 1.2, 'reverse': 0.8, 'evolve:12': 1.5}
    assert StrategyPool.signal_priority('momentum+reverse', priorities) == 1.2
    assert StrategyPool.signal_priority('evolve:12+15', priorities) == 1.5
    assert StrategyPool.signal_priority('unknown', priorities) == 1.0


def test_pool_members_priority_and_update(monkeypatch):
    from elements.notification.worker import TelegramNotification
    from elements.pool.worker import StrategyPool
    from source.sql.main import SQLmain
    SQLmain.create_all_tables()
    sent = []
    monkeypatch.setattr(TelegramNotification, 'send_message', staticmethod(lambda text: sent.append(text)))

    import sqlalchemy as sa
    from source.sql.main import main_engine
    with main_engine.begin() as connection:  # общая БД тестов: пул и этапы прошлых тестов не нужны
        connection.execute(sa.text('DELETE FROM strategy_pool'))
        connection.execute(sa.text('UPDATE evolve_stages SET ACTIVE = 0'))

    add_stage(501, 0.4)
    add_backtest('breakout', 90, 0.006)
    add_backtest('model', 6, -0.05)
    add_backtest('evolve:501', 120, 0.008)
    with main_engine.begin() as connection:  # таблица исполнителя (ALOR_TICKETS_BOT)
        connection.execute(sa.text('CREATE TABLE IF NOT EXISTS positions '
                                   '(id INTEGER PRIMARY KEY, STRATEGY TEXT, STATUS TEXT, PNL_PCT REAL)'))
        for _ in range(25):  # реальные сделки breakout хуже бэктеста
            connection.execute(sa.text("INSERT INTO positions (STRATEGY, STATUS, PNL_PCT) VALUES "
                                       "('breakout+momentum', 'closed', -0.004)"))

    first = StrategyPool.refresh(date(2026, 10, 20))
    assert {'model', 'breakout', 'reverse', 'evolve:501'} <= set(first)
    assert first['evolve:501'] > first['breakout'] > 1.0 > first['model']
    assert first['momentum'] < 1.0  # только реальные сделки, и они в минусе
    assert not sent  # первый пул — без сообщения

    from elements.evolve.worker import Evolver
    add_stage(502, 0.5)
    add_stage(503, 0.45)
    add_stage(504, 0.42)
    Evolver.keep_top(Evolver)  # топ-3: 502, 503, 504 — этап 501 вытеснен
    second = StrategyPool.refresh(date(2026, 10, 21))
    assert 'evolve:501' not in second and 'evolve:502' in second
    assert sent and 'evolve:502' in sent[0] and 'выбыли evolve:501' in sent[0]
    rows = SQLmain.run_sql_command("SELECT COUNT(*) AS n FROM strategy_pool WHERE DATE = '2026-10-21'")
    assert rows[0]['n'] == len(second)
