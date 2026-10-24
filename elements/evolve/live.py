""" Сигналы лучших действующих этапов абстрактного алгоритма на следующий день.

Каждый этап — свой геном со своим выходом (удержание, стоп/тейк в ATR, выход на открытии):
они передаются исполнителю в самом сигнале. Исполнитель берёт сигналы evolve только в paper.
"""
import json
import logging

import pandas as pd

from elements.evolve.genome import Genome
from elements.evolve.market import EMarket
from source.settings.settings import handler, EVOLVE_LIVE_STAGES, EVOLVE_LIVE_SHARE
from source.sql.main import SQLmain


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def best_stages(limit: int = EVOLVE_LIVE_STAGES) -> list[dict]:
    """ Лучшие действующие общие этапы (без режима) — по основной метрике PERIOD_SCORE."""
    return SQLmain.run_sql_command(
        "SELECT * FROM evolve_stages WHERE ACTIVE AND COALESCE(REGIME, '') = '' "
        'ORDER BY PERIOD_SCORE IS NULL, PERIOD_SCORE DESC, SCORE DESC LIMIT :limit', {'limit': limit})


def stages_by_id(ids: list[int]) -> list[dict]:
    if not ids:
        return []
    marks = ', '.join(str(int(i)) for i in ids)
    return SQLmain.run_sql_command(f'SELECT * FROM evolve_stages WHERE id IN ({marks})')


def stage_signals(panel: pd.DataFrame, limit: int = EVOLVE_LIVE_STAGES, ids: list[int] | None = None) -> list[dict]:
    """ Бумаги последнего дня панели, выбранные этапами: строка панели, вес и правило выхода.
    ids — этапы из пула текущего режима рынка (иначе лучшие общие)."""
    stages = stages_by_id(ids) if ids is not None else best_stages(limit)
    if not stages:
        return []
    market = EMarket(panel)
    last = market.last
    result = []
    for stage in stages:
        genome = Genome.from_dict(json.loads(stage['GENOME']))
        view, picks = market.picks(genome)
        exit_params = genome.genes['exit'][0].p
        stats = market.evaluate(genome)['stats']
        for day, row, weight, score in picks:
            if day != last:
                continue
            result.append({
                'ROW': market.panel.iloc[row], 'STAGE': stage['id'],
                'WEIGHT': weight * EVOLVE_LIVE_SHARE / len(stages),
                'HOLD': int(exit_params['hold']), 'STOP': float(exit_params['stop']),
                'TAKE': float(exit_params['take']),
                'AT_OPEN': bool(exit_params['at_open']) and exit_params['hold'] > 1,
                'PROBABILITY': stats['WIN_RATE'], 'EXPECTED_RETURN': stats['AVG_TRADE'],
            })
        logger.info(f'Этап {stage["id"]}: {sum(1 for d, *_ in picks if d == last)} бумаг на '
                    f'{market.date(last)}')
    return result
