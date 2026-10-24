""" Кандидаты в пулы и пулы стратегий по режимам рынка.

Кандидат — стратегия в общем виде «сигналы по строкам панели + правило выхода»:
  rule:<имя>          — правиловая стратегия (elements/strategy) с действующими порогами;
  evolve:<id>         — этап геномного алгоритма (общий или своего режима).
Пул режима — до REGIME_POOL_SIZE кандидатов с лучшей средней сделкой (с поправкой на малую
выборку n / (n + 30)) только по сделкам, открытым в дни этого режима; кандидат без плюса или
с числом сделок меньше REGIME_MIN_TRADES в пул не попадает. Пустой пул — в этом режиме
краткосрочные сделки не открываются (капитал остаётся в ротации фондов).
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from elements.evolve.genome import Genome
from elements.evolve.market import EMarket
from elements.regime.detector import REGIME_KEYS, regime_info
from elements.strategy.worker import STRATEGIES, ACTIVE_PARAMS, rule_condition
from source.settings.settings import (
    handler, HOLD_DAYS, STOP_ATR, TAKE_ATR, EXIT_AT_OPEN, MAX_SIGNALS, REGIME_POOL_SIZE,
    REGIME_MIN_TRADES, PRIORITY_BT_K,
)
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

RULE_SLOT = 1 / (MAX_SIGNALS * HOLD_DAYS)  # доля капитала на сделку правиловой стратегии


@dataclass
class Candidate:
    name: str
    exit: tuple                       # (удержание, стоп ATR, тейк ATR, выход на открытии)
    picks: list = field(default_factory=list)  # [(день, строка, доля капитала, оценка)]
    genome: Genome | None = None

    def view(self, market: EMarket):
        return market.outcome(*self.exit)


def rule_candidate(market: EMarket, name: str, params: dict | None = None) -> Candidate:
    """ Сигналы правиловой стратегии по всей панели: топ MAX_SIGNALS бумаг дня по её оценке."""
    exit_rule = (HOLD_DAYS, STOP_ATR, TAKE_ATR, int(EXIT_AT_OPEN))
    view = market.outcome(*exit_rule)
    condition, score = rule_condition(name, market.panel, params or ACTIVE_PARAMS[name])
    mask = condition.fillna(False).to_numpy(dtype=bool) & view.ok
    if market.day_mask is not None:
        mask &= market.day_mask[market.didx]
    rows = np.nonzero(mask & (market.didx >= market.first))[0]
    score = score.to_numpy(dtype=float)
    candidate = Candidate(f'rule:{name}', exit_rule)
    if not len(rows):
        return candidate
    rows = rows[np.lexsort((-np.nan_to_num(score[rows], nan=-1e9), market.didx[rows]))]
    days = market.didx[rows]
    starts = np.r_[0, np.nonzero(np.diff(days))[0] + 1]
    position = np.arange(len(rows)) - np.repeat(starts, np.diff(np.r_[starts, len(rows)]))
    rows = rows[position < MAX_SIGNALS]
    candidate.picks = [(int(market.didx[r]), int(r), RULE_SLOT, float(score[r])) for r in rows]
    return candidate


def genome_candidate(market: EMarket, name: str, genome: Genome) -> Candidate:
    exit_params = genome.genes['exit'][0].p
    exit_rule = (int(exit_params['hold']), float(exit_params['stop']), float(exit_params['take']),
                 int(exit_params['at_open']))
    _, picks = market.picks(genome)
    return Candidate(name, exit_rule, picks, genome)


def rule_candidates(market: EMarket) -> list[Candidate]:
    return [rule_candidate(market, name) for name in STRATEGIES]


def regime_score(market: EMarket, candidate: Candidate, regime_days: np.ndarray,
                 first: int, last: int) -> dict:
    """ Средняя сделка кандидата только по входам в дни режима в [first, last] с выходом до last."""
    view = candidate.view(market)
    rets = [view.ret[row] for day, row, *_ in candidate.picks
            if first <= day <= last and regime_days[day] and view.eidx[row] <= last
            and np.isfinite(view.ret[row])]
    count = len(rets)
    avg = float(np.mean(rets)) if count else 0.0
    return {'STRATEGY': candidate.name, 'TRADES': count, 'AVG_TRADE': avg,
            'WIN_RATE': float(np.mean(np.array(rets) > 0)) if count else 0.0,
            'RETURN': float(np.sum(rets)), 'SCORE': avg * count / (count + PRIORITY_BT_K)}


def build_pools(market: EMarket, candidates: list[Candidate], regimes: np.ndarray,
                first: int, last: int, size: int = REGIME_POOL_SIZE) -> dict[str, list[dict]]:
    """ Пул каждого режима: лучшие кандидаты по сделкам в днях режима (причинно: только [first, last]).
    regimes — ключ режима по номеру дня рынка."""
    pools = {}
    for key in REGIME_KEYS:
        days = regimes == key
        own = [c for c in candidates if not c.name.startswith('gen:') or c.name.startswith(f'gen:{key}:')
               or c.name.startswith('gen:all:')]
        rows = [regime_score(market, c, days, first, last) for c in own]
        rows = [r for r in rows if r['TRADES'] >= REGIME_MIN_TRADES and r['AVG_TRADE'] > 0]
        rows.sort(key=lambda r: -r['SCORE'])
        pools[key] = rows[:size]
    return pools


def entry_arrays(market: EMarket) -> dict:
    """ Для реалистичного счёта по строкам панели: цена входа (открытие t+1), закрытие t, лот, оборот."""
    panel = market.panel
    lots = SQLmain.securities().set_index('SECID')['LOTSIZE'].to_dict()
    return {
        'entry': panel.groupby('SECID', sort=False)['OPEN'].shift(-1).to_numpy(dtype=float),
        'close': panel['CLOSE'].to_numpy(dtype=float),
        'lot': panel['SECID'].map(lots).fillna(1).clip(lower=1).to_numpy(dtype=int),
        'turnover': panel['TURNOVER20'].to_numpy(dtype=float),
    }


def trades_of(market: EMarket, arrays: dict, items: list[tuple]) -> dict:
    """ items: (день, порядок, строка, доля, доходность, день выхода, код стратегии), отсортированные."""
    rows = np.array([t[2] for t in items], dtype=int)
    if not len(rows):
        return {k: np.array([]) for k in ('day', 'exit', 'ret', 'weight', 'entry', 'close', 'lot',
                                          'atr_pct', 'turnover', 'secid', 'tag')}
    return {
        'day': np.array([t[0] for t in items], dtype=int), 'exit': np.array([t[5] for t in items]),
        'ret': np.array([t[4] for t in items], dtype=float), 'weight': np.array([t[3] for t in items], dtype=float),
        'entry': arrays['entry'][rows], 'close': arrays['close'][rows], 'lot': arrays['lot'][rows],
        'atr_pct': market.atr_pct[rows], 'turnover': arrays['turnover'][rows], 'secid': market.secid[rows],
        'tag': np.array([t[6] for t in items], dtype=int),
    }


def regimes_by_day(market: EMarket, history: pd.DataFrame) -> np.ndarray:
    """ Ключ режима для каждого дня рынка (по истории режимов; нет данных — нейтральный)."""
    series = history['regime'].copy()
    series.index = pd.to_datetime(series.index)
    aligned = series.reindex(pd.DatetimeIndex(market.dates)).ffill().fillna('neutral')
    return aligned.to_numpy(dtype=object)


class RegimePools(SQLmain):

    @staticmethod
    def stage_candidates(market: EMarket) -> list[Candidate]:
        """ Действующие этапы геномного: общие (REGIME пусто) — gen:all, режимные — gen:<режим>."""
        rows = SQLmain.run_sql_command('SELECT * FROM evolve_stages WHERE ACTIVE ORDER BY SCORE DESC')
        result = []
        for row in rows:
            regime = row.get('REGIME') or 'all'
            genome = Genome.from_dict(json.loads(row['GENOME']))
            candidate = genome_candidate(market, f'gen:{regime}:{row["id"]}', genome)
            result.append(candidate)
        return result

    @classmethod
    def rebuild(cls, panel: pd.DataFrame, history: pd.DataFrame, notify: bool = True) -> dict:
        """ Пересобрать пулы всех режимов на всей истории (с EVOLVE_START) и сохранить."""
        market = EMarket(panel, holdout_days=0, blind_days=0)
        market.set_search_end(market.last)
        regimes = regimes_by_day(market, history)
        candidates = rule_candidates(market) + cls.stage_candidates(market)
        pools = build_pools(market, candidates, regimes, market.first, market.last)
        now, data_end = datetime.now(), market.date(market.last)
        with Session(bind=main_engine) as s:
            s.execute(sa.update(tables.RegimePool).values(ACTIVE=False))
            for key, rows in pools.items():
                for rank, row in enumerate(rows, 1):
                    s.add(tables.RegimePool(CREATED_AT=now, DATA_END=data_end, REGIME=key, RANK=rank,
                                            ACTIVE=True, **row))
            s.commit()
        text = pools_text(pools)
        logger.info(text)
        if notify:
            from elements.notification.worker import TelegramNotification as tlg
            tlg.send_message('📦 Пулы стратегий по режимам обновлены:\n' + text)
        return pools

    @staticmethod
    def active(regime: str) -> list[dict]:
        try:
            return [dict(r) for r in SQLmain.run_sql_command(
                'SELECT * FROM regime_pools WHERE ACTIVE AND REGIME = :r ORDER BY RANK', {'r': regime})]
        except sa.exc.OperationalError:
            return []


def strategy_label(name: str) -> str:
    """ rule:momentum -> momentum, gen:bull:12 -> evolve:12 (метка STRATEGY в сигналах)."""
    if name.startswith('rule:'):
        return name[5:]
    if name.startswith('gen:'):
        return f'evolve:{name.rsplit(":", 1)[1]}'
    return name


def pools_text(pools: dict[str, list[dict]]) -> str:
    lines = []
    for key in REGIME_KEYS:
        info, rows = regime_info(key), pools.get(key, [])
        members = ', '.join(f'{strategy_label(r["STRATEGY"])} ({r["AVG_TRADE"]:+.2%}, {r["TRADES"]} сд.)'
                            for r in rows) or 'пусто — только фонды'
        lines.append(f'{info["emoji"]} {info["title"]}: {members}')
    return '\n'.join(lines)
