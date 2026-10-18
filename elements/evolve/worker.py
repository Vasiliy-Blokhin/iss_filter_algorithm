""" Абстрактный алгоритм: непредсказуемый рекурсивный поиск геномов со слепой проверкой.

Шаг поиска:
  1. зерно шага — из системного источника случайности (secrets), пишется в журнал:
     следующий кандидат заранее не известен, но любой шаг воспроизводим по зерну;
  2. родитель — зафиксированный этап (чаще лучшие), геном из пула лучших за сессию или
     ничего (тогда новый случайный геном); доноры для скрещивания — этапы и пул;
  3. оператор (jitter / swap / grow / prune / cross / graft / storm / fresh, см. genome.py)
     выбирается случайно с весами по его истории успехов (потомок лучше родителя);
  4. оценка на периоде поиска; потомок, лучший родителя, занимает место в пуле — от него
     поиск продолжается рекурсивно;
  5. фиксация этапа: CAGR >= EVOLVE_MIN_CAGR на периоде поиска, обе половины в плюсе,
     сделок >= EVOLVE_MIN_TRADES и CAGR >= EVOLVE_MIN_CAGR на контрольном периоде, который
     поиск не видит. Если кандидат похож на действующий этап (корреляция дневной доходности
     >= EVOLVE_SIMILAR), это корректировка: он фиксируется, только если лучше этого этапа, и
     заменяет его; непохожий кандидат добавляется новым этапом.
При новых данных все действующие этапы перепроверяются; не прошедшие — снимаются (ACTIVE = 0).
Действующих этапов не больше EVOLVE_LIVE_STAGES (3): новый этап, не попадающий в топ, не
фиксируется, а вытесненный из топа лучшим — снимается. Поиск, сигналы, бот и отчёты видят только их.
"""
import json
import logging
import random
import secrets
from datetime import datetime
from time import monotonic, sleep
from zoneinfo import ZoneInfo

import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import Session

from elements.algorithm.worker import Algorithm
from elements.evolve.genome import Genome, OPERATORS, OperatorStats
from elements.evolve.market import EMarket
from elements.notification.worker import TelegramNotification as tlg
from elements.selftune.worker import parse_windows, current_window, seconds_to_next_window
from source.settings.settings import (
    handler, TIMEZONE, EVOLVE_DAYS, EVOLVE_WINDOWS, EVOLVE_MIN_CAGR, EVOLVE_MIN_TRADES, EVOLVE_POOL,
    EVOLVE_SIMILAR, EVOLVE_LIVE_STAGES, SELFTUNE_MARGIN,
)
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

SERVICE = 'evolve'
STAGE_PARENT_SHARE = 0.45  # доля шагов, где родитель — зафиксированный этап
POOL_PARENT_SHARE = 0.45   # ... — геном из пула; остальное — новый случайный геном
STALL_STEPS = 200        # шагов без улучшения лучшего в пуле до перезапуска пула
ACTIVITY_PENALTY = 0.10    # вычет из оценки при нуле сделок (линейно до EVOLVE_MIN_TRADES)


def fitness(result: dict) -> float:
    """ Оценка для отбора: CAGR + штраф за просадку, минус штраф за малое число сделок
    (иначе поиск сползает к «почти не торговать» — на этом рынке это лучший нулевой результат)."""
    stats = result['stats']
    shortfall = max(0.0, 1.0 - stats['TRADES'] / max(1, EVOLVE_MIN_TRADES))
    return stats['SCORE'] - ACTIVITY_PENALTY * shortfall + min(0.0, *(h['CAGR'] for h in result['halves']))


def new_session() -> dict:
    return {'steps': 0, 'improved': 0, 'stages': [], 'replaced': [], 'restarts': 0, 'best': None}


def search_ok(result: dict) -> bool:
    stats = result['stats']
    return (stats['CAGR'] >= EVOLVE_MIN_CAGR and stats['TRADES'] >= EVOLVE_MIN_TRADES
            and all(half['CAGR'] > 0 for half in result['halves']))


def holdout_ok(result: dict) -> bool:
    return result['holdout']['CAGR'] >= EVOLVE_MIN_CAGR


def describe(result: dict) -> str:
    stats = result['stats']
    text = (f'поиск CAGR {stats["CAGR"]:+.1%}, просадка {stats["MAX_DRAWDOWN"]:.1%}, '
            f'сделок {stats["TRADES"]}, половины {result["halves"][0]["CAGR"]:+.1%}/'
            f'{result["halves"][1]["CAGR"]:+.1%}')
    if 'holdout' in result:
        text += f'; контроль CAGR {result["holdout"]["CAGR"]:+.1%} ({result["holdout"]["TRADES"]} сд.)'
    return text


class Evolver(SQLmain):

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed if seed is not None else secrets.randbits(64))
        self.market: EMarket | None = None
        self.loaded_end = None
        self.pool: list[dict] = []  # {'genome', 'result'} — лучшие за сессию
        self.seen: set[str] = set()
        self.stall = 0
        self.daily_cache: dict = {}  # id этапа -> дневная доходность на периоде поиска
        self.ops = OperatorStats(self.operator_history())
        self.session = new_session()

    # --- данные и этапы -------------------------------------------------------------
    def ensure_market(self, panel=None) -> bool:
        last = self.run_sql_command('SELECT MAX("DATE") AS last FROM candles')[0]['last']
        if self.market is not None and last == self.loaded_end:
            return False
        if panel is None:
            panel = Algorithm.build_panel(Algorithm.complete_candles())
        self.market = EMarket(panel)
        self.loaded_end = last
        self.pool, self.seen, self.daily_cache = [], set(), {}
        self.revalidate()
        logger.info(f'Абстрактный алгоритм: поиск {self.market.date(self.market.first)}..'
                    f'{self.market.date(self.market.search_end)}, контроль с '
                    f'{self.market.date(self.market.hold_start)}')
        return True

    @staticmethod
    def stages(active: bool = True) -> list[dict]:
        query = 'SELECT * FROM evolve_stages' + (' WHERE ACTIVE' if active else '') + ' ORDER BY SCORE DESC'
        return SQLmain.run_sql_command(query)

    @staticmethod
    def operator_history() -> dict[str, tuple[int, int]]:
        try:
            rows = SQLmain.run_sql_command(
                'SELECT OPERATOR, COUNT(*) AS N, SUM(IMPROVED) AS W FROM evolve_runs GROUP BY OPERATOR')
        except Exception:
            return {}
        return {row['OPERATOR']: (int(row['N']), int(row['W'] or 0)) for row in rows}

    def revalidate(self):
        """ Перепроверить действующие этапы на свежих данных: не прошедшие порог — снять."""
        for stage in self.stages():
            result = self.market.evaluate(Genome.from_dict(json.loads(stage['GENOME'])), holdout=True)
            keep = search_ok(result) and holdout_ok(result)
            values = self.stage_values(result)
            values.update(ACTIVE=keep)
            if not keep:
                values['NOTE'] = (f'снят {datetime.now():%Y-%m-%d}: {describe(result)}; '
                                  + (stage['NOTE'] or ''))[:500]
                logger.info(f'Этап {stage["id"]} снят: {describe(result)}')
            with Session(bind=main_engine) as s:
                s.execute(sa.update(tables.EvolveStage).where(tables.EvolveStage.id == stage['id'])
                          .values(**values))
                s.commit()
        self.keep_top()

    def keep_top(self):
        """ Действующими остаются только EVOLVE_LIVE_STAGES лучших этапов, остальные снимаются."""
        stages = self.stages()
        extra = stages[EVOLVE_LIVE_STAGES:]
        if not extra:
            return
        with Session(bind=main_engine) as s:
            for stage in extra:
                s.execute(
                    sa.update(tables.EvolveStage)
                    .where(tables.EvolveStage.id == stage['id'])
                    .values(
                        ACTIVE=False,
                        NOTE=(f'вытеснен из топ-{EVOLVE_LIVE_STAGES} {datetime.now():%Y-%m-%d}; '
                              + (stage['NOTE'] or ''))[:500]
                    )
                )
                logger.info(f'Этап {stage["id"]} вытеснен из топ-{EVOLVE_LIVE_STAGES}')
            s.commit()

    @staticmethod
    def stage_values(result: dict) -> dict:
        stats, holdout = result['stats'], result['holdout']
        return {
            'DATA_END': result['data_end'], 'HOLDOUT_START': result['holdout_start'],
            'CAGR': stats['CAGR'], 'MAX_DRAWDOWN': stats['MAX_DRAWDOWN'], 'SHARPE': stats['SHARPE'],
            'TRADES': stats['TRADES'], 'SCORE': fitness(result),
            'H1_CAGR': result['halves'][0]['CAGR'], 'H2_CAGR': result['halves'][1]['CAGR'],
            'HOLD_CAGR': holdout['CAGR'], 'HOLD_DD': holdout['MAX_DRAWDOWN'],
            'HOLD_TRADES': holdout['TRADES'],
        }

    # --- шаг поиска -----------------------------------------------------------------
    def pick_parent(self, rng, stages: list[dict]) -> tuple[Genome | None, float | None, str | None, int | None]:
        roll = rng.random()
        if stages and roll < STAGE_PARENT_SHARE:
            weights = [1.0 / (rank + 1) for rank in range(len(stages))]  # лучшие — чаще
            stage = rng.choices(stages, weights=weights)[0]
            return Genome.from_dict(json.loads(stage['GENOME'])), stage['SCORE'], f'stage:{stage["id"]}', stage['id']
        if self.pool and roll < STAGE_PARENT_SHARE + POOL_PARENT_SHARE:
            entrant = max(rng.sample(self.pool, k=min(3, len(self.pool))),
                          key=lambda item: fitness(item['result']))  # турнир
            return entrant['genome'], fitness(entrant['result']), 'pool', None
        return None, None, None, None

    def step(self) -> dict:
        started = monotonic()
        seed = secrets.randbits(32) ^ self.rng.getrandbits(32)
        rng = random.Random(seed)
        stages = self.stages()
        donors = [Genome.from_dict(json.loads(s['GENOME'])) for s in stages[:20]] + \
                 [item['genome'] for item in self.pool]
        parent, parent_score, parent_label, parent_stage = self.pick_parent(rng, stages)
        name = 'fresh' if parent is None else self.ops.choose(rng)
        child = None
        for _ in range(10):
            child = OPERATORS[name][0](rng, self.market, parent, donors).repair(rng, self.market)
            if child.key() not in self.seen:
                break
        self.seen.add(child.key())

        result = self.market.evaluate(child)
        score = fitness(result)
        improved = parent_score is not None and score > parent_score
        if parent is not None:
            self.ops.record(name, improved)
        self.remember(child, result)

        stage_id = None
        if search_ok(result) and not self.run_sql_command(
                'SELECT id FROM evolve_stages WHERE GENOME_KEY = :key', {'key': child.key()}):
            # Похожий на действующий этап (по дневной доходности) — это его корректировка:
            # фиксируется, только если лучше него, и заменяет его. Непохожий — новый этап.
            rival, similarity = self.nearest_stage(result['daily'], stages)
            if rival is not None and similarity < EVOLVE_SIMILAR:
                rival = None
            # Новый (непохожий) этап при полном топе должен быть лучше худшего из топа.
            if rival is not None:
                bar = rival['SCORE']
            elif len(stages) >= EVOLVE_LIVE_STAGES:
                bar = stages[EVOLVE_LIVE_STAGES - 1]['SCORE']
            else:
                bar = None
            if bar is None or score > bar + SELFTUNE_MARGIN:
                result = self.market.evaluate(child, holdout=True)
                if holdout_ok(result):
                    stage_id = self.fix_stage(child, result, rival['id'] if rival else parent_stage,
                                              name, seed, replaces=rival)
                    self.keep_top()
        self.session['steps'] += 1
        self.session['improved'] += int(improved)
        best = self.session['best']
        if best is None or score > fitness(best):
            self.session['best'] = result
        self.log_run(child, result, name, seed, parent_label, improved, stage_id, monotonic() - started)
        return result

    def remember(self, genome: Genome, result: dict):
        """ Пул лучших геномов сессии — от них поиск продолжается рекурсивно. Геномы с тем же
        результатом (отличаются только «немыми» генами) в пул не берутся — он не вырождается.
        Долго нет улучшения лучшего в пуле — пул сбрасывается до двух лучших (перезапуск)."""
        best = fitness(self.pool[0]['result']) if self.pool else None
        signature = (round(fitness(result), 6), result['stats']['TRADES'])
        if all((round(fitness(item['result']), 6), item['result']['stats']['TRADES']) != signature
               for item in self.pool):
            self.pool.append({'genome': genome, 'result': result})
            self.pool.sort(key=lambda item: -fitness(item['result']))
            del self.pool[EVOLVE_POOL:]
        self.stall = 0 if best is None or fitness(result) > best else self.stall + 1
        if self.stall >= STALL_STEPS:
            self.pool, self.stall = self.pool[:2], 0
            self.session['restarts'] += 1

    def stage_daily(self, stage: dict):
        if stage['id'] not in self.daily_cache:
            genome = Genome.from_dict(json.loads(stage['GENOME']))
            self.daily_cache[stage['id']] = self.market.evaluate(genome)['daily']
        return self.daily_cache[stage['id']]

    def nearest_stage(self, daily, stages: list[dict]) -> tuple[dict | None, float]:
        """ Самый похожий действующий этап: корреляция дневной доходности на периоде поиска."""
        best, best_corr = None, -1.0
        for stage in stages:
            other = self.stage_daily(stage)
            if daily.std() == 0 or other.std() == 0:
                continue
            corr = float(np.corrcoef(daily, other)[0, 1])
            if corr > best_corr:
                best, best_corr = stage, corr
        return best, best_corr

    def fix_stage(self, genome: Genome, result: dict, parent_id, operator, seed,
                  replaces: dict | None = None) -> int:
        with Session(bind=main_engine) as s:
            stage = tables.EvolveStage(
                CREATED_AT=datetime.now(), GENOME=json.dumps(genome.to_dict(), sort_keys=True),
                GENOME_KEY=genome.key(), PARENT_ID=parent_id, OPERATOR=operator, SEED=seed,
                ACTIVE=True, NOTE=genome.describe()[:500], **self.stage_values(result),
            )
            s.add(stage)
            s.flush()
            stage_id = stage.id
            if replaces is not None:
                s.execute(sa.update(tables.EvolveStage).where(tables.EvolveStage.id == replaces['id'])
                          .values(ACTIVE=False, NOTE=(f'заменён этапом {stage_id}; '
                                                      + (replaces['NOTE'] or ''))[:500]))
            s.commit()
        self.daily_cache[stage_id] = result['daily']
        logger.info(f'Абстрактный алгоритм: этап {stage_id}'
                    + (f' заменил этап {replaces["id"]}' if replaces else ' — новый')
                    + f', оператор {operator}: {describe(result)}. {genome.describe()}')
        self.session['replaced' if replaces else 'stages'].append(stage_id)
        return stage_id

    def log_run(self, genome, result, operator, seed, parent, improved, stage_id, seconds):
        stats = result['stats']
        self.append_data({
            'CREATED_AT': datetime.now(), 'OPERATOR': operator, 'SEED': seed, 'PARENT': parent,
            'GENOME': json.dumps(genome.to_dict(), sort_keys=True),
            'CAGR': stats['CAGR'], 'MAX_DRAWDOWN': stats['MAX_DRAWDOWN'], 'TRADES': stats['TRADES'],
            'SCORE': fitness(result), 'H1_CAGR': result['halves'][0]['CAGR'],
            'H2_CAGR': result['halves'][1]['CAGR'],
            'HOLD_CAGR': result['holdout']['CAGR'] if 'holdout' in result else None,
            'IMPROVED': improved, 'STAGE_ID': stage_id, 'SECONDS': seconds,
        }, tables.EvolveRun)

    def session_report(self) -> str:
        session = self.session
        lines = [f'Абстрактный алгоритм, воскресная сессия: проверено геномов {session["steps"]}, '
                 f'улучшений родителя {session["improved"]}, новых этапов {len(session["stages"])}, '
                 f'скорректированных (заменённых лучшими) {len(session["replaced"])}, '
                 f'перезапусков пула {session["restarts"]}.']
        if session['best']:
            lines.append(f'Лучший геном сессии: {describe(session["best"])}.')
        stages = self.stages()
        lines.append(f'Действующие этапы (топ-{EVOLVE_LIVE_STAGES}): {len(stages)}.')
        for stage in stages[:EVOLVE_LIVE_STAGES]:
            lines.append(f'• этап {stage["id"]}: поиск CAGR {stage["CAGR"]:+.1%}, просадка '
                         f'{stage["MAX_DRAWDOWN"]:.1%}, сделок {stage["TRADES"]}; контроль '
                         f'CAGR {stage["HOLD_CAGR"]:+.1%} ({stage["HOLD_TRADES"]} сд.)')
        weights = sorted(((self.ops.weight(n), n) for n in OPERATORS), reverse=True)[:3]
        lines.append('Самые успешные операторы: ' + ', '.join(name for _, name in weights) + '.')
        return '\n'.join(lines)


def run_daemon(days: str = EVOLVE_DAYS, spec: str = EVOLVE_WINDOWS):
    weekdays = {int(day) for day in days.split(',') if day.strip()}
    windows = parse_windows(spec)
    evolver = None
    active = False
    logger.info(f'Абстрактный алгоритм запущен: дни {sorted(weekdays)}, окна {spec}')
    while True:
        now = datetime.now(ZoneInfo(TIMEZONE))
        window = current_window(now, windows) if now.weekday() in weekdays else None
        if window is None:
            if active:  # окно закончилось — сводка сессии
                active = False
                try:
                    tlg.send_message(evolver.session_report())
                except Exception as error:
                    logger.exception(error)
            SQLmain.save_activity(SERVICE, 'ожидание окна поиска')
            sleep(min(seconds_to_next_window(now, windows), 300))
            continue
        try:
            if evolver is None:
                evolver = Evolver()
            if not active:
                evolver.session = new_session()
                active = True
            evolver.ensure_market()
            until = monotonic() + 60
            while monotonic() < until:
                evolver.step()
            SQLmain.save_activity(SERVICE, f'шагов {evolver.session["steps"]}, '
                                           f'этапов {len(evolver.session["stages"])}')
        except Exception as error:
            logger.exception(error)
            sleep(30)
