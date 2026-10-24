""" Абстрактный алгоритм: непредсказуемый рекурсивный поиск геномов со слепой проверкой.

Шаг поиска:
  1. зерно шага — из системного источника случайности (secrets), пишется в журнал:
     следующий кандидат заранее не известен, но любой шаг воспроизводим по зерну;
  2. родитель — зафиксированный этап (чаще лучшие, 35%), геном из пула лучших за сессию (35%)
     или ничего (30%) — тогда новый геном: случайный набор генов с затравкой 1–3 успешных
     генов или их параметров от доноров (seeded, 80%) либо совсем случайный (fresh);
     доноры для скрещивания и затравки — этапы и пул;
  3. оператор (jitter / swap / grow / prune / cross / graft / storm / fresh, см. genome.py)
     выбирается случайно с весами по его истории успехов (потомок лучше родителя);
  4. оценка на периоде поиска; потомок, лучший родителя, занимает место в пуле — от него
     поиск продолжается рекурсивно;
  5. фиксация этапа: CAGR >= EVOLVE_MIN_CAGR на периоде поиска, обе половины в плюсе,
     сделок >= EVOLVE_MIN_TRADES и CAGR >= EVOLVE_MIN_CAGR на контрольном периоде, который
     поиск не видит. Если кандидат похож на действующий этап (корреляция дневной доходности
     >= EVOLVE_SIMILAR), это корректировка: он фиксируется, только если лучше этого этапа, и
     заменяет его; непохожий кандидат добавляется новым этапом. Защита от подгонки: кандидат,
     вытесняющий этап (похожий или худший из полного топа), должен иметь контроль не хуже его.
     Оценка штрафует редкую торговлю: EVOLVE_ACTIVITY_PENALTY × нехватка сделок до
     EVOLVE_TARGET_TRADES в год.
     Устойчивость (практики ГА против подгонки, 2026-10-24): оценка — по календарным годам
     (средний год без лучшего + штраф за убыточный худший), в плюсе >= EVOLVE_MIN_POS_YEARS
     годов, штраф за лишние гены; перед фиксацией — Deflated Sharpe с учётом всех испытаний
     (>= EVOLVE_MIN_DSR) и плато соседей (сдвиг одного параметра); пул держит ниши (crowding).
При новых данных все действующие этапы перепроверяются; не прошедшие — снимаются (ACTIVE = 0).
Действующих этапов не больше EVOLVE_LIVE_STAGES (3): новый этап, не попадающий в топ, не
фиксируется, а вытесненный из топа лучшим — снимается. Поиск, сигналы, бот и отчёты видят только их.
"""
import json
import logging
import math
import random
import secrets
from statistics import NormalDist
from datetime import datetime, timedelta
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
    handler, TIMEZONE, DATA_DIR, EVOLVE_DAYS, EVOLVE_WINDOWS, EVOLVE_DAILY_DAYS, EVOLVE_DAILY_WINDOW, EVOLVE_MIN_CAGR, EVOLVE_MIN_TRADES, EVOLVE_POOL,
    EVOLVE_SIMILAR, EVOLVE_LIVE_STAGES, EVOLVE_TARGET_TRADES, EVOLVE_ACTIVITY_PENALTY, SELFTUNE_MARGIN,
    EVOLVE_WORST_YEAR, EVOLVE_MIN_POS_YEARS, EVOLVE_PARSIMONY, EVOLVE_NEIGHBORS, EVOLVE_NEIGHBOR_SHARE,
    EVOLVE_MIN_DSR, REGIME_MIN_TRADES, EVOLVE_BATCH, EVOLVE_FRESH_MIN,
)
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

SERVICE = 'evolve'
STAGE_PARENT_SHARE = 0.35  # доля шагов, где родитель — зафиксированный этап
POOL_PARENT_SHARE = 0.35   # ... — геном из пула; остальное — новый геном
SEEDED_SHARE = 0.8         # новый геном: с затравкой успешных генов (seeded), иначе fresh
STALL_STEPS = 200        # шагов без улучшения лучшего в пуле до перезапуска пула
EULER_GAMMA = 0.5772156649
SHARPE_SAMPLE = 3000     # дневных Шарпов последних геномов — дисперсия испытаний для DSR


def target_trades(result: dict) -> float:
    """ Цель по числу сделок на периоде поиска: EVOLVE_TARGET_TRADES в год (не меньше EVOLVE_MIN_TRADES)."""
    start, end = result.get('search_start'), result.get('search_end')
    years = (end - start).days / 365.25 if start and end else 1.0
    # Геном режима рынка входит только в дни режима — цель пропорциональна их доле.
    return max(min_trades(result), EVOLVE_TARGET_TRADES * years * result.get('day_share', 1.0))


def min_trades(result: dict) -> float:
    return max(REGIME_MIN_TRADES, EVOLVE_MIN_TRADES * result.get('day_share', 1.0))


def min_cagr(result: dict) -> float:
    """ Порог CAGR: у генома режима доходность «разбавлена» днями вне режима."""
    return EVOLVE_MIN_CAGR * result.get('day_share', 1.0)


def robust_core(result: dict) -> float:
    """ Устойчивая доходность: средний год без лучшего (результат не держится на одном
    удачном годе) + EVOLVE_WORST_YEAR × худший год, если он убыточен. Без годов — CAGR."""
    years = sorted(result.get('years') or [])
    if len(years) < 3:
        return result['stats']['CAGR']
    return float(np.mean(years[:-1])) + EVOLVE_WORST_YEAR * min(0.0, years[0])


def fitness(result: dict) -> float:
    """ Оценка для отбора: устойчивая доходность по годам + штраф за просадку, минус штраф за
    малое число сделок — EVOLVE_ACTIVITY_PENALTY × нехватка сделок до цели (упор на частую
    торговлю: иначе поиск сползает к редким сделкам), минус провал худшей половины периода и
    EVOLVE_PARSIMONY за каждый ген сверх минимума (простые правила реже подогнаны)."""
    stats = result['stats']
    shortfall = max(0.0, 1.0 - stats['TRADES'] / target_trades(result))
    return (robust_core(result) + (stats['SCORE'] - stats['CAGR'])
            - EVOLVE_ACTIVITY_PENALTY * shortfall + min(0.0, *(h['CAGR'] for h in result['halves']))
            - EVOLVE_PARSIMONY * result.get('extra_genes', 0))


def new_session() -> dict:
    return {'steps': 0, 'improved': 0, 'stages': [], 'replaced': [], 'restarts': 0, 'best': None,
            'rejected': {'neighbors': 0, 'dsr': 0, 'periods': 0}}


def positive_years(result: dict) -> float:
    years = result.get('years') or []
    return sum(year > 0 for year in years) / len(years) if years else 1.0


def search_ok(result: dict) -> bool:
    stats = result['stats']
    return (stats['CAGR'] >= min_cagr(result) and stats['TRADES'] >= min_trades(result)
            and all(half['CAGR'] > 0 for half in result['halves'])
            and (result.get('day_share', 1.0) < 0.5 or positive_years(result) >= EVOLVE_MIN_POS_YEARS))


def daily_sharpe(daily: np.ndarray) -> float:
    std = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    return float(np.mean(daily)) / std if std > 0 else 0.0


def deflated_sharpe(daily: np.ndarray, trials: int, sr_var: float) -> float:
    """ Deflated Sharpe (Bailey, López de Prado, 2014): вероятность, что дневной Шарп выше
    ожидаемого максимума из trials случайных испытаний с дисперсией Шарпа sr_var."""
    count = len(daily)
    std = float(np.std(daily, ddof=1)) if count > 1 else 0.0
    if count < 30 or std == 0:
        return 0.0
    sr = float(np.mean(daily)) / std
    centered = daily - np.mean(daily)
    skew = float(np.mean(centered ** 3)) / std ** 3
    kurt = float(np.mean(centered ** 4)) / std ** 4
    normal = NormalDist()
    expected_max = 0.0
    if trials >= 2:
        expected_max = math.sqrt(max(0.0, sr_var)) * (
            (1 - EULER_GAMMA) * normal.inv_cdf(1 - 1 / trials)
            + EULER_GAMMA * normal.inv_cdf(1 - 1 / (trials * math.e)))
    denominator = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denominator <= 0:
        return 0.0
    return normal.cdf((sr - expected_max) * math.sqrt(count - 1) / math.sqrt(denominator))


def holdout_guard(result: dict, displaced: dict | None, threshold: float) -> bool:
    """Защита от подгонки: контроль не ниже порога и не хуже, чем у вытесняемого этапа.
    Геном режима, у которого в контроле почти нет дней режима (< REGIME_MIN_TRADES / 2 сделок),
    контролем не проверяется — проверить нечем."""
    holdout = result['holdout']['CAGR']
    if result.get('day_share', 1.0) < 1.0 and result['holdout']['TRADES'] < REGIME_MIN_TRADES // 2:
        return True
    if holdout < threshold * result.get('day_share', 1.0):
        return False
    if displaced is None:
        return True
    previous = displaced.get('HOLD_CAGR')
    return previous is None or holdout >= previous


def holdout_ok(result: dict) -> bool:
    if result.get('day_share', 1.0) < 1.0 and result['holdout']['TRADES'] < REGIME_MIN_TRADES // 2:
        return True
    return result['holdout']['CAGR'] >= min_cagr(result)


def describe(result: dict) -> str:
    stats = result['stats']
    text = (f'поиск CAGR {stats["CAGR"]:+.1%}, просадка {stats["MAX_DRAWDOWN"]:.1%}, '
            f'сделок {stats["TRADES"]}, половины {result["halves"][0]["CAGR"]:+.1%}/'
            f'{result["halves"][1]["CAGR"]:+.1%}')
    if 'holdout' in result:
        text += f'; контроль CAGR {result["holdout"]["CAGR"]:+.1%} ({result["holdout"]["TRADES"]} сд.)'
    if 'blind' in result:
        blind = result['blind']
        text += f'; слепая {blind["TOTAL_RETURN"]:+.1%} за период, CAGR {blind["CAGR"]:+.1%} ({blind["TRADES"]} сд.)'
    return text


class Evolver(SQLmain):
    regime = ''  # режим рынка экземпляра (см. __init__); у класса — общий

    def __init__(self, seed: int | None = None, regime: str | None = None):
        # regime — ключ режима рынка (elements/regime): геномы входят только в дни режима,
        # этапы и топ — свои для режима; None — общий поиск.
        self.regime = regime or ''
        self.batch_step = 0  # номер шага в пачке EVOLVE_BATCH (первые EVOLVE_FRESH_MIN — новые геномы)
        self.meter = None
        self.rng = random.Random(seed if seed is not None else secrets.randbits(64))
        self.market: EMarket | None = None
        self.loaded_end = None
        self.pool: list[dict] = []  # {'genome', 'result'} — лучшие за сессию
        self.seen: set[str] = set()
        self.stall = 0
        self.daily_cache: dict = {}  # id этапа -> дневная доходность на периоде поиска
        self.ops = OperatorStats(self.operator_history())
        self.session = new_session()
        self.stage_share = STAGE_PARENT_SHARE
        # Испытания для поправки на множественность (DSR): все когда-либо оценённые геномы.
        self.trials = self.trial_count()
        self.sharpes: list[float] = []

    # --- данные и этапы -------------------------------------------------------------
    def ensure_market(self, panel=None) -> bool:
        last = self.run_sql_command('SELECT MAX("DATE") AS last FROM candles')[0]['last']
        if self.market is not None and last == self.loaded_end:
            return False
        if panel is None:
            panel = Algorithm.build_panel(Algorithm.complete_candles())
        self.market = EMarket(panel)
        if self.regime:
            from elements.regime.pools import regimes_by_day
            from elements.regime.worker import MarketRegime
            self.market.set_days(regimes_by_day(self.market, MarketRegime.history(panel)) == self.regime)
        from elements.evolve.periods import PeriodMeter
        self.meter = PeriodMeter(self.market)
        self.loaded_end = last
        self.pool, self.seen, self.daily_cache = [], set(), {}
        self.revalidate()
        logger.info(f'Абстрактный алгоритм: поиск {self.market.date(self.market.first)}..'
                    f'{self.market.date(self.market.search_end)}, контроль с '
                    f'{self.market.date(self.market.hold_start)}, слепая проверка с '
                    f'{self.market.date(self.market.blind_start)}')
        return True

    def stages(self=None, active: bool = True, regime: str | None = None) -> list[dict]:
        """ Этапы (по умолчанию действующие) режима — своего у экземпляра, '' — общие, при вызове
        с класса без regime — все. Порядок — основная метрика PERIOD_SCORE, затем SCORE."""
        if regime is None and isinstance(self, Evolver):
            regime = self.regime
        where = ['ACTIVE'] if active else []
        if regime is not None:
            where.append("COALESCE(REGIME, '') = :regime")
        query = ('SELECT * FROM evolve_stages' + (' WHERE ' + ' AND '.join(where) if where else '')
                 + ' ORDER BY PERIOD_SCORE IS NULL, PERIOD_SCORE DESC, SCORE DESC')
        return SQLmain.run_sql_command(query, {'regime': regime or ''})

    @staticmethod
    def operator_history() -> dict[str, tuple[int, int]]:
        try:
            rows = SQLmain.run_sql_command(
                'SELECT OPERATOR, COUNT(*) AS N, SUM(IMPROVED) AS W FROM evolve_runs GROUP BY OPERATOR')
        except Exception:
            return {}
        return {row['OPERATOR']: (int(row['N']), int(row['W'] or 0)) for row in rows}

    @staticmethod
    def trial_count() -> int:
        try:
            return int(SQLmain.run_sql_command('SELECT COUNT(*) AS n FROM evolve_runs')[0]['n'])
        except Exception:
            return 0

    def note_sharpe(self, result: dict):
        """ Дневной Шарп оценённого генома — выборка для дисперсии Шарпа испытаний."""
        self.trials += 1
        if result['stats']['TRADES'] >= EVOLVE_MIN_TRADES:
            self.sharpes.append(daily_sharpe(result['daily']))
            del self.sharpes[:-SHARPE_SAMPLE]

    def dsr(self, result: dict) -> float:
        """ DSR при нулевой гипотезе «у всех испытаний истинный Шарп 0»: дисперсия оценки
        1/(T−1). Разброс Шарпов самих геномов (self.sharpes) сюда не берётся — он раздут
        разной рыночной бетой и делает порог недостижимым (дневной Шарп ~0.28). Число
        испытаний — все оценённые геномы: они сильно коррелированы, так что оценка строгая."""
        return deflated_sharpe(result['daily'], self.trials, 1.0 / max(1, len(result['daily']) - 1))

    def neighbors_ok(self, rng, genome: Genome, score: float) -> tuple[bool, float]:
        """ Плато, а не пик: медиана оценки EVOLVE_NEIGHBORS соседей (сдвиг одного параметра)
        не ниже EVOLVE_NEIGHBOR_SHARE × оценки кандидата."""
        scores = []
        for _ in range(EVOLVE_NEIGHBORS):
            neighbour = OPERATORS['jitter'][0](rng, self.market, genome, []).repair(rng, self.market)
            result = self.market.evaluate(neighbour)
            self.note_sharpe(result)
            scores.append(fitness(result))
        median = float(np.median(scores)) if scores else score
        return median >= EVOLVE_NEIGHBOR_SHARE * score, median

    def revalidate(self):
        """ Перепроверить действующие этапы на свежих данных: не прошедшие порог — снять."""
        for stage in self.stages():
            result = self.market.evaluate(Genome.from_dict(json.loads(stage['GENOME'])), holdout=True)
            keep = search_ok(result) and holdout_ok(result)
            values = self.stage_values(result)
            values.update(self.period_values(Genome.from_dict(json.loads(stage['GENOME']))))
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
                        NOTE=(f'вытеснен из топ-{EVOLVE_LIVE_STAGES}{" режима " + self.regime if self.regime else ""} '
                              f'{datetime.now():%Y-%m-%d}; '
                              + (stage['NOTE'] or ''))[:500]
                    )
                )
                logger.info(f'Этап {stage["id"]} вытеснен из топ-{EVOLVE_LIVE_STAGES}')
            s.commit()

    def period_values(self, genome: Genome) -> dict:
        """ Реалистичная статистика по периодам 1 мес. … 5 лет и оценка по ней (PERIOD_SCORE)."""
        if self.meter is None:
            return {}
        periods, score = self.meter.measure(genome)
        return {'PERIODS': json.dumps(periods), 'PERIOD_SCORE': score, 'REGIME': self.regime}

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
            'BLIND_START': result['blind_start'], 'BLIND_CAGR': result['blind']['CAGR'],
            'BLIND_RETURN': result['blind']['TOTAL_RETURN'], 'BLIND_DD': result['blind']['MAX_DRAWDOWN'],
            'BLIND_TRADES': result['blind']['TRADES'],
        }

    # --- шаг поиска -----------------------------------------------------------------
    def pick_parent(self, rng, stages: list[dict]) -> tuple[Genome | None, float | None, str | None, int | None]:
        roll = rng.random()
        if stages and roll < self.stage_share:
            weights = [1.0 / (rank + 1) for rank in range(len(stages))]  # лучшие — чаще
            stage = rng.choices(stages, weights=weights)[0]
            return Genome.from_dict(json.loads(stage['GENOME'])), stage['SCORE'], f'stage:{stage["id"]}', stage['id']
        if self.pool and roll < self.stage_share + POOL_PARENT_SHARE:
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
        # В каждой пачке шагов — несколько совсем новых геномов (без родителей и затравки).
        forced_fresh = self.batch_step % EVOLVE_BATCH < EVOLVE_FRESH_MIN
        self.batch_step += 1
        if forced_fresh:
            parent, parent_score, parent_label, parent_stage = None, None, None, None
            name = 'fresh'
        elif parent is None:
            name = 'seeded' if donors and rng.random() < SEEDED_SHARE else 'fresh'
        else:
            name = self.ops.choose(rng)
        child = None
        for _ in range(10):
            child = OPERATORS[name][0](rng, self.market, parent, donors).repair(rng, self.market)
            if child.key() not in self.seen:
                break
        self.seen.add(child.key())

        self.last_genome = child
        result = self.market.evaluate(child)
        self.note_sharpe(result)
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
            # Вытесняемый этап (похожий или худший из топа) задаёт и планку контроля.
            if rival is not None:
                displaced = rival
            elif len(stages) >= EVOLVE_LIVE_STAGES:
                displaced = stages[EVOLVE_LIVE_STAGES - 1]
            else:
                displaced = None
            if displaced is None or score > displaced['SCORE'] + SELFTUNE_MARGIN:
                result = self.market.evaluate(child, holdout=True)
                if holdout_guard(result, displaced, EVOLVE_MIN_CAGR):
                    # Поправка на число испытаний и устойчивость к сдвигу параметров.
                    dsr = self.dsr(result)
                    if dsr < EVOLVE_MIN_DSR:
                        self.session['rejected']['dsr'] += 1
                    else:
                        robust, median = self.neighbors_ok(rng, child, score)
                        if not robust:
                            self.session['rejected']['neighbors'] += 1
                        else:
                            periods = self.period_values(child)
                            previous = displaced.get('PERIOD_SCORE') if displaced else None
                            if previous is not None and periods.get('PERIOD_SCORE', 0.0) < previous:
                                self.session['rejected']['periods'] += 1
                            else:
                                note = (f'DSR {dsr:.2f}, соседи {median:+.3f} / {score:+.3f}, '
                                        f'по периодам {periods.get("PERIOD_SCORE", 0.0):+.3f}')
                                stage_id = self.fix_stage(child, result, rival['id'] if rival else parent_stage,
                                                          name, seed, replaces=rival, note=note, extra=periods)
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
        Ниши (crowding): геном, похожий по дневной доходности (>= EVOLVE_SIMILAR) на член пула,
        соревнуется только с ним и заменяет его, если лучше, — пул держит разные семейства
        правил, а не копии одного. Долго нет улучшения лучшего в пуле — пул сбрасывается до
        двух лучших (перезапуск)."""
        best = fitness(self.pool[0]['result']) if self.pool else None
        signature = (round(fitness(result), 6), result['stats']['TRADES'])
        if all((round(fitness(item['result']), 6), item['result']['stats']['TRADES']) != signature
               for item in self.pool):
            entry = {'genome': genome, 'result': result}
            twin = self.pool_twin(result['daily'])
            if twin is not None:
                if fitness(result) > fitness(self.pool[twin]['result']):
                    self.pool[twin] = entry
            else:
                self.pool.append(entry)
            self.pool.sort(key=lambda item: -fitness(item['result']))
            del self.pool[EVOLVE_POOL:]
        self.stall = 0 if best is None or fitness(result) > best else self.stall + 1
        if self.stall >= STALL_STEPS:
            self.pool, self.stall = self.pool[:2], 0
            self.session['restarts'] += 1

    def pool_twin(self, daily) -> int | None:
        """ Индекс самого похожего члена пула, если корреляция >= EVOLVE_SIMILAR."""
        if not self.pool or daily.std() == 0:
            return None
        others = np.array([item['result']['daily'] for item in self.pool])
        spread = others.std(axis=1)
        valid = spread > 0
        if not valid.any():
            return None
        centered = others[valid] - others[valid].mean(axis=1, keepdims=True)
        corr = centered @ (daily - daily.mean()) / (spread[valid] * daily.std() * len(daily))
        index = int(np.argmax(corr))
        return int(np.nonzero(valid)[0][index]) if corr[index] >= EVOLVE_SIMILAR else None

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
                  replaces: dict | None = None, note: str = '', extra: dict | None = None) -> int:
        with Session(bind=main_engine) as s:
            stage = tables.EvolveStage(
                CREATED_AT=datetime.now(), GENOME=json.dumps(genome.to_dict(), sort_keys=True),
                GENOME_KEY=genome.key(), PARENT_ID=parent_id, OPERATOR=operator, SEED=seed,
                ACTIVE=True, NOTE=((note + '; ' if note else '') + genome.describe())[:500],
                **self.stage_values(result), **(extra or {'REGIME': self.regime}),
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
            'BLIND_CAGR': result['blind']['CAGR'] if 'blind' in result else None,
            'IMPROVED': improved, 'STAGE_ID': stage_id, 'SECONDS': seconds, 'REGIME': self.regime,
        }, tables.EvolveRun)

    def session_report(self) -> str:
        """ Коротко: сколько проверено и найдено, топ этапов группы с итогом за год и 5 лет
        на реалистичном счёте (основная метрика), и слепой хвост."""
        session = self.session
        lines = [f'🧬 {"Режим " + self.regime if self.regime else "Общий поиск"}: проверено {session["steps"]}, '
                 f'новых этапов {len(session["stages"])}, улучшено {len(session["replaced"])}.']
        for stage in self.stages()[:EVOLVE_LIVE_STAGES]:
            periods = json.loads(stage.get('PERIODS') or '{}')
            year, five = periods.get('1y', {}), periods.get('5y', {})
            lines.append(f'• #{stage["id"]}: за год {year.get("TOTAL_RETURN", 0):+.1%}, за 5 лет '
                         f'{five.get("TOTAL_RETURN", 0):+.1%}, сделок {stage["TRADES"]}'
                         + (f'; вслепую {stage["BLIND_RETURN"]:+.1%}' if stage.get('BLIND_RETURN') is not None else ''))
        return '\n'.join(lines)


# Поэтапный прогон до заданного времени: (название, доля времени, множитель частоты новых
# генов NEW_GENES, доля шагов с родителем-этапом).
PHASES = [
    ('новые своды правил', 0.35, 4.0, 0.25),
    ('смешанный поиск', 0.30, 1.0, 0.45),
    ('рекурсивная доработка этапов', 0.35, 1.0, 0.70),
]


LOCK = DATA_DIR / 'evolve.lock'  # идёт поиск (этапы PHASES): время последнего шага внутри
LOCK_STALE = 600  # секунд без отметки — блокировка брошена (процесс упал)


def search_busy() -> bool:
    """ Идёт ли уже поэтапный поиск в другом процессе (два поиска сразу портят топ этапов)."""
    try:
        return LOCK.exists() and datetime.now().timestamp() - LOCK.stat().st_mtime < LOCK_STALE
    except OSError:
        return False


def touch_lock():
    LOCK.write_text(f'{datetime.now():%Y-%m-%d %H:%M:%S}', encoding='utf-8')


def release_lock():
    try:
        LOCK.unlink()
    except OSError:
        pass


def uses_new_genes(genome: Genome) -> bool:
    from elements.evolve.genes import NEW_GENES
    return any(gene.name in NEW_GENES for genes in genome.genes.values() for gene in genes)


def search_groups() -> list[str]:
    """ Группы поиска: общий ('') и каждый режим рынка — новый режим в реестре добавляется сам."""
    from elements.regime.detector import REGIME_KEYS
    return ['', *REGIME_KEYS]


def group_title(regime: str) -> str:
    if not regime:
        return 'общий'
    from elements.regime.detector import regime_info
    info = regime_info(regime)
    return f'{info["emoji"]} {info["title"]}'


def run_groups(until: datetime, groups: list[str] | None = None, phases=PHASES) -> list[str]:
    """ Поиск до until по группам (общий + режимы) поровну по времени; в Telegram — одна сводка."""
    from elements.evolve.genes import REGISTRY, NEW_GENES
    groups = search_groups() if groups is None else groups
    touch_lock()
    reports = []
    try:
        for number, regime in enumerate(groups):
            now = datetime.now(ZoneInfo(TIMEZONE))
            left = (until - now).total_seconds()
            if left <= 30:
                break
            end = now + timedelta(seconds=left / (len(groups) - number))
            reports += _run_phases(end, phases, REGISTRY, NEW_GENES, regime, notify=False)
    finally:
        release_lock()
    text = '🧬 Геномный поиск завершён.\n' + '\n'.join(reports)
    logger.info(text)
    print('ИТОГ: ' + text.replace('\n', '\nИТОГ: '))
    try:
        tlg.send_message(text)
    except Exception as error:
        logger.exception(error)
    return reports


def run_phases(until: datetime, phases=PHASES, regime: str | None = None) -> list[str]:
    """ Этапы поиска до момента until (МСК): у каждого своя частота новых генов и доля
    рекурсии от зафиксированных этапов; после каждого — сводка в лог и Telegram."""
    from elements.evolve.genes import REGISTRY, NEW_GENES
    touch_lock()
    try:
        return _run_phases(until, phases, REGISTRY, NEW_GENES, regime)
    finally:
        release_lock()


def _run_phases(until, phases, REGISTRY, NEW_GENES, regime=None, notify=True) -> list[str]:
    evolver = Evolver(regime=regime)
    evolver.ensure_market()
    start = datetime.now(ZoneInfo(TIMEZONE))
    total = (until - start).total_seconds()
    reports = []
    steps = found = 0
    base_weights = {name: REGISTRY[name].weight for name in NEW_GENES}
    for title, share, boost, stage_share in phases:
        end = min(until, datetime.now(ZoneInfo(TIMEZONE)) + timedelta(seconds=total * share))
        for name in NEW_GENES:
            REGISTRY[name].weight = base_weights[name] * boost
        evolver.stage_share = stage_share
        evolver.session = new_session()
        new_tried, new_best = 0, None
        while datetime.now(ZoneInfo(TIMEZONE)) < end:
            try:
                result = evolver.step()
            except Exception as error:
                logger.exception(error)
                continue
            if uses_new_genes(evolver.last_genome):
                new_tried += 1
                if new_best is None or fitness(result) > fitness(new_best):
                    new_best = result
            if evolver.session['steps'] % 50 == 0:
                touch_lock()
                SQLmain.save_activity(SERVICE, f'{title}: шагов {evolver.session["steps"]}')
        report = (f'Этап «{title}» ({group_title(evolver.regime)}).\n' + evolver.session_report()
                  + f'\nГеномов с новыми генами: {new_tried}'
                  + (f', лучший: {describe(new_best)}' if new_best else ''))
        logger.info(report)
        reports.append(report)
        steps += evolver.session['steps']
        found += len(evolver.session['stages']) + len(evolver.session['replaced'])
        if notify:
            try:
                tlg.send_message(report)
            except Exception as error:
                logger.exception(error)
    for name, weight in base_weights.items():
        REGISTRY[name].weight = weight
    if notify:
        return reports
    # Для общей сводки по группам — одна строка.
    top = evolver.stages()[:EVOLVE_LIVE_STAGES]
    return [f'{group_title(evolver.regime)}: проверено {steps}, новых/улучшенных этапов {found}; топ: '
            + (', '.join(f'#{s["id"]} {s["PERIOD_SCORE"] or 0:+.1%}' for s in top) or 'нет')]


def pending_request() -> dict | None:
    rows = SQLmain.run_sql_command("SELECT * FROM evolve_requests WHERE STATUS = 'new' ORDER BY id LIMIT 1")
    return dict(rows[0]) if rows else None


def set_request(request_id: int, **values):
    with Session(bind=main_engine) as s:
        s.execute(sa.update(tables.EvolveRequest).where(tables.EvolveRequest.id == request_id).values(**values))
        s.commit()


def run_request(request: dict):
    """ Итерация поиска по запросу: этапы PHASES на MINUTES минут, сводки — в Telegram."""
    now = datetime.now(ZoneInfo(TIMEZONE))
    set_request(request['id'], STATUS='running', STARTED_AT=now.replace(tzinfo=None))
    tlg.send_message(f'Геномный поиск по запросу #{request["id"]}: {request["MINUTES"]} мин., '
                     f'до {now + timedelta(minutes=request["MINUTES"]):%H:%M}.')
    try:
        run_phases(now + timedelta(minutes=request['MINUTES']))
        set_request(request['id'], STATUS='done', FINISHED_AT=datetime.now())
    except Exception as error:
        logger.exception(error)
        set_request(request['id'], STATUS='failed', FINISHED_AT=datetime.now(), NOTE=str(error)[:500])
        tlg.send_message(f'Геномный поиск #{request["id"]}: ошибка — {error}')


def window_end(now: datetime, window: tuple) -> datetime:
    end = now.replace(hour=window[1] // 60 % 24, minute=window[1] % 60, second=0, microsecond=0)
    return end if end > now else end + timedelta(days=1)


def run_daemon(days: str = EVOLVE_DAYS, spec: str = EVOLVE_WINDOWS,
               daily_days: str = EVOLVE_DAILY_DAYS, daily_spec: str = EVOLVE_DAILY_WINDOW):
    """ Демон: запросы из Telegram (/поиск), ежедневная часовая итерация (пн–сб 09–10) и
    воскресные окна непрерывного поиска. Если поиск уже идёт в другом процессе — ждёт."""
    weekdays = {int(day) for day in days.split(',') if day.strip()}
    windows = parse_windows(spec)
    daily_weekdays = {int(day) for day in daily_days.split(',') if day.strip()}
    daily_windows = parse_windows(daily_spec)
    last_daily = None
    evolver = None
    active = False
    logger.info(f'Абстрактный алгоритм запущен: дни {sorted(weekdays)}, окна {spec}; '
                f'ежедневно {sorted(daily_weekdays)} {daily_spec}; запросы /поиск')
    while True:
        now = datetime.now(ZoneInfo(TIMEZONE))
        try:
            request = pending_request()
            if request and not search_busy():
                run_request(request)
                continue
            daily = current_window(now, daily_windows) if now.weekday() in daily_weekdays else None
            if daily and last_daily != now.date() and not search_busy():
                last_daily = now.date()
                tlg.send_message(f'🧬 Геномный поиск (общий + режимы) до {window_end(now, daily):%H:%M}.')
                run_groups(window_end(now, daily))
                continue
        except Exception as error:
            logger.exception(error)
            sleep(30)
            continue
        window = current_window(now, windows) if now.weekday() in weekdays else None
        if window is not None and search_busy():
            SQLmain.save_activity(SERVICE, 'поиск идёт в другом процессе')
            sleep(60)
            continue
        if window is None:
            if active:  # окно закончилось — сводка сессии
                active = False
                release_lock()
                try:
                    tlg.send_message(evolver.session_report())
                except Exception as error:
                    logger.exception(error)
            SQLmain.save_activity(SERVICE, 'ожидание окна поиска или запроса /поиск')
            sleep(60)  # раз в минуту — проверка запросов из Telegram
            continue
        try:
            if evolver is None:
                evolvers = {regime: Evolver(regime=regime) for regime in search_groups()}
                cycle = 0
            # Каждую минуту — следующая группа: общий, затем режимы по кругу.
            evolver = list(evolvers.values())[cycle % len(evolvers)]
            cycle += 1
            if not active:
                for item in evolvers.values():
                    item.session = new_session()
                active = True
            evolver.ensure_market()
            touch_lock()
            until = monotonic() + 60
            while monotonic() < until:
                evolver.step()
            SQLmain.save_activity(SERVICE, f'шагов {evolver.session["steps"]}, '
                                           f'этапов {len(evolver.session["stages"])}')
        except Exception as error:
            logger.exception(error)
            sleep(30)
