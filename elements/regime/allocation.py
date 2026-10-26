""" Режимный алгоритм распределения капитала: у каждого режима рынка — своя политика.

Политика режима (ген): окна импульса, вид оценки (рост / рост на единицу риска), сколько активов
держать (k), разрешённые активы, условие «лучше денег», веса (поровну / обратно волатильности)
и доля риска (остаток — фонд денег). Общие гены генома — частота пересмотра и пересмотр сразу
при смене режима. Универсального правила нет: в каждый день действует политика режима этого дня
(режим известен по закрытию дня).

Исполнение — как у исполнителя: решение по закрытию дня t, сделки в первой половине дня t+1, поэтому
доля действует с доходности дня t+2 (задержка EXEC_LAG). Издержки — ROTATION_COST на сторону
с оборота, комиссии фондов ROTATION_FUND_FEE в год (деньги — уже за вычетом комиссии фонда).

Генетический поиск (`search`): популяция с элитой, турнирным отбором, кроссовером по режимам
(политика режима переходит целиком), мутацией одного свойства политики, долей новых геномов
каждое поколение (иммигранты) и тёплым стартом от лучших геномов прошлого переобучения.
Приспособленность — по годам обучающего окна (`fitness`): средняя годовая доходность с весами
по давности минус штраф за недобор до цели TARGET в каждом году и за глубокую просадку.
"""
import random
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from elements.regime.assets import ASSETS
from elements.regime.detector import REGIME_KEYS
from source.settings.settings import ROTATION_COST, ROTATION_FUND_FEE

LOOKBACKS = (10, 21, 42, 63, 126, 252)
REBALANCE = (5, 10, 21)
EXPOSURE = (0.0, 0.5, 0.75, 1.0)
RISKY = [a for a in ASSETS if a != 'CASH']
CASH = ASSETS.index('CASH')
EXEC_LAG = 2
TARGET = 0.20
VOL_DAYS = 63


@dataclass(frozen=True)
class Policy:
    lbs: tuple = (3, 4)          # номера окон в LOOKBACKS
    score: str = 'ret'           # 'ret' | 'sharpe'
    k: int = 1
    assets: tuple = tuple(RISKY)
    beat_cash: bool = True
    weighting: str = 'equal'     # 'equal' | 'invvol'
    exposure: float = 1.0

    def text(self) -> str:
        if self.exposure == 0 or not self.assets:
            return 'только деньги'
        lbs = '/'.join(str(LOOKBACKS[i]) for i in self.lbs)
        return (f'топ-{self.k} из {",".join(self.assets)} по {"росту" if self.score == "ret" else "росту/риску"} '
                f'{lbs} дн.{", лучше денег" if self.beat_cash else ""}, '
                f'{"поровну" if self.weighting == "equal" else "по риску"}, риск {self.exposure:.0%}')


@dataclass(frozen=True)
class AllocGenome:
    policies: tuple = field(default_factory=lambda: tuple(Policy() for _ in REGIME_KEYS))
    rebalance: int = 21
    on_change: bool = True

    def key(self) -> tuple:
        return (self.policies, self.rebalance, self.on_change)

    def to_dict(self) -> dict:
        return {'rebalance': self.rebalance, 'on_change': self.on_change,
                'policies': {r: {**p.__dict__, 'lbs': list(p.lbs), 'assets': list(p.assets)}
                             for r, p in zip(REGIME_KEYS, self.policies)}}

    @staticmethod
    def from_dict(data: dict) -> 'AllocGenome':
        return AllocGenome(
            tuple(Policy(**{**data['policies'][r], 'lbs': tuple(data['policies'][r]['lbs']),
                            'assets': tuple(data['policies'][r]['assets'])}) for r in REGIME_KEYS),
            int(data['rebalance']), bool(data['on_change']))


class Universe:
    """ Предрасчёт по ценам активов и режимам дней (всё причинно, по закрытию дня)."""

    def __init__(self, prices: pd.DataFrame, regimes: pd.Series):
        prices = prices[ASSETS]
        self.dates = prices.index
        self.prices = prices.to_numpy(dtype=float)
        self.avail = np.isfinite(self.prices)
        returns = prices.pct_change()
        self.returns = np.nan_to_num(returns.to_numpy(dtype=float), nan=0.0)
        self.mom = np.stack([(prices / prices.shift(lb) - 1).to_numpy(dtype=float) for lb in LOOKBACKS])
        self.vol = (returns.rolling(VOL_DAYS, min_periods=40).std() * np.sqrt(252)).to_numpy(dtype=float)
        aligned = regimes.reindex(self.dates).ffill().fillna('neutral')
        self.regime = np.array([REGIME_KEYS.index(r) for r in aligned], dtype=int)
        self.years = self.dates.year.to_numpy()
        self.fees = np.full(len(ASSETS), ROTATION_FUND_FEE / 252)
        self.fees[CASH] = 0.0
        self._cache: dict = {}

    def index_of(self, moment) -> int:
        return int(self.dates.searchsorted(pd.Timestamp(moment)))

    # --- доли ------------------------------------------------------------------------
    def targets(self, genome: AllocGenome, first: int, last: int) -> tuple[np.ndarray, np.ndarray]:
        """ Целевые доли (дни решений × активы) в [first, last] и номера дней решений."""
        days = np.arange(first, last + 1)
        decide = (days - first) % genome.rebalance == 0
        if genome.on_change:
            decide |= self.regime[days] != self.regime[np.maximum(days - 1, 0)]
        days = days[decide]
        weights = np.zeros((len(days), len(ASSETS)))
        codes = self.regime[days]
        for code, policy in enumerate(genome.policies):
            rows = np.nonzero(codes == code)[0]
            if not len(rows):
                continue
            weights[rows] = self.policy_weights(policy, days[rows])
        return weights, days

    def policy_weights(self, policy: Policy, days: np.ndarray) -> np.ndarray:
        weights = np.zeros((len(days), len(ASSETS)))
        weights[:, CASH] = 1.0
        allowed = [ASSETS.index(a) for a in policy.assets]
        if policy.exposure == 0 or not allowed or policy.k < 1:
            return weights
        ret = np.mean([self.mom[i][days] for i in policy.lbs], axis=0)
        if policy.score == 'sharpe':
            vol = self.vol[days]
            with np.errstate(divide='ignore', invalid='ignore'):  # у денег волатильность ≈ 0
                score = np.mean([self.mom[i][days] / (vol * np.sqrt(LOOKBACKS[i] / 252))
                                 for i in policy.lbs], axis=0)
        else:
            score = ret
        mask = np.zeros(len(ASSETS), dtype=bool)
        mask[allowed] = True
        ok = mask[None, :] & np.isfinite(score) & self.avail[days]
        if policy.beat_cash:
            ok &= ret > ret[:, [CASH]]
        score = np.where(ok, score, -np.inf)
        k = min(policy.k, len(allowed))
        top = np.argsort(-score, axis=1)[:, :k]
        chosen = np.take_along_axis(ok, top, axis=1)
        if policy.weighting == 'invvol':
            vol = np.take_along_axis(np.nan_to_num(self.vol[days], nan=1.0), top, axis=1)
            inv = np.where(chosen, 1 / np.maximum(vol, 0.02), 0.0)
            total = inv.sum(axis=1, keepdims=True)
            part = np.where(total > 0, inv / np.where(total > 0, total, 1) * chosen.sum(axis=1, keepdims=True) / k, 0)
        else:
            part = chosen / k
        part = part * policy.exposure
        rows = np.arange(len(days))[:, None]
        np.add.at(weights, (np.broadcast_to(rows, top.shape), top), part)
        weights[:, CASH] = 1.0 - (weights.sum(axis=1) - weights[:, CASH])
        return weights

    # --- доходность ------------------------------------------------------------------
    def daily(self, genome: AllocGenome, first: int, last: int, start_cash: bool = False) -> np.ndarray:
        """ Быстрая дневная доходность (доли постоянны между пересмотрами) в [first, last]."""
        targets, days = self.targets(genome, first, last)
        held = np.zeros((last - first + 1, len(ASSETS)))
        rows = np.full(last - first + 1, -1)
        rows[days - first] = np.arange(len(days))
        rows = np.maximum.accumulate(rows)
        held[rows >= 0] = targets[rows[rows >= 0]]
        held[rows < 0, CASH] = 1.0
        held = np.vstack([np.tile(held[[0]] if not start_cash else np.eye(len(ASSETS))[[CASH]], (EXEC_LAG, 1)),
                          held])[:last - first + 1]
        turnover = np.abs(np.diff(np.vstack([held[[0]], held]), axis=0)).sum(axis=1)
        part = slice(first, last + 1)
        return (held * self.returns[part]).sum(axis=1) - turnover * ROTATION_COST - held @ self.fees

    def account(self, genome_or_targets, first: int, last: int) -> dict:
        """ Реалистичный счёт с дрейфом долей: старт в деньгах, сделки по целевым долям через
        EXEC_LAG дней после решения, издержки с фактического оборота, комиссии фондов."""
        if isinstance(genome_or_targets, AllocGenome):
            targets, days = self.targets(genome_or_targets, first, last)
        else:
            targets, days = genome_or_targets
        execute = {int(d) + EXEC_LAG - 1: targets[i] for i, d in enumerate(days)}
        values = np.zeros(len(ASSETS))
        values[CASH] = 1.0
        equity = np.empty(last - first + 1)
        weights_log = np.empty((last - first + 1, len(ASSETS)))
        traded = 0.0
        for n, day in enumerate(range(first, last + 1)):
            values *= 1 + self.returns[day]
            values -= values * self.fees
            total = values.sum()
            target = execute.get(day)
            if target is not None:
                wanted = target * total
                trade = np.abs(wanted - values)
                trade[CASH] = 0  # движение денег — это сами сделки в других активах
                cost = trade.sum() * ROTATION_COST
                traded += trade.sum()
                values = wanted * (total - cost) / total
            equity[n] = values.sum()
            weights_log[n] = values / values.sum()
        daily = np.diff(np.r_[1.0, equity]) / np.r_[1.0, equity[:-1]]
        return {'daily': daily, 'equity': equity, 'weights': weights_log, 'turnover': traded}


# --- приспособленность ---------------------------------------------------------------
def yearly(daily: np.ndarray, years: np.ndarray, min_days: int = 126) -> tuple[np.ndarray, np.ndarray]:
    """ Годовые доходности по календарным годам (неполный год от min_days — в пересчёте на год)."""
    result, labels = [], []
    for year in np.unique(years):
        part = daily[years == year]
        if len(part) < min_days:
            continue
        total = np.prod(1 + part)
        result.append(total ** (252 / len(part)) - 1 if len(part) < 240 else total - 1)
        labels.append(year)
    return np.array(result), np.array(labels)


@dataclass(frozen=True)
class Method:
    """ Способ генетического поиска (сравниваются на периоде отбора методов)."""
    name: str
    shortfall: float = 2.0       # штраф за недобор до TARGET в году
    half_life: float = 0.0       # вес года 0.5 ** (давность / half_life); 0 — все годы равны
    drawdown: float = 0.5        # штраф за просадку глубже −DD_FREE
    window: int = 0              # окно обучения в годах; 0 — вся история
    ensemble: int = 1            # усреднить доли N лучших несхожих геномов
    population: int = 60
    generations: int = 25
    immigrants: float = 0.15
    parsimony: float = 0.0       # штраф за число разрешённых активов и окон
    family: str = 'free'         # 'free' — любые политики; 'robust' — только диверсифицированные
    runs: int = 1                # независимых популяций (острова); доли их лучших геномов усредняются


DD_FREE = 0.10


def fitness(daily: np.ndarray, years: np.ndarray, method: Method, genome: AllocGenome | None = None) -> float:
    values, labels = yearly(daily, years)
    if not len(values):
        return -9.0
    if method.half_life > 0:
        age = labels.max() - labels
        weights = 0.5 ** (age / method.half_life)
    else:
        weights = np.ones(len(values))
    weights = weights / weights.sum()
    equity = np.cumprod(1 + daily)
    drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1))
    score = float(weights @ values) - method.shortfall * float(weights @ np.maximum(0, TARGET - values)) \
        - method.drawdown * max(0.0, -drawdown - DD_FREE)
    if method.parsimony and genome is not None:
        size = sum(len(p.assets) + len(p.lbs) for p in genome.policies if p.exposure > 0)
        score -= method.parsimony * size
    return score


# --- генетика ------------------------------------------------------------------------
ROBUST_ASSETS = ('EQ', 'BOND', 'LONG', 'CORP', 'GOLD')
FX_ASSETS = ROBUST_ASSETS + ('FX',)
ALL_LOOKBACKS = tuple(range(len(LOOKBACKS)))


def robust_policy(rng: random.Random, family: str = 'robust') -> Policy:
    """ Устойчивое семейство: двойной импульс по всем окнам, веса по риску, k ≥ 3 из широкого
    набора; режим выбирает только долю риска, k и вид оценки."""
    return Policy(lbs=ALL_LOOKBACKS, score=rng.choice(['ret', 'sharpe']), k=rng.randint(3, len(ROBUST_ASSETS)),
                  assets=rng.choice([ROBUST_ASSETS, FX_ASSETS]) if family == 'robust_fx' else ROBUST_ASSETS,
                  beat_cash=True, weighting='invvol',
                  exposure=rng.choice(EXPOSURE))


def random_policy(rng: random.Random, family: str = 'free') -> Policy:
    if family.startswith('robust'):
        return robust_policy(rng, family)
    n_assets = rng.randint(1, len(RISKY))
    return Policy(
        lbs=tuple(sorted(rng.sample(range(len(LOOKBACKS)), rng.randint(1, 3)))),
        score=rng.choice(['ret', 'sharpe']), k=rng.randint(1, 3),
        assets=tuple(a for a in RISKY if a in rng.sample(RISKY, n_assets)),
        beat_cash=rng.random() < 0.75, weighting=rng.choice(['equal', 'invvol']),
        exposure=rng.choice(EXPOSURE),
    )


def random_genome(rng: random.Random, family: str = 'free') -> AllocGenome:
    return AllocGenome(tuple(random_policy(rng, family) for _ in REGIME_KEYS), rng.choice(REBALANCE),
                       rng.random() < 0.7)


def mutate_policy(rng: random.Random, policy: Policy, family: str = 'free') -> Policy:
    if family.startswith('robust'):
        what = rng.choice(['score', 'k', 'exposure'] + (['universe'] if family == 'robust_fx' else []))
        if what == 'k':
            return replace(policy, k=min(len(policy.assets), max(3, policy.k + rng.choice([-1, 1]))))
        if what == 'universe':
            return replace(policy, assets=FX_ASSETS if policy.assets == ROBUST_ASSETS else ROBUST_ASSETS)
        if what == 'score':
            return replace(policy, score='sharpe' if policy.score == 'ret' else 'ret')
        return replace(policy, exposure=rng.choice([e for e in EXPOSURE if e != policy.exposure]))
    what = rng.choice(['lbs', 'score', 'k', 'assets', 'beat_cash', 'weighting', 'exposure', 'new'])
    if what == 'lbs':
        lbs = set(policy.lbs)
        lbs.symmetric_difference_update({rng.randrange(len(LOOKBACKS))})
        return replace(policy, lbs=tuple(sorted(lbs)) or policy.lbs)
    if what == 'score':
        return replace(policy, score='sharpe' if policy.score == 'ret' else 'ret')
    if what == 'k':
        return replace(policy, k=min(3, max(1, policy.k + rng.choice([-1, 1]))))
    if what == 'assets':
        assets = set(policy.assets)
        assets.symmetric_difference_update({rng.choice(RISKY)})
        return replace(policy, assets=tuple(a for a in RISKY if a in assets))
    if what == 'beat_cash':
        return replace(policy, beat_cash=not policy.beat_cash)
    if what == 'weighting':
        return replace(policy, weighting='invvol' if policy.weighting == 'equal' else 'equal')
    if what == 'exposure':
        return replace(policy, exposure=rng.choice([e for e in EXPOSURE if e != policy.exposure]))
    return random_policy(rng)


def mutate(rng: random.Random, genome: AllocGenome, family: str = 'free') -> AllocGenome:
    if rng.random() < 0.15:
        return replace(genome, rebalance=rng.choice(REBALANCE)) if rng.random() < 0.7 else \
            replace(genome, on_change=not genome.on_change)
    policies = list(genome.policies)
    for _ in range(1 if rng.random() < 0.7 else 2):
        i = rng.randrange(len(policies))
        policies[i] = mutate_policy(rng, policies[i], family)
    return replace(genome, policies=tuple(policies))


def crossover(rng: random.Random, a: AllocGenome, b: AllocGenome) -> AllocGenome:
    policies = tuple(pa if rng.random() < 0.5 else pb for pa, pb in zip(a.policies, b.policies))
    return AllocGenome(policies, rng.choice([a.rebalance, b.rebalance]), rng.choice([a.on_change, b.on_change]))


def search(universe: Universe, first: int, last: int, method: Method, rng: random.Random,
           seeds: list[AllocGenome] = ()) -> list[tuple[AllocGenome, float]]:
    """ GA на обучающем окне [first, last]; возвращает популяцию по убыванию приспособленности."""
    years = universe.years[first:last + 1]
    cache: dict = {}

    def score(genome: AllocGenome) -> float:
        key = genome.key()
        if key not in cache:
            cache[key] = fitness(universe.daily(genome, first, last), years, method, genome)
        return cache[key]

    base = AllocGenome() if method.family == 'free' else \
        AllocGenome(tuple(Policy(lbs=ALL_LOOKBACKS, k=len(ROBUST_ASSETS), assets=ROBUST_ASSETS, weighting='invvol')
                          for _ in REGIME_KEYS), 5, True)
    population = list(dict.fromkeys(list(seeds) + [base]))
    while len(population) < method.population:
        population.append(random_genome(rng, method.family))
    population.sort(key=lambda g: -score(g))
    elite = max(2, method.population // 15)
    for _ in range(method.generations):
        children = population[:elite]
        n_new = int(method.immigrants * method.population)
        children += [random_genome(rng, method.family) for _ in range(n_new)]
        while len(children) < method.population:
            p1 = max(rng.sample(population, 3), key=score)
            if rng.random() < 0.6:
                p2 = max(rng.sample(population, 3), key=score)
                child = crossover(rng, p1, p2)
                if rng.random() < 0.5:
                    child = mutate(rng, child, method.family)
            else:
                child = mutate(rng, p1, method.family)
            children.append(child)
        population = sorted(dict.fromkeys(children), key=lambda g: -score(g))[:method.population]
    return [(g, score(g)) for g in population]


def pick_ensemble(universe: Universe, ranked: list[tuple[AllocGenome, float]], first: int, last: int,
                  size: int, similar: float = 0.95) -> list[AllocGenome]:
    """ Лучшие геномы с корреляцией дневной доходности ниже similar между собой."""
    chosen, series = [], []
    for genome, _ in ranked:
        daily = universe.daily(genome, first, last)
        if any(np.corrcoef(daily, other)[0, 1] > similar for other in series):
            continue
        chosen.append(genome)
        series.append(daily)
        if len(chosen) >= size:
            break
    return chosen


def ensemble_targets(universe: Universe, genomes: list[AllocGenome], first: int, last: int):
    """ Средние доли ансамбля по дням решений любого из геномов (между — последние доли каждого)."""
    parts = [universe.targets(g, first, last) for g in genomes]
    days = np.unique(np.concatenate([d for _, d in parts]))
    total = np.zeros((len(days), len(ASSETS)))
    for weights, own in parts:
        index = np.searchsorted(own, days, side='right') - 1
        current = np.where(index[:, None] >= 0, weights[np.maximum(index, 0)], np.eye(len(ASSETS))[CASH])
        total += current
    return total / len(parts), days


# --- слепой проход ---------------------------------------------------------------------
def refit(universe: Universe, day: int, stop: int, method: Method, rngs: list[random.Random],
          seeds: list[list[AllocGenome]], first_train: int):
    """ Переобучение в день day только по прошлому (до day − 1 − EXEC_LAG): по каждой популяции
    GA, затем доли на [day, stop] — среднее лучших геномов всех популяций (и ансамблей внутри).
    seeds обновляются на месте (тёплый старт следующего переобучения)."""
    train_first = first_train if not method.window else max(first_train, day - 252 * method.window)
    train_last = day - 1 - EXEC_LAG
    members, best = [], -9.0
    for run, rng in enumerate(rngs):
        ranked = search(universe, train_first, train_last, method, rng, seeds[run])
        seeds[run] = [g for g, _ in ranked[:10]]
        best = max(best, ranked[0][1])
        members += pick_ensemble(universe, ranked, train_first, train_last, method.ensemble)             if method.ensemble > 1 else [ranked[0][0]]
    if len(members) > 1:
        weights, days = ensemble_targets(universe, members, day, stop)
    else:
        weights, days = universe.targets(members[0], day, stop)
    return weights, days, members, best


PRODUCTION = Method('regime-robust-fx', shortfall=1.5, family='robust_fx', window=5, runs=3)


def walk_forward(universe: Universe, start, end, method: Method, refit_days: int = 63,
                 train_start='2008-06-01', seed: int = 0, progress=None) -> dict:
    """ Слепой проход: на каждой точке переобучения GA только по прошлым дням, затем политика
    торгует до следующей точки. Возвращает целевые доли по дням решений за весь проход и журнал."""
    rngs = [random.Random(seed * 1000 + run) for run in range(method.runs)]
    first_train = universe.index_of(train_start)
    begin, finish = universe.index_of(start), min(universe.index_of(end), len(universe.dates) - 1)
    refits = list(range(begin, finish + 1, refit_days))
    seeds: list[list[AllocGenome]] = [[] for _ in rngs]
    all_weights, all_days, log = [], [], []
    for number, day in enumerate(refits):
        stop = refits[number + 1] - 1 if number + 1 < len(refits) else finish
        weights, days, members, best = refit(universe, day, stop, method, rngs, seeds, first_train)
        all_weights.append(weights)
        all_days.append(days)
        log.append({'date': str(universe.dates[day].date()), 'fitness': round(best, 4),
                    'genomes': [g.to_dict() for g in members]})
        if progress:
            progress(number + 1, len(refits))
    return {'targets': (np.vstack(all_weights), np.concatenate(all_days)), 'log': log,
            'first': begin, 'last': finish}


# --- работа в live --------------------------------------------------------------------
LIVE_KIND = 'regime_alloc'
MIN_CHANGE = 0.10  # новая цель пишется, только если доли сдвинулись суммарно больше чем на 10%


def save_live(log: list[dict], data_end=None):
    """ Геномы последнего переобучения слепого прохода — действующая политика режимов."""
    from elements.selftune.params import save_active
    if log:
        save_active(LIVE_KIND, {'genomes': log[-1]['genomes'], 'refit': log[-1]['date']},
                    data_end=data_end, note=PRODUCTION.name)


def live_target(universe: Universe) -> tuple[dict, str] | None:
    """ Доли активов на сегодня по действующим геномам (среднее), с учётом режима последнего дня."""
    from elements.selftune.params import load_active_record
    import json as _json
    record = load_active_record(LIVE_KIND)
    if not record:
        return None
    genomes = [AllocGenome.from_dict(g) for g in _json.loads(record['PARAMS'])['genomes']]
    last = len(universe.dates) - 1
    weights = np.mean([universe.policy_weights(g.policies[universe.regime[last]], np.array([last]))[0]
                       for g in genomes], axis=0)
    regime = REGIME_KEYS[universe.regime[last]]
    reason = f'режим {regime}: ' + '; '.join(g.policies[universe.regime[last]].text() for g in genomes[:1])
    return {a: float(w) for a, w in zip(ASSETS, weights) if w > 0.005}, reason


def update_live_target(today=None) -> dict | None:
    """ Записать цель распределения в allocation_targets (читает исполнитель), если она заметно
    изменилась относительно последней записанной или прошло rebalance дней."""
    from datetime import datetime as _dt
    from elements.regime.assets import FUNDS, load_prices
    from elements.regime.worker import MarketRegime
    from source.sql.main import SQLmain
    import source.sql.tables as tables
    rows = SQLmain.run_sql_command('SELECT "DATE", REGIME FROM market_regime ORDER BY "DATE"')
    if not rows:
        return None
    regimes = pd.Series([r['REGIME'] for r in rows], index=pd.to_datetime([r['DATE'] for r in rows]))
    universe = Universe(load_prices(), regimes)
    result = live_target(universe)
    if result is None:
        return None
    target, reason = result
    current = {r['SECID']: r['WEIGHT'] for r in SQLmain.run_sql_command(
        'SELECT SECID, WEIGHT FROM allocation_targets WHERE "DATE" = (SELECT MAX("DATE") FROM allocation_targets)')}
    wanted = {FUNDS[a]: w for a, w in target.items()}
    change = sum(abs(wanted.get(k, 0) - current.get(k, 0)) for k in set(wanted) | set(current))
    if change < MIN_CHANGE:
        return None
    decision = universe.dates[-1].date()
    rows = [{'DATE': decision, 'ASSET': asset, 'SECID': FUNDS[asset], 'WEIGHT': float(target.get(asset, 0.0)),
             'REASON': ('режимное распределение; ' + reason)[:500], 'CREATED_AT': _dt.now()}
            for asset in ASSETS if asset in target or FUNDS[asset] in current]
    SQLmain.upsert(tables.AllocationTarget, rows, ['DATE', 'SECID'])
    return {'date': decision, 'target': wanted, 'reason': reason}
