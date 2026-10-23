""" Гены абстрактного алгоритма: строительные блоки со свободными параметрами.

Ген — это класс с объявленным пространством параметров (`space`) и методом `evaluate(market)`.
Чтобы добавить новый блок, достаточно описать класс и пометить его `@register` — генератор
случайных геномов и все операторы сразу начнут его использовать:

    @register
    class MyFilter(Gene):
        name, slot = 'my_filter', 'filter'
        space = {'feature': Feature(), 'q': Real(0.05, 0.95, 0.01)}
        def evaluate(self, market):
            return market.rank(self.p['feature']) > self.p['q']

Слоты (что делает ген в геноме, см. genome.SLOTS):
    filter — булева маска строк «бумага-день»: можно ли покупать;
    logic  — сколько фильтров должно выполниться (доля);
    score  — числовая оценка для выбора лучших бумаг дня (оценки генов складываются);
    exit   — правило выхода (удержание, стоп/тейк в ATR, выход на открытии);
    size   — размер позиции и число бумаг в день.
"""
import numpy as np


REGISTRY: dict[str, type['Gene']] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def genes_of(slot: str) -> list[type['Gene']]:
    return [cls for cls in REGISTRY.values() if cls.slot == slot]


# --- типы свободных параметров ----------------------------------------------------
class Param:
    def sample(self, rng, market):
        raise NotImplementedError

    def nudge(self, rng, value, market):
        """ Небольшое изменение значения (по умолчанию — новое случайное)."""
        return self.sample(rng, market)


class Choice(Param):
    def __init__(self, options):
        self.options = options

    def values(self, market) -> list:
        return list(self.options(market) if callable(self.options) else self.options)

    def sample(self, rng, market):
        return rng.choice(self.values(market))

    def nudge(self, rng, value, market):
        others = [option for option in self.values(market) if option != value]
        return rng.choice(others) if others else value


class Feature(Choice):
    """ Любой признак бумаги, который есть на рынке (список растёт вместе с панелью)."""
    def __init__(self, group: str = 'stock'):
        super().__init__(lambda market: market.features[group])


class Real(Param):
    def __init__(self, low: float, high: float, step: float = 0.0):
        self.low, self.high, self.step = low, high, step

    def snap(self, value: float) -> float:
        value = min(max(value, self.low), self.high)
        if self.step:
            value = self.low + round((value - self.low) / self.step) * self.step
        return round(float(value), 6)

    def sample(self, rng, market):
        return self.snap(rng.uniform(self.low, self.high))

    def nudge(self, rng, value, market):
        return self.snap(value + rng.gauss(0, (self.high - self.low) * rng.choice([0.03, 0.1, 0.3])))


class Int(Real):
    def snap(self, value: float) -> int:
        return int(min(max(round(value), self.low), self.high))

    def nudge(self, rng, value, market):
        return self.snap(value + rng.choice([-2, -1, 1, 2]))


class Subset(Param):
    def __init__(self, options: list, minimum: int = 1):
        self.options, self.minimum = list(options), minimum

    def sample(self, rng, market):
        size = rng.randint(self.minimum, len(self.options))
        return sorted(rng.sample(self.options, size))

    def nudge(self, rng, value, market):
        value = set(value)
        value ^= {rng.choice(self.options)}
        return sorted(value) if len(value) >= self.minimum else sorted(set(value) | {rng.choice(self.options)})


# --- базовый ген ----------------------------------------------------------------
class Gene:
    name = ''
    slot = ''
    weight = 1.0  # относительная частота при случайном выборе гена слота
    space: dict[str, Param] = {}

    def __init__(self, **params):
        self.p = params

    @classmethod
    def random(cls, rng, market) -> 'Gene':
        return cls(**{key: param.sample(rng, market) for key, param in cls.space.items()})

    def mutate(self, rng, market) -> 'Gene':
        """ Сдвиг одного свободного параметра."""
        key = rng.choice(list(self.space))
        params = dict(self.p)
        params[key] = self.space[key].nudge(rng, params.get(key), market)
        return type(self)(**params)

    def evaluate(self, market):
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {'gene': self.name, **self.p}

    @staticmethod
    def from_dict(data: dict) -> 'Gene':
        data = dict(data)
        return REGISTRY[data.pop('gene')](**data)

    def describe(self) -> str:
        return f'{self.name}(' + ', '.join(f'{k}={v}' for k, v in self.p.items()) + ')'


def compare(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        return values > threshold if op == '>' else values < threshold


OPS = Choice(['<', '>'])


# --- фильтры ------------------------------------------------------------------------
@register
class Level(Gene):
    """ Признак выше/ниже своего исторического квантиля (квантиль — только по периоду поиска)."""
    name, slot, weight = 'level', 'filter', 1.5
    space = {'feature': Feature(), 'op': OPS, 'q': Real(0.01, 0.99, 0.01)}

    def evaluate(self, market):
        return compare(market.values[self.p['feature']], self.p['op'],
                       market.quantile(self.p['feature'], self.p['q']))


@register
class CrossRank(Gene):
    """ Место бумаги среди всех бумаг этого дня по признаку (0 — худшая, 1 — лучшая)."""
    name, slot, weight = 'xrank', 'filter', 1.5
    space = {'feature': Feature(), 'op': OPS, 'q': Real(0.01, 0.99, 0.01)}

    def evaluate(self, market):
        return compare(market.rank(self.p['feature']), self.p['op'], self.p['q'])


@register
class Spread(Gene):
    """ Разница мест двух признаков в кросс-секции: «A заметно сильнее B»."""
    name, slot = 'spread', 'filter'
    space = {'a': Feature(), 'b': Feature(), 'op': OPS, 'd': Real(-0.9, 0.9, 0.02)}

    def evaluate(self, market):
        return compare(market.rank(self.p['a']) - market.rank(self.p['b']), self.p['op'], self.p['d'])


@register
class Regime(Gene):
    """ Состояние рынка в целом (индекс, ширина рынка) выше/ниже исторического квантиля."""
    name, slot = 'regime', 'filter'
    space = {'feature': Feature('market'), 'op': OPS, 'q': Real(0.02, 0.98, 0.01)}

    def evaluate(self, market):
        return compare(market.values[self.p['feature']], self.p['op'],
                       market.quantile(self.p['feature'], self.p['q']))


@register
class Weekday(Gene):
    """ Дни недели, в которые разрешён сигнал."""
    name, slot, weight = 'weekday', 'filter', 0.4
    space = {'days': Subset([0, 1, 2, 3, 4], minimum=2)}

    def evaluate(self, market):
        return np.isin(market.weekday, self.p['days'])


# --- свод правил «динамика бумаги» (22.10.2026) ----------------------------------------
LAGS = Choice([1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 60, 90, 120])
WINDOWS = Choice([5, 10, 20, 40, 60, 120, 250])
# Новые гены (свод «динамика» 22.10 и «вариации» 23.10): этап PHASES «новые своды» берёт их чаще.
NEW_GENES = ('delta', 'xdelta', 'zlevel', 'streak', 'gated', 'delta_score', 'z_score',
             'band', 'xband', 'persist', 'pair_score')


@register
class Delta(Gene):
    """ Изменение признака бумаги за lag дней выше/ниже своего исторического квантиля."""
    name, slot = 'delta', 'filter'
    space = {'feature': Feature(), 'lag': LAGS, 'op': OPS, 'q': Real(0.05, 0.95, 0.01)}

    def evaluate(self, market):
        values = market.change(self.p['feature'], self.p['lag'])
        key = ('change', self.p['feature'], self.p['lag'])
        return compare(values, self.p['op'], market.dquantile(key, values, self.p['q']))


@register
class CrossDelta(Gene):
    """ Место бумаги среди бумаг дня по изменению признака за lag дней (кто ускоряется)."""
    name, slot = 'xdelta', 'filter'
    space = {'feature': Feature(), 'lag': LAGS, 'op': OPS, 'q': Real(0.05, 0.95, 0.01)}

    def evaluate(self, market):
        key = ('change', self.p['feature'], self.p['lag'])
        values = market.drank(key, market.change(self.p['feature'], self.p['lag']))
        return compare(values, self.p['op'], self.p['q'])


@register
class ZLevel(Gene):
    """ Признак далеко от своей нормы для этой бумаги (z-оценка за window дней)."""
    name, slot = 'zlevel', 'filter'
    space = {'feature': Feature(), 'window': WINDOWS, 'op': OPS, 'z': Real(-2.5, 2.5, 0.1)}

    def evaluate(self, market):
        return compare(market.zscore(self.p['feature'], self.p['window']), self.p['op'], self.p['z'])


@register
class Streak(Gene):
    """ Серия дней роста (n > 0) или падения (n < 0) подряд: длиннее/короче n."""
    name, slot, weight = 'streak', 'filter', 0.6
    space = {'op': OPS, 'n': Int(-7, 7)}

    def evaluate(self, market):
        return compare(market.streak(), self.p['op'], self.p['n'])


@register
class Gated(Gene):
    """ Условие на бумагу действует только в своём режиме рынка; вне режима — пропускать
    всех (other = 1) или никого (other = 0)."""
    name, slot = 'gated', 'filter'
    space = {
        'market': Feature('market'), 'mop': OPS, 'mq': Real(0.1, 0.9, 0.05),
        'feature': Feature(), 'op': OPS, 'q': Real(0.05, 0.95, 0.01), 'other': Choice([0, 1]),
    }

    def evaluate(self, market):
        regime = compare(market.values[self.p['market']], self.p['mop'],
                         market.quantile(self.p['market'], self.p['mq']))
        stock = compare(market.rank(self.p['feature']), self.p['op'], self.p['q'])
        return np.where(regime, stock, bool(self.p['other']))


@register
class DeltaScore(Gene):
    """ Вклад места бумаги по изменению признака за lag дней в итоговую оценку."""
    name, slot = 'delta_score', 'score'
    space = {'feature': Feature(), 'lag': LAGS, 'sign': Choice([-1, 1]), 'w': Real(0.1, 1.0, 0.05)}

    def evaluate(self, market):
        key = ('change', self.p['feature'], self.p['lag'])
        values = market.drank(key, market.change(self.p['feature'], self.p['lag']))
        return self.p['sign'] * self.p['w'] * np.nan_to_num(values, nan=0.5)


@register
class ZScoreScore(Gene):
    """ Вклад отклонения признака от своей нормы (z за window дней, место в дне) в оценку."""
    name, slot = 'z_score', 'score'
    space = {'feature': Feature(), 'window': WINDOWS, 'sign': Choice([-1, 1]), 'w': Real(0.1, 1.0, 0.05)}

    def evaluate(self, market):
        key = ('zscore', self.p['feature'], self.p['window'])
        values = market.drank(key, market.zscore(self.p['feature'], self.p['window']))
        return self.p['sign'] * self.p['w'] * np.nan_to_num(values, nan=0.5)


# --- свод «вариации» (23.10.2026): коридоры, устойчивость, пары -------------------------------
@register
class Band(Gene):
    """ Признак внутри коридора своих исторических квантилей [lo, lo + width]."""
    name, slot = 'band', 'filter'
    space = {'feature': Feature(), 'lo': Real(0.0, 0.9, 0.01), 'width': Real(0.05, 0.6, 0.01)}

    def evaluate(self, market):
        values = market.values[self.p['feature']]
        low = market.quantile(self.p['feature'], self.p['lo'])
        high = market.quantile(self.p['feature'], min(1.0, self.p['lo'] + self.p['width']))
        with np.errstate(invalid='ignore'):
            return (values >= low) & (values <= high)


@register
class CrossBand(Gene):
    """ Место бумаги среди бумаг дня внутри коридора [lo, lo + width] — «не крайние»."""
    name, slot = 'xband', 'filter'
    space = {'feature': Feature(), 'lo': Real(0.0, 0.9, 0.01), 'width': Real(0.05, 0.6, 0.01)}

    def evaluate(self, market):
        rank = market.rank(self.p['feature'])
        with np.errstate(invalid='ignore'):
            return (rank >= self.p['lo']) & (rank <= self.p['lo'] + self.p['width'])


@register
class Persist(Gene):
    """ Условие держится: бумага в верхних (1 − q) по признаку не меньше чем в доле share
    из последних n дней."""
    name, slot = 'persist', 'filter'
    space = {'feature': Feature(), 'q': Choice([0.5, 0.6, 0.7, 0.8, 0.9]), 'n': Choice([3, 5, 10, 20, 40, 60]),
             'share': Real(0.4, 1.0, 0.1)}

    def evaluate(self, market):
        hits = market.persist(self.p['feature'], self.p['q'], self.p['n'])
        return hits >= np.ceil(self.p['share'] * self.p['n'] - 1e-9)


@register
class PairScore(Gene):
    """ Вклад сочетания двух признаков: произведение мест бумаги по ним (оба высоко/низко)."""
    name, slot = 'pair_score', 'score'
    space = {'a': Feature(), 'b': Feature(), 'sa': Choice([-1, 1]), 'sb': Choice([-1, 1]),
             'w': Real(0.1, 1.0, 0.05)}

    def evaluate(self, market):
        def oriented(feature, sign):
            rank = np.nan_to_num(market.rank(feature), nan=0.5)
            return rank if sign > 0 else 1 - rank
        return self.p['w'] * oriented(self.p['a'], self.p['sa']) * oriented(self.p['b'], self.p['sb'])


# --- логика, оценка, выход, размер ------------------------------------------------------
@register
class Vote(Gene):
    """ Доля фильтров, которые должны выполниться (1 — все, 0.5 — половина)."""
    name, slot = 'vote', 'logic'
    space = {'need': Real(0.3, 1.0, 0.05)}

    def evaluate(self, market):
        return self.p['need']


@register
class RankScore(Gene):
    """ Вклад места бумаги по признаку в итоговую оценку (знак — «чем больше, тем лучше» или наоборот)."""
    name, slot = 'rank_score', 'score'
    space = {'feature': Feature(), 'sign': Choice([-1, 1]), 'w': Real(0.1, 1.0, 0.05)}

    def evaluate(self, market):
        return self.p['sign'] * self.p['w'] * np.nan_to_num(market.rank(self.p['feature']), nan=0.5)


@register
class Noise(Gene):
    """ Случайный порядок бумаг дня (фиксированный зерном) — «нулевая» оценка для сравнения."""
    name, slot, weight = 'noise', 'score', 0.2
    space = {'seed': Int(0, 999)}

    def evaluate(self, market):
        return np.random.default_rng(self.p['seed']).random(len(market.didx)) * 0.01


@register
class Exit(Gene):
    name, slot = 'exit', 'exit'
    space = {
        'hold': Int(1, 10), 'stop': Choice([0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]),
        'take': Choice([0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 99.0]), 'at_open': Choice([0, 1]),
    }

    def evaluate(self, market):
        return market.outcome(self.p['hold'], self.p['stop'], self.p['take'], self.p['at_open'])


@register
class Size(Gene):
    name, slot = 'size', 'size'
    space = {'slot': Real(0.02, 0.4, 0.01), 'top': Int(1, 20)}

    def evaluate(self, market):
        return self.p['slot'], self.p['top']
