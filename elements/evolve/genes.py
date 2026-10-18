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
    space = {'feature': Feature(), 'op': OPS, 'q': Real(0.02, 0.98, 0.01)}

    def evaluate(self, market):
        return compare(market.values[self.p['feature']], self.p['op'],
                       market.quantile(self.p['feature'], self.p['q']))


@register
class CrossRank(Gene):
    """ Место бумаги среди всех бумаг этого дня по признаку (0 — худшая, 1 — лучшая)."""
    name, slot, weight = 'xrank', 'filter', 1.5
    space = {'feature': Feature(), 'op': OPS, 'q': Real(0.02, 0.98, 0.01)}

    def evaluate(self, market):
        return compare(market.rank(self.p['feature']), self.p['op'], self.p['q'])


@register
class Spread(Gene):
    """ Разница мест двух признаков в кросс-секции: «A заметно сильнее B»."""
    name, slot = 'spread', 'filter'
    space = {'a': Feature(), 'b': Feature(), 'op': OPS, 'd': Real(-0.8, 0.8, 0.02)}

    def evaluate(self, market):
        return compare(market.rank(self.p['a']) - market.rank(self.p['b']), self.p['op'], self.p['d'])


@register
class Regime(Gene):
    """ Состояние рынка в целом (индекс, ширина рынка) выше/ниже исторического квантиля."""
    name, slot = 'regime', 'filter'
    space = {'feature': Feature('market'), 'op': OPS, 'q': Real(0.05, 0.95, 0.01)}

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


# --- логика, оценка, выход, размер ------------------------------------------------------
@register
class Vote(Gene):
    """ Доля фильтров, которые должны выполниться (1 — все, 0.5 — половина)."""
    name, slot = 'vote', 'logic'
    space = {'need': Real(0.4, 1.0, 0.05)}

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
        'hold': Int(1, 5), 'stop': Choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0]),
        'take': Choice([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 99.0]), 'at_open': Choice([0, 1]),
    }

    def evaluate(self, market):
        return market.outcome(self.p['hold'], self.p['stop'], self.p['take'], self.p['at_open'])


@register
class Size(Gene):
    name, slot = 'size', 'size'
    space = {'slot': Real(0.03, 0.3, 0.01), 'top': Int(1, 10)}

    def evaluate(self, market):
        return self.p['slot'], self.p['top']
