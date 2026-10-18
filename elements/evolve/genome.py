""" Геном — набор генов по слотам, и операторы, которые порождают из геномов новые.

Слоты и их границы — в SLOTS; операторы регистрируются `@operator` и выбираются случайно
с весами, которые подстраиваются под их собственную историю успехов (OperatorStats):
закон формирования следующего кандидата заранее не известен, но смещается к тому, что работало.
"""
import hashlib
import json
import math

from elements.evolve.genes import Gene, genes_of


# слот: (минимум генов, максимум генов)
SLOTS = {'filter': (1, 6), 'logic': (1, 1), 'score': (1, 3), 'exit': (1, 1), 'size': (1, 1)}


def random_gene(rng, market, slot: str) -> Gene:
    classes = genes_of(slot)
    cls = rng.choices(classes, weights=[c.weight for c in classes])[0]
    return cls.random(rng, market)


class Genome:

    def __init__(self, genes: dict[str, list[Gene]]):
        self.genes = {slot: list(genes.get(slot, [])) for slot in SLOTS}

    @classmethod
    def random(cls, rng, market) -> 'Genome':
        genes = {}
        for slot, (low, high) in SLOTS.items():
            count = rng.randint(low, min(high, low + 2))
            genes[slot] = [random_gene(rng, market, slot) for _ in range(count)]
        return cls(genes)

    def copy(self) -> 'Genome':
        return Genome({slot: list(genes) for slot, genes in self.genes.items()})

    def repair(self, rng, market) -> 'Genome':
        """ Привести число генов к границам слотов, убрать повторы."""
        for slot, (low, high) in SLOTS.items():
            unique, keys = [], set()
            for gene in self.genes[slot]:
                key = json.dumps(gene.to_dict(), sort_keys=True)
                if key not in keys:
                    unique.append(gene)
                    keys.add(key)
            while len(unique) < low:
                unique.append(random_gene(rng, market, slot))
            self.genes[slot] = unique[:high]
        return self

    def to_dict(self) -> dict:
        return {slot: [gene.to_dict() for gene in genes] for slot, genes in self.genes.items()}

    @classmethod
    def from_dict(cls, data: dict) -> 'Genome':
        return cls({slot: [Gene.from_dict(item) for item in items] for slot, items in data.items()
                    if slot in SLOTS})

    def key(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @property
    def size(self) -> int:
        return sum(len(genes) for genes in self.genes.values())

    def describe(self) -> str:
        return '; '.join(f'{slot}: ' + ', '.join(g.describe() for g in genes)
                         for slot, genes in self.genes.items())


# --- операторы ---------------------------------------------------------------------
OPERATORS: dict[str, tuple] = {}  # имя -> (функция, базовый вес)


def operator(name: str, weight: float = 1.0):
    def wrap(function):
        OPERATORS[name] = (function, weight)
        return function
    return wrap


def any_gene(rng, genome: Genome) -> tuple[str, int]:
    slot = rng.choice([slot for slot, genes in genome.genes.items() if genes])
    return slot, rng.randrange(len(genome.genes[slot]))


@operator('fresh', 0.6)
def fresh(rng, market, parent, donors):
    """ Совершенно новый случайный геном."""
    return Genome.random(rng, market)


@operator('jitter', 2.0)
def jitter(rng, market, parent, donors):
    """ Сдвиг одного свободного параметра одного гена."""
    child = parent.copy()
    slot, index = any_gene(rng, child)
    child.genes[slot][index] = child.genes[slot][index].mutate(rng, market)
    return child


@operator('swap')
def swap(rng, market, parent, donors):
    """ Замена гена новым случайным того же слота."""
    child = parent.copy()
    slot, index = any_gene(rng, child)
    child.genes[slot][index] = random_gene(rng, market, slot)
    return child


@operator('grow')
def grow(rng, market, parent, donors):
    """ Добавление гена (фильтра или оценки)."""
    child = parent.copy()
    slot = rng.choice([s for s, (low, high) in SLOTS.items() if len(child.genes[s]) < high] or ['filter'])
    child.genes[slot].append(random_gene(rng, market, slot))
    return child


@operator('prune')
def prune(rng, market, parent, donors):
    """ Удаление гена (где их больше минимума)."""
    child = parent.copy()
    slots = [s for s, (low, high) in SLOTS.items() if len(child.genes[s]) > low]
    if slots:
        slot = rng.choice(slots)
        child.genes[slot].pop(rng.randrange(len(child.genes[slot])))
    return child


@operator('cross', 1.2)
def cross(rng, market, parent, donors):
    """ Скрещивание: каждый слот — от родителя или от донора (или смесь генов)."""
    if not donors:
        return jitter(rng, market, parent, donors)
    donor = rng.choice(donors)
    genes = {}
    for slot in SLOTS:
        pick = rng.random()
        if pick < 0.4:
            genes[slot] = list(parent.genes[slot])
        elif pick < 0.8:
            genes[slot] = list(donor.genes[slot])
        else:
            mixed = parent.genes[slot] + donor.genes[slot]
            genes[slot] = rng.sample(mixed, k=max(1, len(mixed) // 2))
    return Genome(genes)


@operator('graft', 1.2)
def graft(rng, market, parent, donors):
    """ Пересадка удачного гена из другого генома (донора) в родителя."""
    if not donors:
        return grow(rng, market, parent, donors)
    donor = rng.choice(donors)
    slot, index = any_gene(rng, donor)
    child = parent.copy()
    if len(child.genes[slot]) < SLOTS[slot][1]:
        child.genes[slot].append(donor.genes[slot][index])
    else:
        child.genes[slot][rng.randrange(len(child.genes[slot]))] = donor.genes[slot][index]
    return child


@operator('storm', 0.6)
def storm(rng, market, parent, donors):
    """ Несколько случайных операторов подряд — дальний прыжок."""
    child = parent
    names = [n for n in OPERATORS if n not in ('storm', 'fresh')]
    for _ in range(rng.randint(2, 4)):
        child = OPERATORS[rng.choice(names)][0](rng, market, child, donors)
    return child


class OperatorStats:
    """ Вес оператора = базовый вес × сглаженная доля успехов (потомок лучше родителя)."""

    def __init__(self, history: dict[str, tuple[int, int]] | None = None):
        self.tries = {name: 0 for name in OPERATORS}
        self.wins = {name: 0 for name in OPERATORS}
        for name, (tries, wins) in (history or {}).items():
            if name in OPERATORS:
                self.tries[name], self.wins[name] = tries, wins

    def weight(self, name: str) -> float:
        rate = (self.wins[name] + 1) / (self.tries[name] + 4)
        return OPERATORS[name][1] * max(0.05, rate) * (1 + 1 / math.sqrt(self.tries[name] + 1))

    def choose(self, rng) -> str:
        names = list(OPERATORS)
        return rng.choices(names, weights=[self.weight(n) for n in names])[0]

    def record(self, name: str, success: bool):
        self.tries[name] += 1
        self.wins[name] += int(success)
