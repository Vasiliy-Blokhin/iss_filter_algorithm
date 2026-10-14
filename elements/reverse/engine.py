""" Реверсивный алгоритм: логика строится «от обратного» — от исходов сделок к их причинам.

Обычная стратегия: правило -> сделки -> результат. Здесь наоборот:
  1. исход: из сделок, закрытых к дню решения (окно lookback), берутся лучшие (top_q) —
     «что реально заработало»;
  2. причины: среди условий-«атомов» (хвосты признаков цены/оборота/рынка и новостного фона:
     «RSI(2) < 12», «оборот > 2.1× медианы», «новости позитивные») ищутся одиночные и парные,
     которые чаще встречались перед лучшими сделками, чем в среднем (лифт), и дали
     положительный средний результат в обеих половинах окна (с усадкой к среднему рынка);
  3. жизненный цикл: найденное правило рождается кандидатом и торгует только после проверки
     на сделках, открытых ПОСЛЕ его рождения (вне выборки): подтвердилось — «живое»,
     нет — умирает; живое умирает, когда остывает на последних сделках;
  4. самокоррекция: по собственным закрытым сигналам алгоритм ужесточает или ослабляет
     требования (уровень строгости) и размер позиции; вес новостей оценивается по
     фактическим исходам сделок с новостями (усадка к априорному значению).
Гипотезы может предлагать и локальная ИИ-модель (elements/reverse/mind.py) — они проходят
ту же проверку вне выборки, что и найденные статистикой.

Все решения дня d используют только сделки с выходом не позже d (известны после закрытия d),
поэтому одна и та же механика даёт и живую торговлю, и честный слепой бэктест.
"""
import json

import numpy as np
import pandas as pd

from elements.strategy.worker import tradable, dividend_safe
from source.settings.settings import (
    MAX_SIGNALS,
    MAX_PER_SECTOR,
    STOP_ATR,
    RISK_PER_TRADE,
    MAX_POSITIONS,
)
from source.settings.sectors import SECTORS


# Признаки, из хвостов которых строятся атомы (пороги — квантили окна опыта).
ATOM_FEATURES = [
    'r1', 'r3', 'r5', 'r10', 'r20', 'r5_cs', 'rel5',
    'rsi2', 'rsi14',
    'dist_sma20', 'dist_sma50', 'dist_sma200', 'dist_high20', 'dist_low20',
    'atr_pct', 'atr_ratio', 'range_pct', 'clv', 'gap', 'vol_ratio', 'vol_ratio5',
    'idx_r1', 'idx_r5', 'idx_dist_sma50',
]
LOW_QUANTILES = (0.1, 0.2, 0.33)
HIGH_QUANTILES = (0.67, 0.8, 0.9)
# Новостной фон (elements/reverse/news.py): пороги фиксированные.
NEWS_FEATURES = ['news_score', 'news_count', 'mkt_news']
NEWS_ATOMS = [
    ('news_score', '>', 0.15), ('news_score', '<', -0.15), ('news_count', '>', 0.5),
    ('mkt_news', '>', 0.15), ('mkt_news', '<', -0.15),
]

FEATURE_TITLES = {
    'r1': 'доходность за день', 'r3': 'доходность за 3 дня', 'r5': 'доходность за 5 дней',
    'r10': 'доходность за 10 дней', 'r20': 'доходность за 20 дней',
    'r5_cs': '5-дневная доходность относительно медианы рынка', 'rel5': '5 дней относительно IMOEX',
    'rsi2': 'RSI(2)', 'rsi14': 'RSI(14)',
    'dist_sma20': 'отклонение от SMA20', 'dist_sma50': 'отклонение от SMA50',
    'dist_sma200': 'отклонение от SMA200', 'dist_high20': 'расстояние до 20-дн. максимума',
    'dist_low20': 'расстояние от 20-дн. минимума', 'atr_pct': 'ATR в % цены',
    'atr_ratio': 'ATR к среднему за 60 дней', 'range_pct': 'диапазон дня в % цены',
    'clv': 'положение закрытия в диапазоне дня (1 = у хая)', 'gap': 'гэп открытия',
    'vol_ratio': 'оборот к медиане 20 дней', 'vol_ratio5': 'оборот 5 дней к медиане',
    'idx_r1': 'IMOEX за день', 'idx_r5': 'IMOEX за 5 дней', 'idx_dist_sma50': 'IMOEX к SMA50',
    'news_score': 'новостной фон бумаги (-1..1)', 'news_count': 'число новостей о бумаге',
    'mkt_news': 'новостной фон рынка (-1..1)',
}

DEFAULT_REVERSE = {
    'lookback': 250,       # окно опыта, торговых дней
    'reopt': 10,           # как часто искать новые правила
    'top_q': 0.2,          # доля лучших сделок — «исход», от которого идём назад
    'min_support': 60,     # минимум сделок правила в окне
    'prior_k': 30,         # усадка средних к среднему рынка (в сделках)
    'birth_edge': 0.002,   # средняя сделка правила в окне (после усадки) для рождения
    'min_lift': 1.15,      # во сколько раз чаще правило встречается среди лучших сделок
    'new_rules': 6,        # новых правил за поиск
    'max_rules': 40,       # кандидатов в пуле
    'promote_n': 25,       # сделок вне выборки для решения о кандидате
    'promote_edge': 0.001, # средняя вне выборки (усадка к 0) для перевода в живые
    'death_edge': -0.001,  # ниже — правило умирает
    'live_window': 60,     # последних сделок живого правила для проверки «остывания»
    'max_age': 126,        # кандидат без решения дольше — умирает
    'edge_step': 0.001,    # шаг строгости (самокоррекция)
    'adapt_every': 21,
    'adapt_window': 63,
    'adapt_min': 10,
    'adapt_good': 0.004,
    'slot': 0.1,           # доля капитала на позицию при строгости 0
    'news_prior': 0.004,   # априорный вклад новостного фона +1 в ожидаемую доходность
    'news_k': 50,          # усадка измеренного вклада новостей к априорному
    'news_veto': -0.4,     # фон бумаги ниже — не покупать
    'rebirth_days': 63,    # умершее правило не возрождается раньше
    'excess': 1,           # искать по доходности относительно средней сделки дня
    'market_atoms': 1,     # атомы рынка (IMOEX, фон рынка) в правилах
    'gate': 1,             # реальные сделки только при плюсовой собственной истории
    'gate_edge': 0.0,
}
REVERSE_GRID = {
    'lookback': [126, 250, 500],
    'reopt': [5, 10, 21],
    'top_q': [0.1, 0.2, 0.3],
    'min_support': [30, 60, 120],
    'prior_k': [10, 30, 100],
    'birth_edge': [0.0, 0.001, 0.002, 0.004],
    'min_lift': [1.0, 1.15, 1.3],
    'new_rules': [3, 6, 10],
    'max_rules': [20, 40, 80],
    'promote_n': [15, 25, 50],
    'promote_edge': [0.0, 0.001, 0.002, 0.004],
    'death_edge': [-0.003, -0.001, 0.0],
    'live_window': [30, 60, 120],
    'max_age': [63, 126, 250],
    'slot': [0.05, 0.1, 0.15, 0.2],
    'news_prior': [0.0, 0.002, 0.004, 0.008],
    'news_veto': [-0.6, -0.4, -0.2],
    'excess': [0, 1],
    'market_atoms': [0, 1],
    'gate_edge': [0.0, 0.002, 0.004],
}
MIN_LEVEL, MAX_LEVEL = -1, 3
BLOCKS = 4  # блоков окна опыта для проверки устойчивости правила


def rule_key(atoms: list) -> str:
    return json.dumps(sorted([a[0], a[1], round(float(a[2]), 6)] for a in atoms), ensure_ascii=False)


def describe_atom(atom) -> str:
    feature, op, value = atom
    return f'{feature} {op} {value:.4g}'


def describe(atoms: list) -> str:
    return ' и '.join(describe_atom(a) for a in atoms)


def round_threshold(value: float) -> float:
    """ Порог-квантиль, округлённый до 3 значащих цифр (читаемость правил)."""
    if not np.isfinite(value) or value == 0:
        return float(value)
    return float(f'{value:.3g}')


class RMarket:
    """ Панель, отсортированная по дате, и её массивы для реверсивного алгоритма."""

    def __init__(self, panel: pd.DataFrame):
        panel = panel.sort_values(['DATE', 'SECID']).reset_index(drop=True)
        for column in NEWS_FEATURES:
            if column not in panel:
                panel[column] = np.nan
        self.panel = panel
        self.dates = np.sort(panel['DATE'].unique())
        self.didx = np.searchsorted(self.dates, panel['DATE'].to_numpy())
        exits = pd.to_datetime(panel['EXIT_DATE']).to_numpy() if 'EXIT_DATE' in panel else \
            np.full(len(panel), np.datetime64('NaT'))
        known = ~np.isnat(exits)
        self.eidx = np.full(len(panel), np.iinfo(np.int64).max // 2)
        self.eidx[known] = np.searchsorted(self.dates, exits[known])
        self.ret = panel['TRADE_RET'].to_numpy(dtype=float) if 'TRADE_RET' in panel else \
            np.full(len(panel), np.nan)
        self.live_ok = (tradable(panel) & dividend_safe(panel)).to_numpy(dtype=bool)
        self.known = self.live_ok & np.isfinite(self.ret) & known
        self.secid = pd.factorize(panel['SECID'])[0]
        sectors = panel['SECID'].map(SECTORS).fillna('unknown')
        self.sector = pd.factorize(sectors)[0]
        self.sector[(sectors == 'unknown').to_numpy()] = -1
        self.atr_pct = panel['atr_pct'].to_numpy(dtype=float)
        self.values = {
            name: panel[name].to_numpy(dtype=float)
            for name in ATOM_FEATURES + NEWS_FEATURES if name in panel
        }
        self.day_start = np.searchsorted(self.didx, np.arange(len(self.dates) + 1))

    def day(self, date) -> int:
        return int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(date))))

    def date(self, day: int):
        return pd.Timestamp(self.dates[day]).date()

    def mask(self, atoms: list, rows: slice | np.ndarray = slice(None)) -> np.ndarray:
        result = None
        for feature, op, value in atoms:
            column = self.values.get(feature)
            if column is None:
                return np.zeros(len(self.didx[rows]), dtype=bool)
            column = column[rows]
            with np.errstate(invalid='ignore'):
                part = column > value if op == '>' else column < value
            result = part if result is None else result & part
        return result if result is not None else np.zeros(len(self.didx[rows]), dtype=bool)


class Rule:
    """ Правило-гипотеза и его жизненный цикл."""

    def __init__(self, atoms: list, born: int, origin: str = 'mined', status: str = 'candidate',
                 is_n: int = 0, is_mean: float = 0.0, is_post: float = 0.0, note: str = '',
                 died: int | None = None, promoted: int | None = None):
        self.atoms = [[a[0], a[1], float(a[2])] for a in atoms]
        self.key = rule_key(self.atoms)
        self.born, self.died, self.promoted = born, died, promoted
        self.origin, self.status, self.note = origin, status, note
        self.is_n, self.is_mean, self.is_post = is_n, is_mean, is_post
        self.oos_n, self.oos_mean, self.post, self.recent_post = 0, 0.0, 0.0, 0.0
        self.match = None  # все строки рынка, где выполнено правило
        self.exits = self.cum = None

    @property
    def features(self) -> frozenset:
        return frozenset((a[0], a[1]) for a in self.atoms)

    def attach(self, market: RMarket):
        """ Строки рынка, подходящие под правило, и история вне выборки (после рождения).

        Наблюдение — день, а не бумаго-день: сделки одного дня сильно связаны (общее
        движение рынка), поэтому день с 10 совпадениями весит как один (средняя за день)."""
        self.match = market.mask(self.atoms)
        rows = np.nonzero(self.match & market.known & (market.didx >= self.born))[0]
        days, first, size = np.unique(market.didx[rows], return_index=True, return_counts=True)
        if len(rows):
            means = np.add.reduceat(market.ret[rows], first) / size
            exits = np.maximum.reduceat(market.eidx[rows], first)
        else:
            means = exits = np.array([])
        order = np.argsort(exits, kind='stable')
        self.exits = exits[order].astype(int)
        self.cum = np.concatenate([[0.0], np.cumsum(means[order])])

    def oos(self, day: int, last: int | None = None) -> tuple[int, float]:
        """ (дней, сумма средних по дням) вне выборки, закрытых к дню day; last — только последние."""
        hi = int(np.searchsorted(self.exits, day, side='right'))
        lo = max(0, hi - last) if last else 0
        return hi - lo, float(self.cum[hi] - self.cum[lo])


class ReverseEngine:
    """ Пул правил, их жизненный цикл и выбор бумаг на каждый день."""

    def __init__(self, market: RMarket, params: dict, state: dict | None = None):
        self.market = market
        self.p = {**DEFAULT_REVERSE, **(params or {})}
        state = state or {}
        self.level = int(state.get('level', 0))
        self.news_weight = float(state.get('news_weight', self.p['news_prior']))
        self.last_mine = state.get('last_mine')  # номер дня
        self.last_adapt = state.get('last_adapt')
        self.trading = bool(state.get('trading', False))
        self.rules: list[Rule] = []
        self.picks: list[tuple[int, int, float, float]] = []  # (день, строка, вес, оценка)
        self.journal: list[tuple[int, str, str]] = []  # (день, вид, текст)

    # --- состояние -----------------------------------------------------------------------
    def state(self, day: int) -> dict:
        return {
            'level': self.level, 'news_weight': round(self.news_weight, 6),
            'last_mine': self.last_mine, 'last_adapt': self.last_adapt, 'last_day': day,
            'trading': self.trading,
        }

    def edge(self, name: str) -> float:
        return self.p[name] + self.level * self.p['edge_step']

    def slot(self) -> float:
        return self.p['slot'] * max(0.25, 1 - 0.2 * self.level)

    def alive(self) -> list[Rule]:
        return [rule for rule in self.rules if rule.status in ('candidate', 'live')]

    def log(self, day: int, kind: str, text: str):
        self.journal.append((day, kind, text))

    def add_rule(self, rule: Rule) -> bool:
        if any(r.key == rule.key and (r.status != 'dead' or rule.born - (r.died or 0) < self.p['rebirth_days'])
               for r in self.rules):
            return False
        rule.attach(self.market)
        self.rules.append(rule)
        self.log(rule.born, 'birth', f'{rule.origin}: {describe(rule.atoms)} '
                 f'(в окне {rule.is_n} сд., средняя {rule.is_mean:+.2%})'
                 + (f' — {rule.note}' if rule.note else ''))
        return True

    # --- поиск «от обратного» ------------------------------------------------------------
    def atoms_for(self, rows: np.ndarray) -> list:
        atoms = []
        for name in ATOM_FEATURES:
            column = self.market.values.get(name)
            if column is None:
                continue
            values = column[rows]
            values = values[np.isfinite(values)]
            if len(values) < 100:
                continue
            for q in LOW_QUANTILES:
                atoms.append((name, '<', round_threshold(np.quantile(values, q))))
            for q in HIGH_QUANTILES:
                atoms.append((name, '>', round_threshold(np.quantile(values, q))))
        for atom in NEWS_ATOMS:
            if np.isfinite(self.market.values[atom[0]][rows]).any():
                atoms.append(atom)
        return atoms

    def mine(self, day: int) -> list[Rule]:
        """ Правила-кандидаты по сделкам с выходом не позже day (окно lookback)."""
        p, m = self.p, self.market
        first = max(0, day - p['lookback'])
        span = slice(m.day_start[first], m.day_start[day + 1])
        local = np.nonzero(m.known[span] & (m.eidx[span] <= day))[0]
        rows = local + span.start
        if len(rows) < max(500, 5 * p['min_support']):
            return []
        atoms = self.atoms_for(rows)
        if not p['market_atoms']:
            atoms = [atom for atom in atoms if not atom[0].startswith(('idx_', 'mkt_'))]
        y = m.ret[rows]
        if p['excess']:
            # «Исход» — доходность относительно средней сделки того же дня: ищем, чем
            # бумага отличалась, а не в какой день рынок рос.
            didx = m.didx[rows]
            day_sum = np.bincount(didx - first, weights=y)
            day_count = np.bincount(didx - first)
            y = y - (day_sum / np.maximum(day_count, 1))[didx - first]
        matrix = np.column_stack([m.mask([atom], rows) for atom in atoms]).astype(np.float32)
        win = (y >= np.quantile(y, 1 - p['top_q'])).astype(np.float32)
        y32 = y[:, None].astype(np.float32)
        count = matrix.T @ matrix
        total = matrix.T @ (matrix * y32)
        wins = matrix.T @ (matrix * win[:, None])
        # Устойчивость: средняя должна быть в плюсе в большинстве блоков окна.
        block = np.minimum(((m.didx[rows] - first) * BLOCKS) // max(1, day + 1 - first), BLOCKS - 1)
        positive = np.zeros_like(count)
        for number in range(BLOCKS):
            part = block == number
            part_count = matrix[part].T @ matrix[part]
            part_total = matrix[part].T @ (matrix[part] * y32[part])
            positive += (part_total > 0) & (part_count >= p['min_support'] / (2 * BLOCKS))

        base = float(y.mean())
        k = p['prior_k']
        with np.errstate(invalid='ignore', divide='ignore'):
            post = (total + k * base) / (count + k)
            lift = (wins / count) / p['top_q']
        names = np.array([a[0] for a in atoms])
        valid = (
            (count >= p['min_support']) & (positive >= BLOCKS - 1)
            & (post > self.edge('birth_edge')) & (lift >= p['min_lift'])
            & ((names[:, None] != names[None, :]) | np.eye(len(atoms), dtype=bool))
        )
        valid = np.triu(valid)
        pairs = np.argwhere(valid)
        if not len(pairs):
            return []
        order = np.argsort(-post[pairs[:, 0], pairs[:, 1]])
        taken = {rule.features for rule in self.alive()}
        born = []
        for i, j in pairs[order]:
            chosen = [atoms[i]] if i == j else [atoms[i], atoms[j]]
            features = frozenset((a[0], a[1]) for a in chosen)
            if features in taken:
                continue
            rule = Rule(chosen, born=day, is_n=int(count[i, j]),
                        is_mean=float(total[i, j] / count[i, j]), is_post=float(post[i, j]),
                        note=f'лифт {lift[i, j]:.2f}')
            if self.add_rule(rule):
                taken.add(features)
                born.append(rule)
            if len(born) >= p['new_rules']:
                break
        return born

    # --- жизненный цикл ------------------------------------------------------------------
    def update_rules(self, day: int):
        p, k = self.p, self.p['prior_k']
        for rule in self.alive():
            n, total = rule.oos(day)
            rule.oos_n, rule.oos_mean = n, total / n if n else 0.0
            rule.post = total / (n + k)  # усадка к нулю: торгуем только доказанное вне выборки
            if rule.status == 'candidate':
                if n >= p['promote_n'] and rule.post >= self.edge('promote_edge'):
                    rule.status, rule.promoted = 'live', day
                    self.log(day, 'promote', f'{describe(rule.atoms)}: вне выборки {n} сд., '
                             f'средняя {rule.oos_mean:+.2%} — в торговлю')
                elif n >= p['promote_n'] and rule.post < p['death_edge']:
                    self.kill(rule, day, f'не подтвердилось вне выборки ({n} сд., {rule.oos_mean:+.2%})')
                elif day - rule.born > p['max_age']:
                    self.kill(rule, day, f'нет подтверждения за {p["max_age"]} дн. ({n} сд.)')
            else:
                recent_n, recent_total = rule.oos(day, p['live_window'])
                rule.recent_post = recent_total / (recent_n + k)
                if recent_n >= p['promote_n'] and rule.recent_post < p['death_edge']:
                    self.kill(rule, day, f'остыло: последние {recent_n} сд., средняя '
                              f'{recent_total / recent_n:+.2%}')
        candidates = [r for r in self.alive() if r.status == 'candidate']
        for rule in sorted(candidates, key=lambda r: (r.is_post, r.born))[:max(0, len(candidates) - p['max_rules'])]:
            self.kill(rule, day, 'вытеснено более сильными кандидатами')

    def kill(self, rule: Rule, day: int, reason: str):
        rule.status, rule.died = 'dead', day
        rule.note = reason
        self.log(day, 'death', f'{describe(rule.atoms)}: {reason}')

    def adapt(self, day: int):
        """ Самокоррекция строгости по собственным сигналам, закрытым к дню day."""
        p, m = self.p, self.market
        if self.last_adapt is not None and day - self.last_adapt < p['adapt_every']:
            return
        self.last_adapt = day
        count, mean = self.own_record(day)
        if count < p['adapt_min']:
            self.learn_news(day)
            return
        old = self.level
        if mean < 0:
            self.level = min(MAX_LEVEL, self.level + 1)
        elif mean > p['adapt_good']:
            self.level = max(MIN_LEVEL, self.level - 1)
        if self.level != old:
            self.log(day, 'params', f'свои сигналы за {p["adapt_window"]} дн.: {count} сд., '
                     f'средняя {mean:+.2%} — строгость {old} -> {self.level}, позиция {self.slot():.1%}')
        self.learn_news(day)

    def learn_news(self, day: int):
        """ Вклад новостного фона: наклон доходности сделок по фону, с усадкой к априорному."""
        m, p = self.market, self.p
        score = m.values['news_score']
        rows = np.nonzero(m.known & (m.eidx <= day) & np.isfinite(score) & (score != 0))[0]
        if len(rows) < 20:
            return
        x, y = score[rows], m.ret[rows] - np.nanmean(m.ret[rows])
        slope = float((x * y).sum() / ((x * x).sum() + 1e-9))
        old = self.news_weight
        self.news_weight = (len(rows) * slope + p['news_k'] * p['news_prior']) / (len(rows) + p['news_k'])
        if abs(self.news_weight - old) > 1e-4:
            self.log(day, 'params', f'вклад новостей по {len(rows)} сделкам: наклон {slope:+.4f}, '
                     f'вес {old:+.4f} -> {self.news_weight:+.4f}')

    # --- решения дня ---------------------------------------------------------------------
    def choose(self, day: int) -> list[tuple[int, float, list[Rule]]]:
        """ Бумаги дня day: подходят под живые правила, оценка = лучшая доказанная средняя
        правил + вклад новостного фона. Возвращает [(строка, оценка, правила)]."""
        m, p = self.market, self.p
        span = slice(m.day_start[day], m.day_start[day + 1])
        live = [rule for rule in self.rules if rule.status == 'live']
        if not live or span.start == span.stop:
            return []
        scores = np.full(span.stop - span.start, -np.inf)
        matched: dict[int, list[Rule]] = {}
        for rule in live:
            hits = np.nonzero(rule.match[span])[0]
            for local in hits:
                matched.setdefault(local, []).append(rule)
                scores[local] = max(scores[local], rule.post)
        news = np.nan_to_num(m.values['news_score'][span])
        market_news = np.nan_to_num(m.values['mkt_news'][span])
        scores = scores + self.news_weight * (news + 0.5 * market_news)
        ok = m.live_ok[span] & np.isfinite(scores) & (scores > 0) & (news > p['news_veto'])
        order = [i for i in np.argsort(-scores) if ok[i]]
        chosen, per_sector = [], {}
        for local in order:
            row = span.start + local
            sector = m.sector[row]
            if sector >= 0 and per_sector.get(sector, 0) >= MAX_PER_SECTOR:
                continue
            per_sector[sector] = per_sector.get(sector, 0) + 1
            chosen.append((row, float(scores[local]), matched[local]))
            if len(chosen) >= MAX_SIGNALS:
                break
        return chosen

    def step(self, day: int) -> list[tuple[int, float, list[Rule]]]:
        """ Один торговый день (после закрытия day): статусы правил, самокоррекция,
        поиск новых правил, выбор бумаг на вход day + 1."""
        self.update_rules(day)
        self.adapt(day)
        if self.last_mine is None or day - self.last_mine >= self.p['reopt']:
            self.mine(day)
            self.last_mine = day
        chosen = self.choose(day)
        trading = self.confident(day)
        if trading != self.trading:
            count, mean = self.own_record(day)
            self.log(day, 'mode', ('реальные сделки' if trading else 'только теневые сделки')
                     + f': свои сигналы за {self.p["adapt_window"]} дн. — {count} сд., средняя {mean:+.2%}')
        self.trading = trading
        weight = self.slot() if self.trading else 0.0
        for row, score, _ in chosen:
            self.picks.append((day, row, weight, score))
        return chosen

    def own_record(self, day: int) -> tuple[int, float]:
        """ (число, средняя) собственных сигналов (реальных и теневых), закрытых за adapt_window."""
        m, p = self.market, self.p
        returns = [m.ret[row] for _, row, _, _ in self.picks
                   if day - p['adapt_window'] <= m.eidx[row] <= day and np.isfinite(m.ret[row])]
        return len(returns), float(np.mean(returns)) if returns else 0.0

    def confident(self, day: int) -> bool:
        """ Реальные сделки — только когда собственные сигналы последних adapt_window дней
        в плюсе (не меньше adapt_min); иначе алгоритм торгует «в тени» и продолжает учиться."""
        if not self.p['gate']:
            return True
        count, mean = self.own_record(day)
        return count >= self.p['adapt_min'] and mean > self.p['gate_edge']

    def run(self, first_day: int, last_day: int):
        for day in range(first_day, last_day + 1):
            self.step(day)
        return self


def simulate_entries(market: RMarket, picks: list, first_day: int, last_day: int) -> dict:
    """ Портфель по сигналам (день, строка, доля капитала): вход t+1, выход по исходу сделки,
    ограничения исполнителя — риск до стопа, MAX_POSITIONS, 100% капитала."""
    from elements.selftune.simulator import equity_stats
    daily = np.zeros(len(market.dates) + 1)
    open_positions = []
    taken, exposure_sum = [], 0.0
    by_day: dict[int, list] = {}
    for day, row, weight, score in picks:
        if first_day <= day <= last_day and np.isfinite(market.ret[row]) and market.eidx[row] < len(market.dates):
            by_day.setdefault(day, []).append((row, weight))
    for day in range(first_day, last_day + 1):
        open_positions = [p for p in open_positions if p[0] > day]
        exposure = sum(p[1] for p in open_positions)
        exposure_sum += exposure
        held = {p[2] for p in open_positions}
        for row, weight in by_day.get(day, []):
            secid = market.secid[row]
            if secid in held or len(open_positions) >= MAX_POSITIONS:
                continue
            size = weight
            if market.atr_pct[row] > 0:
                size = min(size, RISK_PER_TRADE / (STOP_ATR * market.atr_pct[row]))
            size = min(size, 1.0 - exposure)
            if size < 0.01:
                continue
            exposure += size
            open_positions.append((market.eidx[row], size, secid))
            held.add(secid)
            daily[market.eidx[row]] += size * market.ret[row]
            taken.append(row)
    returns = daily[first_day:last_day + 1]
    stats = equity_stats(returns)
    rets = market.ret[taken] if taken else np.array([])
    stats.update({
        'TRADES': len(taken), 'AVG_TRADE': float(rets.mean()) if len(rets) else 0.0,
        'WIN_RATE': float((rets > 0).mean()) if len(rets) else 0.0,
        'EXPOSURE': exposure_sum / max(1, last_day - first_day + 1),
    })
    return {'stats': stats, 'daily': returns}
