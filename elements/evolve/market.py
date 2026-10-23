""" Рынок для абстрактного алгоритма: массивы признаков, кэши и слепая оценка генома.

Период данных делится на две части:
    поиск   — EVOLVE_START .. (последний день − EVOLVE_HOLDOUT_DAYS): здесь считается оценка,
              по которой отбираются геномы, и квантили для порогов генов;
    контроль — следующие EVOLVE_HOLDOUT_DAYS − EVOLVE_BLIND_DAYS дней: поиск их не видит; этап
              фиксируется (и вытесняет другой) только если здесь результат не хуже порога и не
              хуже вытесняемого — поэтому со временем контроль тоже «подгоняется»;
    слепая проверка — последние EVOLVE_BLIND_DAYS дней: считается, но ни на что не влияет —
              единственная по-настоящему нетронутая оценка этапа.
"""
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pandas as pd

from elements.evolve.genome import Genome
from elements.features.worker import FEATURES, trade_outcomes
from elements.reverse.engine import simulate_entries
from elements.selftune.blind import score_of
from elements.strategy.worker import tradable
from source.settings.settings import EVOLVE_START, EVOLVE_HOLDOUT_DAYS, EVOLVE_BLIND_DAYS


EXTRA_FEATURES = ['atr_ratio', 'div_yield', 'div_days']
MARKET_FEATURES = ['idx_r1', 'idx_r5', 'idx_dist_sma50', 'breadth50', 'mkt_r1', 'mkt_disp']
# Вариантов правила выхода ~1800 (удержание × стоп × тейк × на открытии), расчёт одного ~1.5 с:
# держим 96 последних (float32/int32, ~3 МБ на вариант при 400 тыс. строк).
EXIT_CACHE = 96
DERIVED_CACHE = 96  # производных рядов (изменение, z-оценка) в памяти
NEVER = np.iinfo(np.int32).max // 2


class EMarket:

    def __init__(self, panel: pd.DataFrame, start: str = EVOLVE_START, holdout_days: int = EVOLVE_HOLDOUT_DAYS,
                 blind_days: int = EVOLVE_BLIND_DAYS):
        panel = panel.sort_values(['DATE', 'SECID']).reset_index(drop=True)
        by_date = panel.groupby('DATE')
        panel['breadth50'] = by_date['dist_sma50'].transform(lambda s: (s > 0).mean())
        panel['mkt_r1'] = by_date['r1'].transform('median')
        panel['mkt_disp'] = by_date['r1'].transform('std')
        self.panel = panel
        self.dates = np.sort(panel['DATE'].unique())
        self.didx = np.searchsorted(self.dates, panel['DATE'].to_numpy())
        self.weekday = panel['DATE'].dt.weekday.to_numpy()
        self.secid = pd.factorize(panel['SECID'])[0]
        self.atr_pct = panel['atr_pct'].to_numpy(dtype=float)
        self.base_ok = tradable(panel).to_numpy(dtype=bool)
        self.div_days = panel['div_days'].to_numpy(dtype=float) if 'div_days' in panel else \
            np.full(len(panel), np.nan)
        stock = [f for f in FEATURES + EXTRA_FEATURES if f in panel and f not in MARKET_FEATURES]
        self.features = {'stock': stock, 'market': [f for f in MARKET_FEATURES if f in panel]}
        self.values = {f: panel[f].to_numpy(dtype=float) for f in stock + self.features['market']}
        self.groups = list(panel.groupby('SECID', sort=False).indices.values())

        self.last = len(self.dates) - 1
        self.first = int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(start))))
        self.hold_start = max(self.first + 252, self.last - holdout_days + 1)
        self.search_end = self.hold_start - 1
        # Слепая часть — не больше половины контрольного периода.
        self.blind_start = max(self.hold_start + (self.last - self.hold_start + 1) // 2,
                               self.last - blind_days + 1)
        self.guard_end = self.blind_start - 1
        self.search_rows = (self.didx >= self.first) & (self.didx <= self.search_end)
        self._ranks: dict[str, np.ndarray] = {}
        self._quantiles: dict[tuple, float] = {}
        self._exits: OrderedDict = OrderedDict()
        self._derived: OrderedDict = OrderedDict()

    def date(self, day: int):
        return pd.Timestamp(self.dates[day]).date()

    # --- кэшируемые величины -----------------------------------------------------------
    def rank(self, feature: str) -> np.ndarray:
        """ Место в кросс-секции дня (0..1), только по данным этого дня."""
        if feature not in self._ranks:
            self._ranks[feature] = (
                pd.Series(self.values[feature]).groupby(self.didx).rank(pct=True).to_numpy()
            )
        return self._ranks[feature]

    def quantile(self, feature: str, q: float) -> float:
        """ Квантиль признака по периоду поиска (контрольный период не участвует)."""
        key = (feature, round(q, 4))
        if key not in self._quantiles:
            values = self.values[feature][self.search_rows]
            values = values[np.isfinite(values)]
            self._quantiles[key] = float(np.quantile(values, q)) if len(values) else 0.0
        return self._quantiles[key]

    # --- производные ряды по бумаге (своды правил «динамика») -----------------------------
    def derived(self, key: tuple, compute) -> np.ndarray:
        """ Кэш производного ряда (по строкам панели); строки бумаги идут по датам."""
        if key in self._derived:
            self._derived.move_to_end(key)
            return self._derived[key]
        values = compute()
        self._derived[key] = values
        if len(self._derived) > DERIVED_CACHE:
            self._derived.popitem(last=False)
        return values

    def change(self, feature: str, lag: int) -> np.ndarray:
        """ Изменение признака бумаги за lag торговых дней."""
        def compute():
            series = pd.Series(self.values[feature])
            return (series - series.groupby(self.secid).shift(lag)).to_numpy()
        return self.derived(('change', feature, lag), compute)

    def zscore(self, feature: str, window: int) -> np.ndarray:
        """ Отклонение признака от своего среднего за window дней (в стандартных отклонениях)."""
        def compute():
            grouped = pd.Series(self.values[feature]).groupby(self.secid)
            mean = grouped.transform(lambda s: s.rolling(window, min_periods=window // 2).mean())
            std = grouped.transform(lambda s: s.rolling(window, min_periods=window // 2).std())
            with np.errstate(invalid='ignore', divide='ignore'):
                return ((self.values[feature] - mean) / std.replace(0, np.nan)).to_numpy()
        return self.derived(('zscore', feature, window), compute)

    def streak(self) -> np.ndarray:
        """ Серия дней роста (>0) или падения (<0) закрытия подряд, со знаком."""
        def compute():
            sign = np.sign(np.nan_to_num(self.values['r1']))
            result = np.zeros(len(sign))
            for rows in self.groups:
                run = 0.0
                for row in rows:
                    s = sign[row]
                    run = run + s if s != 0 and np.sign(run) == s else s
                    result[row] = run
            return result
        return self.derived(('streak',), compute)

    def persist(self, feature: str, q: float, n: int) -> np.ndarray:
        """ Сколько из последних n дней бумага была выше места q по признаку."""
        def compute():
            hits = pd.Series((self.rank(feature) > q).astype(float))
            return hits.groupby(self.secid).transform(lambda s: s.rolling(n, min_periods=1).sum()).to_numpy()
        return self.derived(('persist', feature, q, n), compute)

    def drank(self, key: tuple, values: np.ndarray) -> np.ndarray:
        """ Место производного ряда в кросс-секции дня (0..1)."""
        return self.derived(('rank',) + key, lambda: pd.Series(values).groupby(self.didx).rank(pct=True).to_numpy())

    def dquantile(self, key: tuple, values: np.ndarray, q: float) -> float:
        """ Квантиль производного ряда по периоду поиска."""
        full = (key, round(q, 4))
        if full not in self._quantiles:
            part = values[self.search_rows]
            part = part[np.isfinite(part)]
            self._quantiles[full] = float(np.quantile(part, q)) if len(part) else 0.0
        return self._quantiles[full]

    def outcome(self, hold: int, stop: float, take: float, at_open: int) -> SimpleNamespace:
        """ Исходы сделок по каждой строке для правила выхода (вход — открытие t+1)."""
        key = (hold, stop, take, at_open)
        if key in self._exits:
            self._exits.move_to_end(key)
            return self._exits[key]
        ret = np.full(len(self.didx), np.nan, dtype=np.float32)
        eidx = np.full(len(self.didx), NEVER, dtype=np.int32)
        columns = self.panel[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'ATR']]
        for rows in self.groups:
            result = trade_outcomes(columns.iloc[rows], hold, stop, take, bool(at_open))
            offsets = result['EXIT_OFFSET'].to_numpy()
            done = np.isfinite(offsets)
            ret[rows] = result['TRADE_RET'].to_numpy(dtype=float)
            positions = np.nonzero(done)[0]
            eidx[rows[done]] = self.didx[rows[positions + offsets[done].astype(int)]]
        ok = self.base_ok & (np.isnan(self.div_days) | (self.div_days > hold))
        view = SimpleNamespace(ret=ret, eidx=eidx, dates=self.dates, secid=self.secid,
                               atr_pct=self.atr_pct, ok=ok)
        self._exits[key] = view
        if len(self._exits) > EXIT_CACHE:
            self._exits.popitem(last=False)
        return view

    # --- оценка генома ----------------------------------------------------------------
    def picks(self, genome: Genome) -> tuple[SimpleNamespace, list]:
        exit_view = genome.genes['exit'][0].evaluate(self)
        slot, top = genome.genes['size'][0].evaluate(self)
        need = genome.genes['logic'][0].evaluate(self)
        filters = genome.genes['filter']
        votes = np.zeros(len(self.didx), dtype=np.int16)
        for gene in filters:
            votes += gene.evaluate(self).astype(np.int16)
        mask = exit_view.ok & (votes >= max(1, int(np.ceil(need * len(filters) - 1e-9))))
        score = np.zeros(len(self.didx))
        for gene in genome.genes['score']:
            score = score + gene.evaluate(self)
        rows = np.nonzero(mask & (self.didx >= self.first))[0]
        if not len(rows):
            return exit_view, []
        rows = rows[np.lexsort((-score[rows], self.didx[rows]))]
        days = self.didx[rows]
        starts = np.r_[0, np.nonzero(np.diff(days))[0] + 1]
        position = np.arange(len(rows)) - np.repeat(starts, np.diff(np.r_[starts, len(rows)]))
        rows = rows[position < top]
        return exit_view, [(int(self.didx[r]), int(r), slot, float(score[r])) for r in rows]

    def evaluate(self, genome: Genome, holdout: bool = False) -> dict:
        """ Оценка на периоде поиска (весь и две половины), по запросу — на контрольном."""
        view, picks = self.picks(genome)
        middle = (self.first + self.search_end) // 2
        search = simulate_entries(view, picks, self.first, self.search_end)
        parts = [
            search['stats'],
            simulate_entries(view, picks, self.first, middle)['stats'],
            simulate_entries(view, picks, middle + 1, self.search_end)['stats'],
        ]
        parts = [{**stats, 'SCORE': score_of(stats)} for stats in parts]
        result = {'stats': parts[0], 'halves': parts[1:], 'signals': len(picks), 'daily': search['daily'],
                  'search_start': self.date(self.first), 'search_end': self.date(self.search_end),
                  'holdout_start': self.date(self.hold_start), 'blind_start': self.date(self.blind_start),
                  'data_end': self.date(self.last)}
        if holdout:
            # Контроль (для фиксации и защиты) и слепая проверка (только для сведения).
            stats = simulate_entries(view, picks, self.hold_start, self.guard_end)['stats']
            result['holdout'] = {**stats, 'SCORE': score_of(stats)}
            stats = simulate_entries(view, picks, self.blind_start, self.last)['stats']
            result['blind'] = {**stats, 'SCORE': score_of(stats)}
        return result
