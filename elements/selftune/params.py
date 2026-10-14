""" Пространства поиска самокоррекции и хранение действующих параметров в БД.

Три вида параметров (таблица algo_params, по одной активной записи на вид):
    strategy   — пороги правиловых стратегий (то, чем торгуют сигналы);
    correction — коррекция по текущей статистике: веса стратегий по их недавним
                 теневым сделкам и размер позиции;
    tuner      — как пороги переподбираются в слепом бэктесте и для живой торговли
                 (окно обучения, частота, целевая функция).
"""
import json
import logging
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Session

from elements.reverse.engine import DEFAULT_REVERSE
from elements.strategy.worker import DEFAULT_PARAMS, set_active_params
from source.settings.settings import handler
from source.sql.main import main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

RULES = list(DEFAULT_PARAMS)

# Допустимые значения порогов. -9 / -1 — условие выключено (см. strategy.worker).
STRATEGY_GRID = {
    'mean_reversion': {
        'rsi': [5, 10, 15, 20], 'sma200': [-9.0, -0.15, -0.1, 0.0, 0.05],
        'r20': [-9.0, 0.0, 0.03], 'clv': [-1.0, 0.3, 0.4, 0.5, 0.6], 'regime': [0, 1],
    },
    'momentum': {
        'vol': [1.5, 1.7, 2.0, 2.5, 3.0], 'high': [-0.01, -0.005, 0.0], 'r5cs': [0.0, 0.02],
        'clv': [-1.0, 0.5, 0.7], 'sma200': [-9.0, 0.0], 'gap': [-9.0, -0.01], 'regime': [0, 1],
    },
    'breakout': {
        'high': [-0.005, -0.001], 'vol': [1.3, 1.8, 2.2, 3.0], 'clv': [0.6, 0.7, 0.85],
        'sma50': [-9.0, 0.0, 0.05], 'regime': [0, 1],
    },
    'trend_pullback': {
        'sma50': [-0.02, 0.0, 0.03], 'r20': [0.03, 0.05, 0.08],
        'r3': [-0.02, -0.03, -0.035, -0.045], 'rsi': [25, 30, 40],
        'clv': [-1.0, 0.3, 0.4, 0.5], 'regime': [0, 1],
    },
    'squeeze': {
        'atr': [0.7, 0.8, 0.9], 'rng': [0.6, 0.7, 0.8], 'clv': [0.6, 0.7, 0.8],
        'sma20': [-9.0, 0.0], 'idx': [-9.0, 0.0], 'regime': [0, 1],
    },
    'capitulation': {
        'r1': [-0.03, -0.04, -0.05], 'vol': [1.0, 1.3, 2.0], 'idx': [-9.0, -0.02],
        'clv': [0.3, 0.5], 'sma200': [-0.2, -0.1], 'regime': [0, 1],
    },
}

# Стартовая точка слепого бэктеста — пороги, заданные до подбора на истории 2020-2026
# (иначе в «слепой» прогон попало бы знание о будущем).
NEUTRAL_PARAMS = {
    'mean_reversion': {'rsi': 10, 'sma200': 0.0, 'r20': -9.0, 'clv': -1.0, 'regime': 0},
    'momentum': {'vol': 1.5, 'high': -0.01, 'r5cs': 0.0, 'clv': -1.0, 'sma200': -9.0,
                 'gap': -9.0, 'regime': 1},
    'breakout': {'high': -0.001, 'vol': 1.3, 'clv': 0.7, 'sma50': -9.0, 'regime': 1},
    'trend_pullback': {'sma50': 0.03, 'r20': 0.05, 'r3': -0.02, 'rsi': 25, 'clv': -1.0,
                       'regime': 0},
    'squeeze': {'atr': 0.8, 'rng': 0.7, 'clv': 0.6, 'sma20': 0.0, 'idx': -9.0, 'regime': 0},
    'capitulation': {'r1': -0.03, 'vol': 2.0, 'idx': -9.0, 'clv': 0.5, 'sma200': -0.2,
                     'regime': 0},
}

# Коррекция по текущей статистике: вес стратегии w по её теневым сделкам, закрытым
# за последние lookback торговых дней (n сделок, средняя доходность avg):
#   n < min_trades  -> w = 1 (статистики мало);
#   avg <= cut      -> w = w_min (стратегия «остыла», 0 — не торговать);
#   avg >= boost    -> w = w_max (стратегия «в форме»);
#   иначе           -> w = 1.
# Позиция = slot * w доли капитала (не больше риска RISK_PER_TRADE до стопа).
DEFAULT_CORRECTION = {
    'lookback': 63, 'min_trades': 5, 'cut': 0.0, 'w_min': 0.0,
    'boost': 0.005, 'w_max': 1.5, 'slot': 0.125,
}
CORRECTION_GRID = {
    'lookback': [21, 42, 63, 126, 250],
    'min_trades': [3, 5, 8, 12],
    'cut': [-0.01, -0.005, -0.002, 0.0, 0.002],
    'w_min': [0.0, 0.25, 0.5, 1.0],
    'boost': [0.003, 0.005, 0.01, 0.02, 9.0],
    'w_max': [1.0, 1.5, 2.0],
    'slot': [0.05, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3],
}

# Переподбор порогов: окно обучения (торговых дней), шаг переподбора, длина блока
# целевой функции (сумма блоков + lam * худший блок), проходы покоординатного спуска,
# минимум сделок стратегии в окне обучения, выключение убыточных на обучении стратегий.
DEFAULT_TUNER = {
    'train_days': 500, 'reopt_days': 21, 'block_days': 63, 'lam': 2.0,
    'passes': 2, 'min_trades': 20, 'allow_off': 0,
}
TUNER_GRID = {
    'train_days': [250, 500, 750, 1000],
    'reopt_days': [21, 63, 126],
    'block_days': [63, 126],
    'lam': [0.0, 1.0, 2.0, 3.0],
    'passes': [1, 2],
    'min_trades': [10, 20, 40],
    'allow_off': [0, 1],
}

DEFAULTS = {
    'strategy': DEFAULT_PARAMS, 'correction': DEFAULT_CORRECTION, 'tuner': DEFAULT_TUNER,
    # Реверсивный алгоритм: собственный пул параметров и состояние жизненного цикла.
    'reverse': DEFAULT_REVERSE, 'reverse_state': {},
}


def params_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def load_active(kind: str) -> dict:
    """ Активные параметры вида kind из БД (поверх значений по умолчанию)."""
    with Session(bind=main_engine) as s:
        row = s.execute(
            sa.select(tables.AlgoParam.PARAMS)
            .where(tables.AlgoParam.KIND == kind, tables.AlgoParam.ACTIVE.is_(True))
            .order_by(tables.AlgoParam.id.desc()).limit(1)
        ).scalar()
    stored = json.loads(row) if row else {}
    if kind == 'strategy':
        return {name: {**DEFAULT_PARAMS[name], **stored.get(name, {})} for name in RULES}
    return {**DEFAULTS[kind], **stored}


def load_active_record(kind: str) -> dict | None:
    with Session(bind=main_engine) as s:
        row = s.execute(
            sa.select(tables.AlgoParam.__table__)
            .where(tables.AlgoParam.KIND == kind, tables.AlgoParam.ACTIVE.is_(True))
            .order_by(tables.AlgoParam.id.desc()).limit(1)
        ).mappings().first()
    return dict(row) if row else None


def save_active(kind: str, params: dict, metrics: dict | None = None, data_end=None, note: str = ''):
    """ Новая активная запись вида kind; прежняя остаётся в истории (ACTIVE = False)."""
    metrics = metrics or {}
    with Session(bind=main_engine) as s:
        s.execute(
            sa.update(tables.AlgoParam)
            .where(tables.AlgoParam.KIND == kind, tables.AlgoParam.ACTIVE.is_(True))
            .values(ACTIVE=False)
        )
        s.add(tables.AlgoParam(
            KIND=kind, PARAMS=json.dumps(params, sort_keys=True), ACTIVE=True,
            CREATED_AT=datetime.now(), DATA_END=data_end,
            BLIND_CAGR=metrics.get('CAGR'), BLIND_DD=metrics.get('MAX_DRAWDOWN'),
            BLIND_SHARPE=metrics.get('SHARPE'), BLIND_SCORE=metrics.get('SCORE'),
            BLIND_TRADES=metrics.get('TRADES'), NOTE=note[:500],
        ))
        s.commit()


def update_active_metrics(kind: str, metrics: dict, data_end):
    """ Пересчитанный на свежих данных результат активной записи (без смены параметров)."""
    with Session(bind=main_engine) as s:
        s.execute(
            sa.update(tables.AlgoParam)
            .where(tables.AlgoParam.KIND == kind, tables.AlgoParam.ACTIVE.is_(True))
            .values(
                DATA_END=data_end, BLIND_CAGR=metrics.get('CAGR'),
                BLIND_DD=metrics.get('MAX_DRAWDOWN'), BLIND_SHARPE=metrics.get('SHARPE'),
                BLIND_SCORE=metrics.get('SCORE'), BLIND_TRADES=metrics.get('TRADES'),
            )
        )
        s.commit()


def apply_active_strategy_params() -> dict:
    """ Подставить действующие пороги из БД в стратегии (вызывать перед расчётом сигналов)."""
    return set_active_params(load_active('strategy'))
