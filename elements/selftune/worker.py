""" Демон самокоррекции: в заданные окна времени проверяет кандидатов слепым бэктестом.

Каждая проверка (тик):
  1. если появились новые данные (или демон только запущен) — «refresh»: слепой бэктест
     действующих параметров по текущий день и пересчёт живых порогов стратегий
     (подбор на окне обучения, заканчивающемся последним днём данных) -> algo_params;
  2. иначе — один кандидат: мутация параметров коррекции (дёшево, секунды) или, в ночном
     окне, параметров переподбора (tuner, десятки секунд). Кандидат принимается, только если
     слепая оценка выше действующей на SELFTUNE_MARGIN и ни одна половина слепого периода
     не хуже действующей больше чем на SELFTUNE_HALF_TOLERANCE.
Все проверки пишутся в selftune_runs.
"""
import json
import logging
import random
from datetime import datetime
from time import sleep, monotonic
from zoneinfo import ZoneInfo

from elements.algorithm.worker import Algorithm
from elements.notification.worker import TelegramNotification as tlg
from elements.selftune.blind import BlindBacktest, SegmentStore
from elements.selftune.params import (
    CORRECTION_GRID, TUNER_GRID, load_active, load_active_record, save_active,
    update_active_metrics, params_key,
)
from source.settings.settings import (
    handler,
    TIMEZONE,
    SELFTUNE_WINDOWS,
    SELFTUNE_NIGHT,
    SELFTUNE_MARGIN,
    SELFTUNE_HALF_TOLERANCE,
    TARGET_CAGR,
)
from source.sql.main import SQLmain
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

SERVICE = 'selftune'
TUNER_SHARE = 0.35  # доля «тяжёлых» кандидатов (tuner) в ночном окне


def parse_windows(spec: str) -> list[tuple[int, int, int, int]]:
    """ '08:00-10:00/600-900;22:00-02:00/120-300' -> [(начало мин, конец мин, мин с, макс с)]."""
    windows = []
    for part in filter(None, (item.strip() for item in spec.split(';'))):
        span, _, interval = part.partition('/')
        start, end = (int(h) * 60 + int(m) for h, m in (t.split(':') for t in span.split('-')))
        low, high = (int(x) for x in (interval or '600-900').split('-'))
        windows.append((start, end, low, high))
    return windows


def in_window(minute: int, start: int, end: int) -> bool:
    return start <= minute < end if start < end else minute >= start or minute < end


def current_window(now: datetime, windows: list) -> tuple | None:
    minute = now.hour * 60 + now.minute
    return next((w for w in windows if in_window(minute, w[0], w[1])), None)


def seconds_to_next_window(now: datetime, windows: list) -> float:
    minute = now.hour * 60 + now.minute
    waits = [((w[0] - minute) % (24 * 60)) * 60 - now.second for w in windows]
    return max(1.0, min(wait if wait > 0 else 24 * 3600 for wait in waits))


def mutate(params: dict, grid: dict, rng: random.Random) -> dict:
    """ 1-2 параметра: соседнее значение сетки (70%) или случайное (30%)."""
    result = dict(params)
    for key in rng.sample(list(grid), k=rng.choice([1, 1, 2])):
        values = grid[key]
        if result.get(key) in values and rng.random() < 0.7:
            index = values.index(result[key]) + rng.choice([-1, 1])
            result[key] = values[min(max(index, 0), len(values) - 1)]
        else:
            result[key] = rng.choice(values)
    return result


def describe(stats: dict) -> str:
    return (f'CAGR {stats["CAGR"]:+.1%}, просадка {stats["MAX_DRAWDOWN"]:.1%}, '
            f'Sharpe {stats["SHARPE"]:.2f}, сделок {stats["TRADES"]}')


class SelfTuner(SQLmain):

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.store = SegmentStore()
        self.blind: BlindBacktest | None = None
        self.loaded_end = None
        self.incumbent: dict | None = None
        self.tried: set[str] = set()

    # --- данные ---------------------------------------------------------------
    def ensure_market(self) -> bool:
        """ Пересобрать панель, если в БД появились новые свечи. True — данные обновились."""
        last = self.run_sql_command('SELECT MAX("DATE") AS last FROM candles')[0]['last']
        if self.blind is not None and last == self.loaded_end:
            return False
        panel = Algorithm.build_panel(Algorithm.complete_candles())
        self.blind = BlindBacktest(panel, self.store)
        self.loaded_end = last
        self.tried.clear()
        logger.info(f'Самокоррекция: данные по {self.blind.data_end}')
        return True

    # --- проверки -------------------------------------------------------------
    def refresh(self) -> dict:
        """ Слепой результат действующих параметров по текущий день и новые живые пороги."""
        started = monotonic()
        tuner, correction = load_active('tuner'), load_active('correction')
        result = self.blind.run(tuner, correction)
        stats, data_end = result['stats'], result['data_end']
        for kind, params in (('tuner', tuner), ('correction', correction)):
            if load_active_record(kind) is None:
                save_active(kind, params, stats, data_end, 'начальные значения')
            else:
                update_active_metrics(kind, stats, data_end)
        live = self.blind.live_params(tuner)
        current = load_active_record('strategy')
        if current is None or json.loads(current['PARAMS']) != live:
            save_active('strategy', live, stats, data_end, f'переподбор по данным на {data_end}')
        else:
            update_active_metrics('strategy', stats, data_end)
        self.incumbent = {'tuner': tuner, 'correction': correction, 'result': result}
        self.log_run('refresh', tuner, correction, result, None, True, monotonic() - started,
                     f'живые пороги по {data_end}')
        logger.info(f'Самокоррекция refresh: слепой результат {describe(stats)} '
                    f'(цель {TARGET_CAGR:.0%} годовых)')
        return result

    def trial(self, heavy: bool) -> dict:
        started = monotonic()
        kind = 'tuner' if heavy and self.rng.random() < TUNER_SHARE else 'correction'
        tuner, correction = dict(self.incumbent['tuner']), dict(self.incumbent['correction'])
        for _ in range(20):
            if kind == 'tuner':
                tuner = mutate(self.incumbent['tuner'], TUNER_GRID, self.rng)
            else:
                correction = mutate(self.incumbent['correction'], CORRECTION_GRID, self.rng)
            key = params_key({'tuner': tuner, 'correction': correction})
            if key not in self.tried:
                break
        self.tried.add(key)
        result = self.blind.run(tuner, correction)
        best = self.incumbent['result']
        accepted, note = self.accept(result, best)
        self.log_run(kind, tuner, correction, result, best['stats']['SCORE'], accepted,
                     monotonic() - started, note)
        if accepted:
            changed = 'tuner' if kind == 'tuner' else 'correction'
            save_active(changed, tuner if kind == 'tuner' else correction, result['stats'],
                        result['data_end'], note)
            if kind == 'tuner':
                live = self.blind.live_params(tuner)
                save_active('strategy', live, result['stats'], result['data_end'],
                            'переподбор после смены tuner')
            self.incumbent = {'tuner': tuner, 'correction': correction, 'result': result}
            message = (f'Самокоррекция: приняты новые параметры ({changed}). Слепой бэктест '
                       f'{result["blind_start"]}..{result["data_end"]}: {describe(result["stats"])}; '
                       f'было CAGR {best["stats"]["CAGR"]:+.1%}.')
            logger.info(message)
            tlg.send_message(message)
        return result

    @staticmethod
    def accept(result: dict, best: dict) -> tuple[bool, str]:
        gain = result['stats']['SCORE'] - best['stats']['SCORE']
        halves_ok = all(
            new['SCORE'] >= old['SCORE'] - SELFTUNE_HALF_TOLERANCE
            for new, old in zip(result['halves'], best['halves'])
        )
        accepted = gain > SELFTUNE_MARGIN and halves_ok
        note = (f'оценка {result["stats"]["SCORE"]:+.4f} vs {best["stats"]["SCORE"]:+.4f}, '
                f'половины {result["halves"][0]["CAGR"]:+.1%}/{result["halves"][1]["CAGR"]:+.1%}'
                + ('' if halves_ok else ' (одна из половин хуже)'))
        return accepted, note

    def log_run(self, kind, tuner, correction, result, incumbent_score, accepted, seconds, note):
        stats = result['stats']
        self.append_data({
            'CREATED_AT': datetime.now(), 'KIND': kind,
            'PARAMS': json.dumps({'tuner': tuner, 'correction': correction}, sort_keys=True),
            'DATA_END': result['data_end'], 'BLIND_START': result['blind_start'],
            'CAGR': stats['CAGR'], 'MAX_DRAWDOWN': stats['MAX_DRAWDOWN'], 'SHARPE': stats['SHARPE'],
            'TRADES': stats['TRADES'], 'SCORE': stats['SCORE'],
            'H1_SCORE': result['halves'][0]['SCORE'], 'H2_SCORE': result['halves'][1]['SCORE'],
            'INCUMBENT_SCORE': incumbent_score, 'ACCEPTED': accepted, 'SECONDS': seconds,
            'NOTE': note[:500],
        }, tables.SelftuneRun)

    def tick(self, heavy: bool = False) -> dict:
        if self.ensure_market() or self.incumbent is None:
            return self.refresh()
        return self.trial(heavy)


def run_daemon(spec: str = SELFTUNE_WINDOWS):
    windows = parse_windows(spec)
    night = parse_windows(SELFTUNE_NIGHT + '/0-0')[0]
    tuner = SelfTuner()
    logger.info(f'Самокоррекция запущена, окна: {spec}')
    while True:
        now = datetime.now(ZoneInfo(TIMEZONE))
        window = current_window(now, windows)
        if window is None:
            SQLmain.save_activity(SERVICE, 'ожидание окна самокоррекции')
            sleep(min(seconds_to_next_window(now, windows), 300))
            continue
        started = monotonic()
        try:
            heavy = in_window(now.hour * 60 + now.minute, night[0], night[1])
            result = tuner.tick(heavy)
            SQLmain.save_activity(SERVICE, f'слепой CAGR {result["stats"]["CAGR"]:+.1%}')
        except Exception as error:
            logger.exception(error)
            tlg.send_message(f'Самокоррекция: ошибка - {error}')
        pause = tuner.rng.uniform(window[2], window[3]) - (monotonic() - started)
        sleep(max(5.0, pause))
