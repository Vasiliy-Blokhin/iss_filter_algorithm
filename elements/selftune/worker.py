""" Демон самокоррекции: в заданные окна времени проверяет кандидатов слепым бэктестом.

Каждая проверка (тик):
  1. если появились новые данные (или демон только запущен) — «refresh»: слепой бэктест
     действующих параметров по текущий день и пересчёт живых порогов стратегий
     (подбор на окне обучения, заканчивающемся последним днём данных) -> algo_params;
  2. иначе — один кандидат: мутация параметров коррекции (дёшево, секунды) или, в ночном
     окне, параметров переподбора (tuner, десятки секунд). Кандидат принимается, только если
     слепая оценка выше действующей на SELFTUNE_MARGIN и ни одна половина слепого периода
     не хуже действующей больше чем на SELFTUNE_HALF_TOLERANCE.
В воскресенье в окнах SELFTUNE_SUNDAY — рекурсивный поиск по всем алгоритмам (correction,
tuner -> пороги стратегий, параметры реверсивного): перебираются все соседние значения сетки
действующих параметров; принятый кандидат сохраняется и становится новой точкой, от которой
поиск начинается заново, пока окрестность не исчерпана (локальный оптимум — тогда случайные
прыжки на 2-3 параметра). Правило принятия то же.
Все проверки пишутся в selftune_runs; в Telegram — одна сводка в сутки (SELFTUNE_REPORT_TIME).
"""
import json
import logging
import random

import sqlalchemy as sa
from datetime import datetime, timedelta
from time import sleep, monotonic
from zoneinfo import ZoneInfo

from elements.algorithm.worker import Algorithm
from elements.notification.worker import TelegramNotification as tlg
from elements.selftune.blind import BlindBacktest, SegmentStore
from elements.reverse.engine import REVERSE_GRID
from elements.selftune.params import (
    CORRECTION_GRID, TUNER_GRID, load_active, load_active_record, save_active,
    update_active_metrics, params_key,
)
from source.settings.settings import (
    handler,
    TIMEZONE,
    SELFTUNE_WINDOWS,
    SELFTUNE_NIGHT,
    SELFTUNE_SUNDAY,
    SELFTUNE_REPORT_TIME,
    SELFTUNE_MARGIN,
    SELFTUNE_HALF_TOLERANCE,
    TARGET_CAGR,
)
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

SERVICE = 'selftune'
TUNER_SHARE = 0.35  # доля «тяжёлых» кандидатов (tuner) в ночном окне
SUNDAY = 6
SUNDAY_GRIDS = {'correction': CORRECTION_GRID, 'tuner': TUNER_GRID, 'reverse': REVERSE_GRID}
SUNDAY_JUMPS = 3  # случайных прыжков на алгоритм, когда окрестность исчерпана


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


def neighbours(params: dict, grid: dict) -> list[dict]:
    """ Все точки на шаг сетки от params (по одному параметру); вне сетки — все значения."""
    result = []
    for key, values in grid.items():
        if params.get(key) in values:
            index = values.index(params[key])
            options = [values[i] for i in (index - 1, index + 1) if 0 <= i < len(values)]
        else:
            options = list(values)
        result.extend({**params, key: value} for value in options)
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
        self.panel = None
        self.reverse: dict | None = None  # {'params', 'result', 'market'} — для воскресного поиска
        self.queue: list[tuple[str, dict]] = []

    # --- данные ---------------------------------------------------------------
    def ensure_market(self) -> bool:
        """ Пересобрать панель, если в БД появились новые свечи. True — данные обновились."""
        last = self.run_sql_command('SELECT MAX("DATE") AS last FROM candles')[0]['last']
        if self.blind is not None and last == self.loaded_end:
            return False
        self.panel = Algorithm.build_panel(Algorithm.complete_candles())
        self.blind = BlindBacktest(self.panel, self.store)
        self.loaded_end = last
        self.tried.clear()
        self.reverse = None
        self.queue = []
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
        self.log_run('refresh', {'tuner': tuner, 'correction': correction}, result, None, True,
                     monotonic() - started, f'живые пороги по {data_end}')
        logger.info(f'Самокоррекция refresh: слепой результат {describe(stats)} '
                    f'(цель {TARGET_CAGR:.0%} годовых)')
        return result

    def trial(self, heavy: bool) -> dict:
        kind = 'tuner' if heavy and self.rng.random() < TUNER_SHARE else 'correction'
        for _ in range(20):
            candidate = mutate(self.current(kind), SUNDAY_GRIDS[kind], self.rng)
            if self.key(kind, candidate) not in self.tried:
                break
        return self.check(kind, candidate)[0]

    # --- общая проверка кандидата -----------------------------------------------
    def current(self, kind: str) -> dict:
        return self.reverse['params'] if kind == 'reverse' else self.incumbent[kind]

    def combined(self, kind: str, candidate: dict) -> dict:
        """ Полный набор параметров, который проверяется слепым бэктестом."""
        if kind == 'reverse':
            return {'reverse': candidate}
        return {'tuner': self.incumbent['tuner'], 'correction': self.incumbent['correction'],
                kind: candidate}

    def key(self, kind: str, candidate: dict) -> str:
        return params_key(self.combined(kind, candidate))

    def ensure_reverse(self):
        """ Слепой результат действующих параметров реверсивного (точка отсчёта поиска)."""
        from elements.reverse.mind import reverse_blind
        from elements.reverse.worker import ReverseAlgorithm
        params = load_active('reverse')
        if self.reverse is not None and self.reverse['params'] == params:
            return
        started = monotonic()
        market = self.reverse['market'] if self.reverse else ReverseAlgorithm.market(self.panel)
        result = reverse_blind(market, params)
        self.reverse = {'params': params, 'result': result, 'market': market}
        self.log_run('reverse_refresh', {'reverse': params}, result, None, True,
                     monotonic() - started, f'реверсивный, слепо по {result["data_end"]}')

    def check(self, kind: str, candidate: dict) -> tuple[dict, bool]:
        """ Слепой бэктест кандидата; при положительном результате — сохранить в algo_params."""
        started = monotonic()
        self.tried.add(self.key(kind, candidate))
        if kind == 'reverse':
            from elements.reverse.mind import reverse_blind
            result = reverse_blind(self.reverse['market'], candidate)
            best = self.reverse['result']
        else:
            params = self.combined(kind, candidate)
            result = self.blind.run(params['tuner'], params['correction'])
            best = self.incumbent['result']
        accepted, note = self.accept(result, best)
        self.log_run(kind, self.combined(kind, candidate), result, best['stats']['SCORE'], accepted,
                     monotonic() - started, note)
        if not accepted:
            return result, False
        save_active(kind, candidate, result['stats'], result['data_end'], note)
        if kind == 'reverse':
            self.reverse = {**self.reverse, 'params': candidate, 'result': result}
        else:
            if kind == 'tuner':
                live = self.blind.live_params(candidate)
                save_active('strategy', live, result['stats'], result['data_end'],
                            'переподбор после смены tuner')
            self.incumbent = {**self.incumbent, kind: candidate, 'result': result}
        logger.info(f'Самокоррекция: приняты новые параметры ({kind}). Слепой бэктест '
                    f'{result["blind_start"]}..{result["data_end"]}: {describe(result["stats"])}; '
                    f'было CAGR {best["stats"]["CAGR"]:+.1%}.')
        return result, True

    # --- воскресный рекурсивный поиск -------------------------------------------
    def sunday_queue(self, first: str | None = None) -> list[tuple[str, dict]]:
        """ Непроверенные соседи действующих параметров всех алгоритмов (сначала first);
        если окрестность исчерпана — случайные прыжки от действующей точки."""
        kinds = sorted(SUNDAY_GRIDS, key=lambda kind: kind != first)
        queue = []
        for kind in kinds:
            options = [p for p in neighbours(self.current(kind), SUNDAY_GRIDS[kind])
                       if self.key(kind, p) not in self.tried]
            self.rng.shuffle(options)
            queue += [(kind, p) for p in options]
        if not queue:
            for kind in kinds:
                for _ in range(SUNDAY_JUMPS):
                    jump = self.current(kind)
                    for _ in range(self.rng.choice([2, 3])):
                        jump = mutate(jump, SUNDAY_GRIDS[kind], self.rng)
                    if self.key(kind, jump) not in self.tried:
                        queue.append((kind, jump))
        return queue

    def sunday_tick(self) -> dict:
        if self.ensure_market() or self.incumbent is None:
            return self.refresh()
        self.ensure_reverse()
        if not self.queue:
            self.queue = self.sunday_queue()
        if not self.queue:
            return self.incumbent['result']
        kind, candidate = self.queue.pop(0)
        result, accepted = self.check(kind, candidate)
        if accepted:  # рекурсия: поиск заново от новой точки, начиная с того же алгоритма
            self.queue = self.sunday_queue(first=kind)
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

    def log_run(self, kind, params, result, incumbent_score, accepted, seconds, note):
        stats = result['stats']
        self.append_data({
            'CREATED_AT': datetime.now(), 'KIND': kind,
            'PARAMS': json.dumps(params, sort_keys=True),
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


def get_state(key: str) -> str | None:
    with main_engine.begin() as connection:
        connection.execute(sa.text('CREATE TABLE IF NOT EXISTS bot_state (KEY TEXT PRIMARY KEY, VALUE TEXT)'))
        return connection.execute(sa.text('SELECT VALUE FROM bot_state WHERE KEY = :key'),
                                  {'key': key}).scalar()


def set_state(key: str, value: str):
    with main_engine.begin() as connection:
        connection.execute(sa.text(
            'INSERT INTO bot_state VALUES (:key, :value) '
            'ON CONFLICT(KEY) DO UPDATE SET VALUE = excluded.VALUE'), {'key': key, 'value': value})


def daily_report(since: datetime, errors: list[str]) -> str:
    """ Сводка проверок за период: сколько кандидатов, что принято, текущий слепой результат."""
    runs = SQLmain.run_sql_command(
        'SELECT * FROM selftune_runs WHERE CREATED_AT > :since ORDER BY CREATED_AT', {'since': since})
    trials = [run for run in runs if not run['KIND'].endswith('refresh')]
    accepted = [run for run in trials if run['ACCEPTED']]
    lines = [f'Самокоррекция за сутки: проверено кандидатов {len(trials)}, принято {len(accepted)}.']
    for run in accepted:
        lines.append(f'✅ {run["KIND"]}: CAGR {run["CAGR"]:+.1%}, просадка {run["MAX_DRAWDOWN"]:.1%}, '
                     f'Sharpe {run["SHARPE"]:.2f} ({run["NOTE"]})')
    current = [run for run in runs if run['KIND'] == 'refresh'
               or (run['ACCEPTED'] and run['KIND'] in ('tuner', 'correction'))]
    if current:
        best = current[-1]
        lines.append(f'Действующие параметры: слепой бэктест {best["BLIND_START"]}..{best["DATA_END"]}: '
                     f'CAGR {best["CAGR"]:+.1%}, просадка {best["MAX_DRAWDOWN"]:.1%}, '
                     f'Sharpe {best["SHARPE"]:.2f}, сделок {best["TRADES"]} (цель {TARGET_CAGR:.0%}).')
    if errors:
        lines.append(f'Ошибок: {len(errors)}, последняя: {errors[-1]}')
    return '\n'.join(lines)


def maybe_report(now: datetime, errors: list[str]):
    """ Раз в сутки после SELFTUNE_REPORT_TIME отправить сводку."""
    today = now.date().isoformat()
    if now.hour + now.minute / 100 < SELFTUNE_REPORT_TIME or get_state('selftune_report') == today:
        return
    since = get_state('selftune_report_at')
    since = datetime.fromisoformat(since) if since else datetime.now() - timedelta(days=1)
    tlg.send_message(daily_report(since, errors))
    errors.clear()
    set_state('selftune_report', today)
    set_state('selftune_report_at', datetime.now().isoformat(sep=' '))


def pick_window(now: datetime, windows: list, sunday: list) -> tuple[tuple | None, bool]:
    """ Активное окно и признак воскресного рекурсивного поиска (воскресные окна важнее)."""
    if now.weekday() == SUNDAY:
        window = current_window(now, sunday)
        if window is not None:
            return window, True
    return current_window(now, windows), False


def run_daemon(spec: str = SELFTUNE_WINDOWS, sunday_spec: str = SELFTUNE_SUNDAY):
    windows = parse_windows(spec)
    sunday = parse_windows(sunday_spec)
    night = parse_windows(SELFTUNE_NIGHT + '/0-0')[0]
    tuner = SelfTuner()
    errors: list[str] = []
    logger.info(f'Самокоррекция запущена, окна: {spec}; воскресенье: {sunday_spec}')
    while True:
        now = datetime.now(ZoneInfo(TIMEZONE))
        try:
            maybe_report(now, errors)
        except Exception as error:
            logger.exception(error)
        window, recursive = pick_window(now, windows, sunday)
        if window is None:
            SQLmain.save_activity(SERVICE, 'ожидание окна самокоррекции')
            waits = windows + (sunday if now.weekday() == SUNDAY else [])
            sleep(min(seconds_to_next_window(now, waits), 300))
            continue
        started = monotonic()
        try:
            if recursive:
                result = tuner.sunday_tick()
            else:
                heavy = in_window(now.hour * 60 + now.minute, night[0], night[1])
                result = tuner.tick(heavy)
            SQLmain.save_activity(SERVICE, ('воскресный поиск, ' if recursive else '')
                                  + f'слепой CAGR {result["stats"]["CAGR"]:+.1%}')
        except Exception as error:
            logger.exception(error)
            errors.append(str(error)[:300])
        pause = tuner.rng.uniform(window[2], window[3]) - (monotonic() - started)
        sleep(max(5.0, pause))
