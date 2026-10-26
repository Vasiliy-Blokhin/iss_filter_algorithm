""" Движок сигналов для сделок на 1-2 дня (акции TQBR Мосбиржи).

Команды:
    python main.py update      — загрузить/обновить дневную историю с ISS
    python main.py backtest    — бэктест модели и бенчмарков (последние ~250 дней)
    python main.py signals     — сформировать сигналы по последнему закрытию
    python main.py statistic   — последние результаты бэктестов
    python main.py stats       — теневая статистика стратегий и выбор (--days N — история)
    python main.py run         — демон: каждое утро буднего дня update + signals,
                                 по будням в BACKTEST_TIME (23:56) — бэктест завершённого дня
    python main.py selftune    — демон самокоррекции (слепые бэктесты в окнах SELFTUNE_WINDOWS)
    python main.py blind       — слепой бэктест действующих параметров по текущий день
    python main.py rotation    — ротация фондов: обновить индексы, цель месяца и бэктест
    python main.py news        — собрать ленты и оценить новости локальной моделью
    python main.py reverse     — реверсивный алгоритм: шаг жизненного цикла, правила, журнал
                                 (--blind — слепой прогон с нуля, --reflect — разбор опыта ИИ)
    python main.py reverse-daemon — демон новостей и ночного разбора опыта (iss_reverse)
    python main.py evolve      — демон абстрактного алгоритма: случайный рекурсивный поиск геномов
                                 в окнах EVOLVE_WINDOWS по дням EVOLVE_DAYS (iss_evolve),
                                 ежедневно EVOLVE_REGIME_WINDOW (05:00–06:50) — только по режимам;
                                 --steps N — сразу N шагов без ожидания окна;
                                 --until 02:00 — этапы PHASES (новые гены, смешанный,
                                 доработка этапов) до 02:00 со сводкой после каждого
    python main.py stages      — зафиксированные этапы абстрактного алгоритма
    python main.py regime      — режим рынка сейчас (и история в таблице market_regime)
    python main.py pools       — пересобрать пулы стратегий по режимам рынка
    python main.py blind-full  — полностью слепой тест всей системы по периодам 1 мес. … 5 лет
                                 (--no-ga — без генетики, --periods 1y,5y — свои периоды)
    python main.py tasks       — демон задач из Telegram (кнопки «Запуск», iss_tasks)
    evolve --minutes N [--regimes] — поиск N минут: общий или общий + каждый режим рынка
"""
import argparse
import json
import logging
from datetime import datetime, timedelta
from time import sleep
from zoneinfo import ZoneInfo


from elements.algorithm.worker import Algorithm
from elements.backtest.worker import Backtest
from elements.data.loader import DataLoader
from elements.selector.worker import StrategySelector
from elements.rotation.worker import Rotation, backtest as rotation_backtest
from elements.selftune.params import apply_active_strategy_params, load_active
from elements.strategy.worker import STRATEGY_TITLES
from elements.notification.worker import TelegramNotification as tlg, signals_message
from source.settings.settings import (
    handler, TIMEZONE, SIGNAL_TIME, BACKTEST_TIME, LOOP_SLEEP, ROTATION_ENABLED,
)


# Запуск логгера.
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

SERVICE = 'iss_filter_algorithm'


def msk_now() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


def run_signals(use_ai: bool = True):
    signals = Algorithm.generate_signals(use_ai=use_ai)
    selection = Algorithm.run_sql_command(
        'SELECT * FROM strategy_selection ORDER BY "DATE" DESC LIMIT 1'
    )
    header = f'{Algorithm.regime_text}\n\n' if Algorithm.regime_text else \
        (f'Стратегия: {selection[0]["REASON"]}\n\n' if selection else '')
    tlg.send_message(header + signals_message(signals))
    return signals


def show_strategy_stats(days: int):
    apply_active_strategy_params()
    panel = Algorithm.build_panel(Algorithm.complete_candles())
    if days > 1:
        StrategySelector.backfill(panel, days)
    allocation, stats = StrategySelector.select_strategy(panel)
    table = stats.pivot(index='STRATEGY', columns='PERIOD', values='RETURN')
    trades = stats.pivot(index='STRATEGY', columns='PERIOD', values='TRADES')
    for name in table.sort_values('week', ascending=False).index:
        logger.info(
            f'{STRATEGY_TITLES[name]:34s} день {table.loc[name, "day"]:+.2%} ({trades.loc[name, "day"]}), '
            f'неделя {table.loc[name, "week"]:+.2%} ({trades.loc[name, "week"]}), '
            f'месяц {table.loc[name, "month"]:+.2%} ({trades.loc[name, "month"]})'
            + (f'  <- доля {allocation[name]:.0%}' if name in allocation else '')
        )


def run_backtest(days: int = 250, update: bool = True) -> dict:
    """ Бэктест всех стратегий, геномного и реверсивного по последнему завершённому дню."""
    if update:
        loader = DataLoader()
        loader.update_history()
        loader.update_dividends()
    apply_active_strategy_params()
    panel = Algorithm.build_panel(Algorithm.complete_candles())
    results = Backtest.run(panel, test_days=days)
    best = sorted(results.items(), key=lambda item: -item[1]['stats']['SHARPE'])[:3]
    logger.info(
        f'Бэктест по {panel["DATE"].max().date()}: '
        + ', '.join(f'{name} Sharpe {result["stats"]["SHARPE"]:.2f}' for name, result in best)
    )
    return {'panel': panel, 'results': results}


def last_done(table: str, shift_hours: float = 0):
    """ День последнего прогона по CREATED_AT таблицы (после перезапуска не повторять его)."""
    try:
        moment = Algorithm.run_sql_command(f'SELECT MAX(CREATED_AT) AS t FROM {table}')[0]['t']
    except Exception:
        return None
    return (datetime.fromisoformat(str(moment)) - timedelta(hours=shift_hours)).date() if moment else None


def daemon():
    last_run = last_done('strategy_selection')
    # Бэктест стартует в BACKTEST_TIME (23:56) и пишет строки уже после полуночи.
    last_backtest = last_done('backtest_runs', shift_hours=1)
    tlg.send_message('Сервис стартовал.')
    while True:
        try:
            now = msk_now()
            Algorithm.save_activity(SERVICE, f'last signals run: {last_run}, backtest: {last_backtest}')
            current = now.hour + now.minute / 100
            # Сразу после вечерней сессии (свеча дня уже завершена — см. complete_candles).
            if now.weekday() < 5 and current >= BACKTEST_TIME and last_backtest != now.date():
                last_backtest = now.date()
                run_backtest()
            if now.weekday() < 5 and current >= SIGNAL_TIME and last_run != now.date():
                loader = DataLoader()
                loader.update_history()
                loader.update_hour_history()
                loader.update_dividends()
                run_signals()
                if ROTATION_ENABLED:
                    run_rotation(now.date())
                last_run = now.date()
        except Exception as error:
            logger.exception(error)
            tlg.send_message(f'Получена ошибка - {error}')
        sleep(LOOP_SLEEP)


def rotation_cap() -> tuple[float, bool]:
    """ Доля акций ротации: по режиму рынка, если слепой тест включил это (regime_switch),
    иначе прежняя постоянная. Второе значение — режим сменился сегодня (пересчитать цель сразу)."""
    from elements.selftune.params import load_active
    from source.settings.settings import ROTATION_EQ_CAP, REGIME_ENABLED
    if not (REGIME_ENABLED and load_active('regime_switch').get('rotation')):
        return ROTATION_EQ_CAP, False
    import pandas as pd
    from elements.rotation.worker import select_caps
    rows = Algorithm.run_sql_command('SELECT "DATE", REGIME FROM market_regime ORDER BY "DATE"')
    if not rows:
        return ROTATION_EQ_CAP, False
    regimes = pd.Series([r['REGIME'] for r in rows], index=pd.to_datetime([r['DATE'] for r in rows]))
    caps = select_caps(Rotation.load_prices(), regimes, regimes.index[-1])
    changed = len(regimes) > 1 and regimes.iloc[-1] != regimes.iloc[-2]
    return caps[regimes.iloc[-1]], changed


def run_rotation(today=None):
    Rotation.update_indexes()
    from elements.selftune.params import load_active
    from source.settings.settings import REGIME_ENABLED
    if REGIME_ENABLED and load_active('regime_switch').get('rotation'):
        # слепой тест включил режимное распределение фондов вместо прежней ротации
        from elements.regime import assets
        from elements.regime.allocation import update_live_target
        assets.update()
        result = update_live_target(today)
        if result:
            tlg.send_message('Режимное распределение (данные на {}): '.format(result['date'])
                             + ', '.join(f'{secid} {weight:.0%}' for secid, weight in result['target'].items())
                             + '\n' + result['reason'])
        run_dividend()
        return result
    cap, changed = rotation_cap()
    result = Rotation.update_target(today, cap, now=changed)
    if result:
        tlg.send_message(
            f'Ротация на месяц (данные на {result["date"]}): '
            + ', '.join(f'{secid} {weight:.0%}' for secid, weight in result['target'].items())
            + '\n' + result['reason']
        )
    run_dividend()
    return result


def run_dividend():
    """ Набор бумаг дивидендного разгона на следующий торговый день (исполнитель берёт их
    вместо части фонда акций ротации)."""
    from source.settings.settings import DIVIDEND_ENABLED
    if not DIVIDEND_ENABLED:
        return None
    from elements.dividend.worker import DividendOverlay, describe
    result = DividendOverlay.update()
    text = describe(result)
    if text:
        tlg.send_message(text)
    return result


def show_rotation():
    run_rotation(msk_now().date())
    for row in Rotation.current_target():
        logger.info(f'Цель ротации (данные на {row["DATE"]}): {row["ASSET"]} {row["SECID"]} {row["WEIGHT"]:.0%}')
    stats = rotation_backtest(Rotation.load_prices())
    logger.info(
        f'Бэктест ротации с 2021: {stats["CAGR"]:+.1%} годовых, просадка {stats["MAX_DRAWDOWN"]:.1%}, '
        f'худшие 12 мес. {stats["WORST_12M"]:+.1%}; по годам: '
        + ', '.join(f'{year} {value:+.1%}' for year, value in stats['YEARLY'].items())
    )


def show_news():
    from elements.reverse.news import NewsCollector, NewsScorer
    NewsCollector.collect()
    NewsScorer().run(limit=args.days if args.days != 250 else 300)


def show_reverse():
    from elements.reverse.engine import describe
    from elements.reverse.worker import ReverseAlgorithm
    panel = Algorithm.build_panel(Algorithm.complete_candles())
    if args.blind:
        result = ReverseAlgorithm.blind(panel)
        stats = result['stats']
        logger.info(
            f'Слепой прогон реверсивного {result["start"]}..{result["end"]}: CAGR {stats["CAGR"]:+.1%}, '
            f'итог {stats["TOTAL_RETURN"]:+.1%}, просадка {stats["MAX_DRAWDOWN"]:.1%}, Sharpe {stats["SHARPE"]:.2f}, '
            f'сделок {stats["TRADES"]} (средняя {stats["AVG_TRADE"]:+.2%}), загрузка {stats["EXPOSURE"]:.0%}; '
            f'все сигналы, вкл. теневые: {result["signals"]} (средняя {result["signal_avg"]:+.2%}); '
            f'половины {result["halves"][0]["CAGR"]:+.1%} / {result["halves"][1]["CAGR"]:+.1%}; по годам: '
            + ', '.join(f'{year} {value:+.1%}' for year, value in result['yearly'].items())
        )
        return
    if args.reflect:
        from elements.reverse.mind import reflect
        logger.info(reflect(panel))
        return
    picks, result = ReverseAlgorithm.signals(panel)
    logger.info(result['summary'])
    for pick in picks:
        logger.info(f'{pick["ROW"]["SECID"]}: оценка {pick["SCORE"]:+.2%}, доля {pick["WEIGHT"]:.1%} — {pick["LOGIC"]}')
    for rule in ReverseAlgorithm.rules('live'):
        logger.info(f'живое [{rule["ORIGIN"]}] с {rule["BORN"]}: {describe(json.loads(rule["ATOMS"]))}; '
                    f'вне выборки {rule["OOS_N"]} дн., средняя {rule["OOS_MEAN"]:+.2%}')
    for row in reversed(ReverseAlgorithm.journal(15)):
        logger.info(f'{row["DATE"]} {row["KIND"]}: {row["TEXT"]}')


def run_evolve(steps: int, until: str = '', minutes: int = 0, regimes: bool = False):
    from elements.evolve.worker import Evolver, run_daemon as run_evolve_daemon, run_phases, run_groups, search_busy
    if minutes:
        if search_busy():
            say('Геномный поиск уже идёт — запуск отменён.')
            return
        run_groups(msk_now() + timedelta(minutes=minutes), None if regimes else [''])
        return
    if until:
        if search_busy():
            logger.warning('Геномный поиск уже идёт в другом процессе (data/evolve.lock) — запуск отменён.')
            return
        # Поэтапный прогон до ЧЧ:ММ МСК (если время уже прошло сегодня — до завтрашнего).
        now = msk_now()
        hour, minute = (int(part) for part in until.split(':'))
        moment = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if moment <= now:
            moment += timedelta(days=1)
        run_phases(moment)
        return
    if not steps:
        run_evolve_daemon()
        return
    evolver = Evolver()
    evolver.ensure_market()
    for _ in range(steps):
        evolver.step()
    logger.info(evolver.session_report())


def show_stages():
    from elements.evolve.worker import Evolver
    stages = Evolver.stages(active=False)
    if not stages:
        logger.info('Этапов абстрактного алгоритма пока нет.')
    for row in stages:
        logger.info(
            f'этап {row["id"]} {"действует" if row["ACTIVE"] else "снят"}'
            + (f', от этапа {row["PARENT_ID"]}' if row['PARENT_ID'] else '')
            + f' ({row["OPERATOR"]}): поиск CAGR {row["CAGR"]:+.1%}, просадка {row["MAX_DRAWDOWN"]:.1%}, '
            f'сделок {row["TRADES"]}; контроль с {row["HOLDOUT_START"]} CAGR {row["HOLD_CAGR"]:+.1%}; '
            + (f'слепая с {row["BLIND_START"]} {row["BLIND_RETURN"]:+.1%} ({row["BLIND_TRADES"]} сд.); '
               if row['BLIND_RETURN'] is not None else '')
            + f'{row["NOTE"]}'
        )


def say(text: str):
    """ Короткий итог для бота: строки с ИТОГ: демон задач пересылает в Telegram."""
    for line in text.splitlines():
        print(f'ИТОГ: {line}', flush=True)


def regime_panel():
    from elements.regime.worker import MarketRegime
    apply_active_strategy_params()
    panel = Algorithm.build_panel(Algorithm.complete_candles())
    return panel, MarketRegime.history(panel)


def show_regime():
    from elements.regime.worker import MarketRegime
    panel, _ = regime_panel()
    say(MarketRegime.update(panel)['text'])


def run_pools():
    from elements.regime.pools import RegimePools, pools_text
    panel, history = regime_panel()
    say('Пулы стратегий по режимам:\n' + pools_text(RegimePools.rebuild(panel, history, notify=False)))


def run_blind_full(no_ga: bool, periods: str):
    from elements.regime.blind import FullBlind, summary_lines, decide_switches
    from elements.selftune.params import save_active
    from source.settings.settings import BLIND_PERIODS, BLIND_GA_STEPS
    from elements.regime import assets
    assets.update()
    panel, history = regime_panel()
    blind = FullBlind(panel, history, ga_steps=0 if no_ga else BLIND_GA_STEPS, periods=periods or BLIND_PERIODS)
    result = blind.run(progress=lambda done, total: Algorithm.save_activity(
        'blind_full', f'переобучение {done}/{total}'))
    Algorithm.save_activity('blind_full', 'готово')
    switches, note = decide_switches(result)
    blind.save(result, note=('без генетики; ' if no_ga else '') + note)
    save_active('regime_switch', switches, data_end=result['periods'][0]['END'], note=note)
    from elements.regime.allocation import save_live
    save_live(result.get('allocation_log', []), data_end=result['periods'][0]['END'])
    longest = result['periods'][-1]
    say('Слепой тест (счёт 50 тыс. ₽, комиссии АЛОР, лоты' + (', без генетики' if no_ga else '') + '):\n'
        + '\n'.join(summary_lines(result))
        + f'\nКраткосрочные сделки за {longest["PERIOD"]}: {longest["SHORT_RETURN"]:+.1%} '
        f'({longest["SHORT_TRADES"]} сд.), ротация {longest["ROTATION_RETURN"]:+.1%} '
        f'(прежняя {longest["OLD_ROTATION_RETURN"]:+.1%}).'
        f'\nДивидендный разгон за {longest["PERIOD"]}: система {longest["NODIV_RETURN"]:+.1%} → {longest["TOTAL_RETURN"]:+.1%}, '
        f'ротация {longest["ROTATION_RETURN"]:+.1%} → {longest["ROTATION_DIV_RETURN"]:+.1%}, '
        f'прежняя {longest["OLD_ROTATION_RETURN"]:+.1%} → {longest["OLD_ROTATION_DIV_RETURN"]:+.1%}.'
        f'\nРешение: {note}.')


def show_statistic():
    rows = Algorithm.run_sql_command(
        'SELECT * FROM backtest_runs ORDER BY id DESC LIMIT 10'
    )
    for row in rows:
        logger.info(
            f'{row["CREATED_AT"]} {row["STRATEGY"]}: {row["PERIOD_START"]}..{row["PERIOD_END"]}, '
            f'сделок {row["TRADES"]}, win {row["WIN_RATE"]:.1%}, средняя {row["AVG_TRADE"]:.2%}, '
            f'итог {row["TOTAL_RETURN"]:.1%}, просадка {row["MAX_DRAWDOWN"]:.1%}, '
            f'Sharpe {row["SHARPE"]:.2f}'
        )


def show_blind():
    from elements.selftune.blind import BlindBacktest
    tuner, correction = load_active('tuner'), load_active('correction')
    blind = BlindBacktest(Algorithm.build_panel(Algorithm.complete_candles()))
    result = blind.run(tuner, correction)
    stats = result['stats']
    logger.info(
        f'Слепой бэктест {result["blind_start"]}..{result["data_end"]}: '
        f'CAGR {stats["CAGR"]:+.1%}, итог {stats["TOTAL_RETURN"]:+.1%}, '
        f'просадка {stats["MAX_DRAWDOWN"]:.1%}, Sharpe {stats["SHARPE"]:.2f}, '
        f'сделок {stats["TRADES"]}, средняя загрузка капитала {stats["EXPOSURE"]:.0%}; '
        f'половины: {result["halves"][0]["CAGR"]:+.1%} / {result["halves"][1]["CAGR"]:+.1%}'
    )
    logger.info(f'tuner: {tuner}; correction: {correction}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=[
        'update', 'backtest', 'signals', 'statistic', 'stats', 'run', 'selftune', 'blind', 'rotation',
        'news', 'reverse', 'reverse-daemon', 'evolve', 'stages', 'regime', 'pools', 'blind-full', 'tasks',
    ])
    parser.add_argument('--minutes', type=int, default=0, help='evolve: поиск N минут')
    parser.add_argument('--regimes', action='store_true', help='evolve: общий + каждый режим рынка')
    parser.add_argument('--no-ga', dest='no_ga', action='store_true', help='blind-full: без генетики')
    parser.add_argument('--periods', default='', help='blind-full: периоды, например 1m,1y,5y')
    parser.add_argument('--steps', type=int, default=0, help='evolve: выполнить N шагов сразу')
    parser.add_argument('--until', default='', help='evolve: поэтапный прогон до ЧЧ:ММ МСК (PHASES)')
    parser.add_argument('--no-ai', action='store_true', help='не вызывать AI_Manager')
    parser.add_argument('--blind', action='store_true', help='reverse: слепой прогон с нуля')
    parser.add_argument('--reflect', action='store_true', help='reverse: разбор опыта локальной моделью')
    parser.add_argument('--days', type=int, default=250, help='длина теста / истории статистики в торговых днях')
    args = parser.parse_args()

    Algorithm.create_all_tables()
    if args.command == 'update':
        loader = DataLoader()
        loader.update_history()
        loader.update_hour_history()
        loader.update_dividends()
        Rotation.update_indexes()
    elif args.command == 'backtest':
        panel = run_backtest(args.days, update=False)['panel']
        result = StrategySelector.evaluate(panel, days=args.days)
        for name, stats in result.items():
            logger.info(
                f'{name:15s} сделок {stats["TRADES"]}, win {stats["WIN_RATE"]:.1%}, '
                f'средняя {stats["AVG_TRADE"]:+.2%}, итог {stats["TOTAL_RETURN"]:+.1%}, '
                f'просадка {stats["MAX_DRAWDOWN"]:.1%}, Sharpe {stats["SHARPE"]:.2f}'
            )
    elif args.command == 'signals':
        for signal in run_signals(use_ai=not args.no_ai):
            logger.info(signal)
    elif args.command == 'statistic':
        show_statistic()
    elif args.command == 'stats':
        show_strategy_stats(args.days if args.days != 250 else 1)
    elif args.command == 'selftune':
        from elements.selftune.worker import run_daemon
        run_daemon()
    elif args.command == 'blind':
        show_blind()
    elif args.command == 'rotation':
        show_rotation()
    elif args.command == 'news':
        show_news()
    elif args.command == 'reverse':
        show_reverse()
    elif args.command == 'reverse-daemon':
        from elements.reverse.daemon import run_daemon as run_reverse_daemon
        run_reverse_daemon()
    elif args.command == 'evolve':
        run_evolve(args.steps, args.until, args.minutes, args.regimes)
    elif args.command == 'regime':
        show_regime()
    elif args.command == 'pools':
        run_pools()
    elif args.command == 'blind-full':
        run_blind_full(args.no_ga, args.periods)
    elif args.command == 'tasks':
        from elements.tasks.worker import run_daemon as run_tasks_daemon
        run_tasks_daemon()
    elif args.command == 'stages':
        show_stages()
    else:
        daemon()
