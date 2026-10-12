""" Движок сигналов для сделок на 1-2 дня (акции TQBR Мосбиржи).

Команды:
    python main.py update      — загрузить/обновить дневную историю с ISS
    python main.py backtest    — бэктест модели и бенчмарков (последние ~250 дней)
    python main.py signals     — сформировать сигналы по последнему закрытию
    python main.py statistic   — последние результаты бэктестов
    python main.py run         — демон: каждое утро буднего дня update + signals
"""
import argparse
import logging
from datetime import datetime
from time import sleep
from zoneinfo import ZoneInfo

from elements.algorithm.worker import Algorithm
from elements.backtest.worker import Backtest
from elements.data.loader import DataLoader
from elements.features.worker import build_dataset
from elements.notification.worker import TelegramNotification as tlg, signals_message
from source.settings.settings import handler, TIMEZONE, SIGNAL_TIME, LOOP_SLEEP


# Запуск логгера.
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

SERVICE = 'iss_filter_algorithm'


def msk_now() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


def run_signals(use_ai: bool = True):
    signals = Algorithm.generate_signals(use_ai=use_ai)
    tlg.send_message(signals_message(signals))
    return signals


def daemon():
    last_run = None
    tlg.send_message('Сервис стартовал.')
    while True:
        try:
            now = msk_now()
            Algorithm.save_activity(SERVICE, f'last signals run: {last_run}')
            current = now.hour + now.minute / 100
            if now.weekday() < 5 and current >= SIGNAL_TIME and last_run != now.date():
                loader = DataLoader()
                loader.update_history()
                loader.update_hour_history()
                loader.update_dividends()
                run_signals()
                last_run = now.date()
        except Exception as error:
            logger.exception(error)
            tlg.send_message(f'Получена ошибка - {error}')
        sleep(LOOP_SLEEP)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['update', 'backtest', 'signals', 'statistic', 'run'])
    parser.add_argument('--no-ai', action='store_true', help='не вызывать AI_Manager')
    parser.add_argument('--days', type=int, default=250, help='длина теста в торговых днях')
    args = parser.parse_args()

    Algorithm.create_all_tables()
    if args.command == 'update':
        loader = DataLoader()
        loader.update_history()
        loader.update_hour_history()
        loader.update_dividends()
    elif args.command == 'backtest':
        Backtest.run(build_dataset(Algorithm.load_candles()), test_days=args.days)
    elif args.command == 'signals':
        for signal in run_signals(use_ai=not args.no_ai):
            logger.info(signal)
    elif args.command == 'statistic':
        show_statistic()
    else:
        daemon()
