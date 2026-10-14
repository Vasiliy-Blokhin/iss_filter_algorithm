""" Демон реверсивного алгоритма (`python main.py reverse-daemon`, в start_all.bat — iss_reverse).

    - каждые REVERSE_NEWS_EVERY секунд: сбор лент и оценка новых новостей локальной моделью
      (REVERSE_NEWS_MODEL), сначала свежие; к REVERSE_SCORE_DEADLINE (до сигналов) новости
      должны быть оценены — сигналы 06:30 учитывают фон;
    - раз в сутки в ночном окне REVERSE_NIGHT: разбор опыта моделью с рассуждением
      (REVERSE_MIND_MODEL) — нарратив, новые гипотезы, предложения параметров со слепой
      проверкой; итог — в Telegram и в reverse_journal.
Сам шаг жизненного цикла (правила, сигналы) выполняется в утреннем запуске сигналов.
"""
import logging
from datetime import datetime, timedelta
from time import sleep, monotonic
from zoneinfo import ZoneInfo

from elements.notification.worker import TelegramNotification as tlg
from elements.reverse.news import NewsCollector, NewsScorer
from elements.selftune.worker import parse_windows, current_window
from source.settings.settings import (
    handler, TIMEZONE, REVERSE_NEWS_EVERY, REVERSE_NIGHT, REVERSE_SCORE_DEADLINE, REVERSE_MIND_MODEL,
)
from source.sql.main import SQLmain


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

SERVICE = 'reverse'


def msk_now() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)


def deadline_for(now: datetime) -> datetime | None:
    """ До утренних сигналов не начинать новую пачку оценки позже REVERSE_SCORE_DEADLINE."""
    hours = int(REVERSE_SCORE_DEADLINE)
    moment = now.replace(hour=hours, minute=round((REVERSE_SCORE_DEADLINE - hours) * 100), second=0)
    return moment if now < moment and now >= moment - timedelta(hours=3) else None


def reflected(last_day) -> bool:
    rows = SQLmain.run_sql_command(
        'SELECT COUNT(*) AS N FROM reverse_journal WHERE KIND = \'reflection\' AND "DATE" = :day',
        {'day': last_day})
    return bool(rows and rows[0]['N'])


def night_reflection():
    from elements.algorithm.worker import Algorithm
    from elements.reverse.mind import reflect
    from elements.selftune.params import load_active
    last_day = load_active('reverse_state').get('last_day')
    if not last_day or reflected(last_day):
        return
    logger.info(f'Разбор опыта моделью {REVERSE_MIND_MODEL} (день {last_day})')
    result = reflect(Algorithm.build_panel(Algorithm.complete_candles()))
    if 'error' in result:
        logger.warning(result['error'])
        return
    tlg.send_message(
        f'Реверсивный, разбор опыта ({REVERSE_MIND_MODEL}):\n{result["narrative"]}\n'
        + (f'Новые гипотезы: {"; ".join(result["hypotheses"])}\n' if result['hypotheses'] else '')
        + f'Параметры {result["params"]}: {result["check"]} — '
        + ('приняты' if result['accepted'] else 'отклонены')
    )


def run_daemon():
    night = parse_windows(f'{REVERSE_NIGHT}/60-60')
    last_news = -float('inf')
    tlg.send_message('Реверсивный алгоритм: демон новостей и разбора опыта стартовал.')
    while True:
        try:
            now = msk_now()
            if monotonic() - last_news >= REVERSE_NEWS_EVERY:
                last_news = monotonic()
                NewsCollector.collect()
                scored = NewsScorer().run(limit=300, deadline=deadline_for(now) or now + timedelta(minutes=50))
                SQLmain.save_activity(SERVICE, f'news scored {scored} at {now:%H:%M}')
            if current_window(now, night):
                night_reflection()
        except Exception as error:
            logger.exception(error)
            tlg.send_message(f'Реверсивный: ошибка — {error}')
        sleep(60)
