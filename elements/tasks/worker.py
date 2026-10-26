import logging, subprocess, sys
from datetime import datetime
from pathlib import Path
from time import sleep
import sqlalchemy as sa
from sqlalchemy.orm import Session
from elements.notification.worker import TelegramNotification as tlg
from source.settings.settings import handler, TASKS_POLL
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables

logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

SERVICE = 'iss_tasks'
ROOT = Path(__file__).resolve().parents[2]   # folder with main.py
RESULT_PREFIX = 'ИТОГ:'   # lines of child output starting with this are the short result

TASKS = {  # kind: (title, list of args to `python main.py`, timeout seconds)
    'evolve': ('Геномный поиск, 1 ч', ['evolve', '--minutes', '60'], 2 * 3600),
    'evolve_regimes': ('Геномный поиск по режимам, 1 ч', ['evolve', '--minutes', '60', '--regimes'], 2 * 3600),
    'blind': ('Слепой тест 1 мес. – 5 лет', ['blind-full'], 4 * 3600),
    'blind_fast': ('Слепой тест без генетики по акциям', ['blind-full', '--no-ga'], 3600),
    'pools': ('Пересборка пулов по режимам', ['pools'], 3600),
    'regime': ('Режим рынка', ['regime'], 600),
    'backtest': ('Бэктест стратегий', ['backtest'], 3600),
    'update': ('Обновление данных', ['update'], 1800),
}

def set_status(task_id: int, **values):
    """Обновить статус задачи."""
    with Session(bind=main_engine) as session:
        stmt = sa.update(tables.TaskRequest).where(tables.TaskRequest.id == task_id).values(**values)
        session.execute(stmt)
        session.commit()

def next_task() -> dict | None:
    """Получить следующую задачу."""
    rows = SQLmain.run_sql_command(
        "SELECT * FROM task_requests WHERE STATUS = 'new' ORDER BY id LIMIT 1"
    )
    return dict(rows[0]) if rows else None

def short_result(output: str, limit: int = 1500) -> str:
    """Короткий результат."""
    lines = output.splitlines()
    res = [l[len(RESULT_PREFIX):].strip() for l in lines if l.startswith(RESULT_PREFIX)]
    if not res:
        res = [l for l in lines if l.strip()][-5:]
    result = '\n'.join(res)
    return result[:limit]

def run_task(task: dict) -> None:
    """Выполнить задачу."""
    kind = task['KIND']
    if kind not in TASKS:
        set_status(task['id'], STATUS='failed', FINISHED_AT=datetime.now(), RESULT='неизвестная задача')
        return
    title, args, timeout = TASKS[kind]
    extra = (task.get('ARGS') or '').split()
    args = args + extra
    set_status(task['id'], STATUS='running', STARTED_AT=datetime.now())
    tlg.send_message(f'▶️ Запущено: {title} (#{task["id"]}).')
    started = datetime.now()
    try:
        completed = subprocess.run(
            [sys.executable, 'main.py', *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'}
        )
        output = (completed.stdout or '') + '\n' + (completed.stderr or '')
        ok = completed.returncode == 0
    except subprocess.TimeoutExpired:
        output, ok = 'превышено время выполнения', False
    minutes = (datetime.now() - started).total_seconds() / 60
    result = short_result(output)
    set_status(task['id'], STATUS='done' if ok else 'failed', FINISHED_AT=datetime.now(), RESULT=result)
    tlg.send_message(f'{"✅ Готово" if ok else "❌ Ошибка"}: {title} за {minutes:.0f} мин.\n{result}')
    logger.info(f'Задача #{task["id"]} {kind}: {"готово" if ok else "ошибка"}')

# Автономные задачи: (вид, день недели 0=пн … 6=вс, с какого часа МСК, не чаще чем раз в N дней).
SCHEDULE = [
    ('blind', 5, 14, 6),   # суббота после 14:00 — полностью слепой тест всей системы
]


def enqueue(kind: str, source: str = 'schedule', chat_id: int | None = None) -> int:
    """ Поставить задачу в очередь (если такой же ещё нет в очереди); id задачи."""
    rows = SQLmain.run_sql_command(
        "SELECT id FROM task_requests WHERE KIND = :k AND STATUS IN ('new', 'running')", {'k': kind})
    if rows:
        return rows[0]['id']
    with Session(bind=main_engine) as session:
        task = tables.TaskRequest(CREATED_AT=datetime.now(), KIND=kind, ARGS=None, CHAT_ID=chat_id,
                                  STATUS='new', RESULT=source)
        session.add(task)
        session.commit()
        return task.id


def schedule(now: datetime):
    """ Плановые задачи: слепой тест раз в неделю — основа самонастройки (переключатели режимов)."""
    for kind, weekday, hour, every in SCHEDULE:
        if now.weekday() != weekday or now.hour < hour:
            continue
        last = SQLmain.run_sql_command(
            "SELECT MAX(CREATED_AT) AS t FROM task_requests WHERE KIND = :k", {'k': kind})[0]['t']
        if last and (now - datetime.fromisoformat(str(last))).days < every:
            continue
        logger.info(f'Плановая задача {kind}: #{enqueue(kind)}')


def run_daemon():
    """Демон, ожидающий задачи."""
    logger.info('Демон задач запущен')
    while True:
        try:
            SQLmain.save_activity(SERVICE, 'ожидание задач')
            schedule(datetime.now())
            task = next_task()
            if task:
                run_task(task)
                continue
        except Exception as error:
            logger.exception(error)
        sleep(TASKS_POLL)
