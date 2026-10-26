""" Общая настройка тестов: всегда временная БД, даже если тестовый модуль импортирует настройки
раньше, чем выставит свой DB_URL (load_dotenv не перезаписывает уже заданные переменные).
Без этого тесты пишут синтетические свечи и параметры в рабочую data/trading.db."""
import os
import tempfile
from pathlib import Path

os.environ['DB_URL'] = f'sqlite:///{Path(tempfile.mkdtemp()).as_posix()}/test.db'
os.environ.setdefault('AI_ENABLED', 'false')
os.environ.setdefault('MIN_TURNOVER', '1000000')


def pytest_configure(config):
    from source.settings.settings import DB_URL
    assert 'trading.db' not in DB_URL, f'тесты не должны работать с рабочей БД: {DB_URL}'
