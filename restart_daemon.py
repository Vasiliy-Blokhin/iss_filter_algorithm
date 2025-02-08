import logging
from time import sleep
import subprocess

from elements.algorithm.worker import Algorithm as a
from source.settings.settings import (
    handler,
    TIME_UPDATE,
    ERROR_MESSAGE,
    RESTART_MESSAGE,
    BASE_DIR,
    RESTART_BASH_NAME
)
from elements.notification.worker import TelegramNotification as tlg
import source.sql.tables as tables

# Запуск логгера.
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

while True:
    try:
        diff_time = TIME_UPDATE / 30
        activity_time = a.get_all_data(table=tables.Activity)[0]['last_time']
        current_time = a.curent_msc_time()
        if activity_time < current_time - diff_time or current_time == 9.0:
            tlg.send_message(text=RESTART_MESSAGE)
            subprocess.run(['sh', f'{BASE_DIR}{RESTART_BASH_NAME}'])
            logger.info(RESTART_MESSAGE)
    except Exception as error:
        tlg.send_message(text=ERROR_MESSAGE + str(error))
        logger.error(error)
    finally:
        sleep(50)
