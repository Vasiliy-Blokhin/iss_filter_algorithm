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

subprocess.run(['sh', f'{BASE_DIR}{RESTART_BASH_NAME}'])
while True:
    try:
        diff_time = 2 * TIME_UPDATE / 60
        activity_time = a.get_all_data(table=tables.Activity)[0]['last_time']
        current_time = a.curent_msc_time()
        if activity_time < current_time - diff_time or current_time == 9.0:
            subprocess.run(['sh', f'{BASE_DIR}{RESTART_BASH_NAME}'])
            tlg.send_message(text=RESTART_MESSAGE)
            logger.info(RESTART_MESSAGE)
    except Exception as error:
        tlg.send_message(text=ERROR_MESSAGE + str(error))
        logger.error(error)
    finally:
        sleep(50)
