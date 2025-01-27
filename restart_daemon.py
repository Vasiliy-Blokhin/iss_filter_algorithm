import logging
from time import sleep
import subprocess

from elements.algorithm.worker import Algorithm as a
from source.settings.settings import (
    handler,
    TIME_UPDATE,
    SERVICE_NAME,
    ERROR_MESSAGE,
    RESTART_MESSAGE
)
from elements.notification.worker import TelegramNotification as tlg
import source.sql.tables as tables

# Запуск логгера.
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

while True:
    try:
        diff_time = 2 * TIME_UPDATE / 60
        activity_time = a.get_all_data(table=tables.Activity)[0]['last_time']
        current_time = a.curent_msc_time()
        if activity_time < current_time - diff_time:
            subprocess.run(['sudo', 'systemctl', 'restart', SERVICE_NAME])
            tlg.send_message(text=RESTART_MESSAGE)
            logger.info(RESTART_MESSAGE)
    except Exception as error:
        tlg.send_message(text=ERROR_MESSAGE + str(error))
        logger.error(error)
    finally:
        sleep(TIME_UPDATE)
