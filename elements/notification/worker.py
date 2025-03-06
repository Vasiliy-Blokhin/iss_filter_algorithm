import logging

from source.settings.settings import (
    TELEGRAM_URL,
    START_MESSAGE,
    handler,
)
from source.json_worker.worker import JSONSaveAndRead


# Запуск логгера.
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class TelegramNotification(JSONSaveAndRead):

    @classmethod
    def send_message(self, text):
        message = START_MESSAGE + text
        url = TELEGRAM_URL + message
        self.get_api_response(url=url)
        logger.info(f'Сообщение: {text} отправлено')
