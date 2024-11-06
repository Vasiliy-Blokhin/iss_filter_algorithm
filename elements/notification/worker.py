from source.settings.settings import (
    TELEGRAM_URL,
    START_MESSAGE,
)
from source.json_worker.worker import JSONSaveAndRead


class TelegramNotification(JSONSaveAndRead):

    @classmethod
    def send_message(self, text):
        message = START_MESSAGE + text
        url = TELEGRAM_URL + message
        print(url)
        self.get_api_response(
            url=url,
        )
