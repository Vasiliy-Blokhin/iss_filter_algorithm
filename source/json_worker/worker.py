import requests
import json
import logging
import sys
from time import sleep

# Запуск логгера.
# Описание хандлера для логгера.
handler = logging.StreamHandler(sys.stdout)
formater = logging.Formatter(
    '%(name)s, %(funcName)s, %(asctime)s, %(levelname)s - %(message)s.'
)

handler.setFormatter(formater)
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class JSONSaveAndRead():
    """ Родительский класс для базовых действий."""

    def __init__(self, url: str | None, file: str | None) -> None:
        self.url: str | None = url
        self.file: str | None = file

    @classmethod
    def get_api_response(
        cls,
        url=None,
        post=False,
        headers=None,
        body=None,
        delete=None
    ):
        """ Получение информации с запроса на сервер."""
        try:
            if url is None:
                url = cls.url
            if post:
                return requests.post(url, headers=headers, json=body).json()
            elif delete:
                return requests.delete(url, headers=headers).json()
            return requests.get(url, headers=headers).json()
        except json.decoder.JSONDecodeError:
            return []
        finally:
            sleep(5)

    @classmethod
    def save_file(cls, data, file=None):
        """ Сохранение информации в файле."""
        if file is None:
            file = cls.file
        with open(file, 'w') as outfile:
            json.dump(data, outfile)

    @classmethod
    def read_file(cls, file=None):
        """ Чтение информации с файла."""
        if file is None:
            file = cls.file
        with open(file) as json_file:
            return json.load(json_file)

    class Meta:
        abstract = True
