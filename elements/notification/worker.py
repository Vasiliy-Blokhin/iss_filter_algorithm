import logging

import requests
import sqlalchemy as sa

from source.sql.main import main_engine

from source.settings.settings import (
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID,
    START_MESSAGE,
    HTTP_TIMEOUT,
    handler,
)


# Запуск логгера.
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class TelegramNotification:

    @staticmethod
    def chat_ids() -> list:
        """ TLG_REPORT_CHAT_ID или подписчики Telegram-бота (нажавшие /start в боте отчётов)."""
        if TELEGRAM_CHAT_ID:
            return [TELEGRAM_CHAT_ID]
        try:
            with main_engine.connect() as connection:
                return [row[0] for row in connection.execute(sa.text('SELECT CHAT_ID FROM tg_subscribers'))]
        except sa.exc.OperationalError:
            return []

    @classmethod
    def send_message(cls, text: str):
        """ Уведомление в Telegram; без токена или получателей — только в лог."""
        if not TELEGRAM_TOKEN:
            logger.info(f'(нет TLG_REPORT_TOKEN) {text}')
            return
        recipients = cls.chat_ids()
        if not recipients:
            logger.info(f'(нет подписчиков — нажмите /start в боте отчётов) {text}')
            return
        for chat_id in recipients:
            try:
                requests.post(
                    f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
                    json={'chat_id': chat_id, 'text': START_MESSAGE + text},
                    timeout=HTTP_TIMEOUT,
                )
            except requests.RequestException as error:
                logger.error(f'Telegram: {error}')


def signals_message(signals: list[dict]) -> str:
    if not signals:
        return 'Сигналов на завтра нет.'
    lines = [f'Сигналы на основе закрытия {signals[0]["SIGNAL_DATE"]}:']
    for signal in signals:
        mark = '✅' if signal['STATUS'] == 'approved' else '⛔'
        lines.append(
            f'{mark} {signal["SECID"]}: p={signal["PROBABILITY"]:.0%}, '
            f'E={signal["EXPECTED_RETURN"]:.2%}, стоп {signal["STOP_PRICE"]:.4g}, '
            f'тейк {signal["TAKE_PRICE"]:.4g}'
            + (f' | ИИ: {signal["AI_DECISION"]} ({signal.get("AI_REASON") or ""})'
               if signal['AI_DECISION'] not in ('n/a', None) else '')
        )
    return '\n'.join(lines)
