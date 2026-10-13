""" ИИ-проверка кандидатов через AI_Manager (OpenAI-совместимый /v1).

ИИ работает как фильтр-вето: модель уже отобрала кандидатов по статистике,
ИИ оценивает контекст (новости, аномалии свечей) и может отклонить сделку.
Решения ИИ сохраняются в сигналах, чтобы позже сравнить их с фактическим исходом.
"""
import json
import logging
import re
import xml.etree.ElementTree as ET
from os import getenv
from time import sleep

import requests

from elements.strategy.worker import STRATEGY_LOGIC
from source.settings.settings import (
    handler,
    AI_URL,
    AI_PASSWORD,
    AI_MODEL,
    AI_TIMEOUT,
    AI_RETRIES,
    AI_PROJECT,
    HOLD_DAYS,
    HTTP_TIMEOUT,
)


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def strategy_logic(strategy: str | None) -> str:
    """ Описание логики стратегии(й) сигнала, например 'squeeze+momentum'."""
    names = (strategy or 'model').split('+')
    return '; '.join(STRATEGY_LOGIC.get(name, name) for name in names)

SYSTEM_PROMPT = (
    'Ты — риск-аналитик краткосрочных сделок на Московской бирже. '
    'Статистическая модель уже отобрала акцию для покупки с удержанием '
    f'{HOLD_DAYS} торговых дня (вход на открытии следующего дня, выход по стопу, тейку '
    'или по времени). Твоя задача — найти причины НЕ входить: корпоративные события '
    '(отсечка дивидендов в ближайшие дни — цена упадёт на размер дивиденда, SPO, '
    'делистинг, санкции), аномальные свечи, явный негатив в новостях. '
    'НЕ отклоняй сделку из-за технических показателей (перекупленность, перепроданность, '
    'RSI, близость к максимумам или минимумам): они — часть логики стратегии, которая '
    'отобрала акцию, и уже проверены статистикой. '
    'Если явных причин нет — подтверди сделку. Не выдумывай факты, которых нет в данных. '
    'Ответь строго JSON без пояснений вокруг: '
    '{"decision": "buy" или "skip", "confidence": целое 0-100, "reason": "кратко по-русски"}'
)


class AIReviewer:

    def __init__(self, url: str = AI_URL, password: str = AI_PASSWORD, model: str = AI_MODEL):
        self.url = url.rstrip('/')
        self.password = password
        self.model = model

    def available(self) -> bool:
        try:
            response = requests.get(
                f'{self.url}/v1/models', headers=self.headers(), timeout=5
            )
            return response.ok
        except requests.RequestException:
            return False

    def headers(self) -> dict:
        return {
            'Authorization': f'Bearer {self.password}',
            'X-Project': AI_PROJECT,
            'X-Job-Timeout': str(int(AI_TIMEOUT)),
        }

    def chat(self, messages: list[dict], max_tokens: int = 400) -> str:
        # AI_Manager обслуживает несколько проектов и может перезапускать модель —
        # обрыв соединения повторяем.
        for attempt in range(1, AI_RETRIES + 1):
            try:
                response = requests.post(
                    f'{self.url}/v1/chat/completions',
                    headers=self.headers(),
                    json={
                        'model': self.model,
                        'messages': messages,
                        'temperature': 0.2,
                        'max_tokens': max_tokens,
                        'response_format': {'type': 'json_object'},
                    },
                    timeout=AI_TIMEOUT,
                )
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content'] or ''
            except (requests.ConnectionError, requests.Timeout) as error:
                if attempt == AI_RETRIES:
                    raise
                logger.warning(f'AI_Manager: попытка {attempt} не удалась ({error}), повтор через 60 с')
                sleep(60)
        return ''

    @staticmethod
    def parse(content: str) -> dict:
        match = re.search(r'\{.*\}', content, re.S)
        data = json.loads(match.group(0) if match else content)
        decision = str(data.get('decision', 'skip')).lower()
        return {
            'decision': 'buy' if decision == 'buy' else 'skip',
            'confidence': int(max(0, min(100, float(data.get('confidence', 0))))),
            'reason': str(data.get('reason', ''))[:500],
        }

    def review(self, candidate: dict, last_candles: list[dict], news: list[str]) -> dict:
        prompt = (
            f'Акция: {candidate["SECID"]} ({candidate.get("SHORTNAME")}), '
            f'сектор: {candidate.get("SECTOR")}.\n'
            f'Стратегия, отобравшая акцию: {strategy_logic(candidate.get("STRATEGY"))}.\n'
            f'Оценка модели: вероятность прибыли {candidate["PROBABILITY"]:.0%}, '
            f'ожидаемая доходность {candidate["EXPECTED_RETURN"]:.2%}.\n'
            f'Стоп {candidate["STOP_PRICE"]:.4g}, тейк {candidate["TAKE_PRICE"]:.4g}, '
            f'закрытие {candidate["CLOSE"]:.4g}.\n'
            f'Признаки: {json.dumps(candidate["FEATURES"], ensure_ascii=False)}\n'
            f'Последние дневные свечи (дата, O, H, L, C, оборот млн руб): '
            f'{json.dumps(last_candles, ensure_ascii=False)}\n'
            'Новости (могут быть нерелевантны): '
            + ('; '.join(news[:15]) if news else 'нет данных')
        )
        content = self.chat([
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ])
        return self.parse(content)


def fetch_news(keywords: list[str], limit: int = 15) -> list[str]:
    """ Заголовки из RSS (NEWS_RSS в .env, через запятую), отфильтрованные по ключевым словам."""
    feeds = [url.strip() for url in (getenv('NEWS_RSS') or '').split(',') if url.strip()]
    keywords = [word.lower() for word in keywords if word]
    titles = []
    for url in feeds:
        try:
            root = ET.fromstring(requests.get(url, timeout=HTTP_TIMEOUT).content)
        except (requests.RequestException, ET.ParseError) as error:
            logger.warning(f'RSS {url}: {error}')
            continue
        for item in root.iter('item'):
            title = (item.findtext('title') or '').strip()
            if title and any(word in title.lower() for word in keywords):
                titles.append(title)
    return titles[:limit]
