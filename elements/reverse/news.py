""" Новостной фон: сбор лент, оценка локальной моделью (AI_Manager) и признаки по бумагам.

Сбор (дёшево, без ИИ): RSS-ленты NEWS_FEEDS и новости сайта Мосбиржи (ISS sitenews)
-> таблица news (дубликаты по хэшу заголовка отбрасываются).

Оценка (REVERSE_NEWS_MODEL, по умолчанию qwen36-35b-a3b — лучший русский): пачками по
NEWS_BATCH заголовков модель определяет затронутые бумаги (только из списка ликвидных,
названия сопоставляются с тикерами локально), тональность для цены на 1-3 дня (-1..1),
важность (0..1) и тип события -> news_scores; новости о рынке в целом — SECID = 'MARKET'.

Признаки дня D (сигнал после закрытия D, вход D+1): новости, опубликованные до утреннего
запуска сигналов следующего торгового дня, т.е. время публикации минус NEWS_SHIFT_HOURS
приводится к последнему торговому дню не позже этой даты:
    news_score — сумма тональность x важность по бумаге, обрезанная до [-1, 1];
    news_count — число новостей о бумаге;
    mkt_news   — то же для новостей о рынке/экономике в целом.
"""
import hashlib
import html
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import sqlalchemy as sa
from sqlalchemy.orm import Session

from elements.ai.worker import AIReviewer
from source.settings.settings import (
    handler,
    HTTP_TIMEOUT,
    ISS_BASE_URL,
    MARKET_PROXY,
    TIMEZONE,
    NEWS_FEEDS,
    NEWS_BATCH,
    NEWS_SHIFT_HOURS,
    REVERSE_NEWS_MODEL,
    MIN_TURNOVER,
)
from source.sql.main import SQLmain, main_engine
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

MARKET = 'MARKET'
EVENTS = ['dividends', 'earnings', 'guidance', 'deal', 'regulation', 'sanctions', 'macro',
          'rate', 'management', 'buyback', 'spo', 'index', 'legal', 'operations', 'other']

# Привычные названия ликвидных компаний (для промпта и сопоставления ответа модели).
ALIASES = {
    'SBER': 'Сбербанк, Сбер', 'SBERP': 'Сбербанк преф', 'GAZP': 'Газпром', 'LKOH': 'Лукойл',
    'ROSN': 'Роснефть', 'NVTK': 'Новатэк', 'GMKN': 'Норникель, Норильский никель',
    'PLZL': 'Полюс', 'YDEX': 'Яндекс', 'T': 'Т-Технологии, Т-Банк, Тинькофф', 'OZON': 'Озон',
    'VTBR': 'ВТБ (банк)', 'SMLT': 'ГК Самолет (девелопер жилья, не авиация)', 'MAGN': 'ММК, Магнитогорский меткомбинат',
    'TATN': 'Татнефть', 'TATNP': 'Татнефть преф', 'X5': 'X5, ИКС 5, Пятёрочка',
    'CHMF': 'Северсталь', 'SNGS': 'Сургутнефтегаз', 'SNGSP': 'Сургутнефтегаз преф',
    'MTSS': 'МТС', 'VKCO': 'ВК, VK', 'ALRS': 'Алроса', 'NLMK': 'НЛМК', 'SVCB': 'Совкомбанк',
    'MGNT': 'Магнит', 'RUAL': 'Русал', 'SIBN': 'Газпром нефть', 'AFKS': 'АФК Система',
    'PHOR': 'ФосАгро', 'AFLT': 'Аэрофлот', 'TRNFP': 'Транснефть', 'MOEX': 'Мосбиржа',
    'HEAD': 'HeadHunter, Хэдхантер', 'POSI': 'Positive Technologies, Позитив',
    'SPBE': 'СПБ Биржа', 'DOMRF': 'ДОМ.РФ', 'MTLR': 'Мечел', 'RAGR': 'Русагро',
    'IRAO': 'Интер РАО', 'BSPB': 'Банк Санкт-Петербург', 'FLOT': 'Совкомфлот',
    'SELG': 'Селигдар', 'RTKM': 'Ростелеком', 'RNFT': 'РуссНефть', 'UPRO': 'Юнипро',
    'SGZH': 'Сегежа', 'ASTR': 'Астра', 'CBOM': 'МКБ, Московский кредитный банк',
    'ENPG': 'Эн+', 'FEES': 'Россети ФСК', 'HYDR': 'РусГидро', 'UGLD': 'ЮГК',
    'WUSH': 'Whoosh, Вуш', 'VSEH': 'ВсеИнструменты', 'PIKK': 'ПИК', 'LSRG': 'ЛСР',
    'BELU': 'Novabev, Белуга', 'FIVE': 'X5', 'AQUA': 'Инарктика', 'LENT': 'Лента',
    'FIXP': 'Fix Price', 'MDMG': 'Мать и дитя', 'SFIN': 'ЭсЭфАй', 'RENI': 'Ренессанс страхование',
    'MSNG': 'Мосэнерго', 'OGKB': 'ОГК-2', 'TGKA': 'ТГК-1', 'NMTP': 'НМТП', 'KMAZ': 'КамАЗ',
    'GLTR': 'Глобалтранс', 'ELFV': 'ЭЛ5-Энерго', 'MVID': 'М.Видео', 'SOFL': 'Софтлайн',
    'DIAS': 'Диасофт', 'DELI': 'Делимобиль', 'MBNK': 'МТС Банк', 'LEAS': 'Европлан',
    'CARM': 'Кармани', 'PRMD': 'Промомед', 'GECO': 'Генетико', 'IVAT': 'IVA Technologies',
    'APTK': 'Аптечная сеть 36,6', 'KZOS': 'Казаньоргсинтез', 'NKNC': 'Нижнекамскнефтехим',
    'BANEP': 'Башнефть преф', 'MRKC': 'Россети Центр', 'UWGN': 'ОВК', 'ZAYM': 'Займер',
}

SYSTEM_PROMPT = (
    'Ты — аналитик новостей для краткосрочной торговли акциями на Московской бирже. '
    'Для каждой новости определи, какие акции из справочника она прямо затрагивает (тикеры только '
    'из справочника; новость о рынке, экономике, ставке ЦБ, рубле, нефти, санкциях в целом — '
    'тикер MARKET), и для КАЖДОГО тикера отдельно — тональность для его цены на ближайшие 1-3 дня '
    'от -1 (сильно негативно) до 1 (сильно позитивно): у участников одного события она может '
    'быть разной (истец и ответчик, покупатель и продавец). Также важность новости от 0 (шум) '
    'до 1 (событие, двигающее цену) и тип события. Не выдумывай связь, которой нет в тексте. '
    'В ответ включай ТОЛЬКО новости, затрагивающие рынок или компании справочника; '
    'остальные (спорт, происшествия, политика без экономических последствий) пропускай. '
    'Ответ — строго JSON без пояснений: {"items": [{"i": номер, "t": {"SBER": 0.3, "MARKET": -0.1}, '
    '"w": важность, "e": "' + '|'.join(EVENTS) + '"}]}.'
)


def msk_now() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)


def clean(text: str | None, limit: int = 300) -> str:
    text = html.unescape(re.sub(r'<[^>]+>', ' ', text or ''))
    return re.sub(r'\s+', ' ', text).strip()[:limit]


def news_uid(title: str) -> str:
    return hashlib.sha1(re.sub(r'\W+', '', title.lower()).encode('utf-8')).hexdigest()


def to_msk(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        moment = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(ZoneInfo(TIMEZONE)).replace(tzinfo=None)
    return moment


class NewsCollector(SQLmain):

    @staticmethod
    def fetch_rss(url: str) -> list[dict]:
        response = requests.get(url, timeout=HTTP_TIMEOUT, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items = []
        for item in root.iter('item'):
            title = clean(item.findtext('title'), 400)
            if not title:
                continue
            items.append({
                'SOURCE': url.split('/')[2], 'TITLE': title,
                'SUMMARY': clean(item.findtext('description')),
                'URL': (item.findtext('link') or '').strip()[:500],
                'PUBLISHED': to_msk(item.findtext('pubDate')) or msk_now(),
            })
        return items

    @staticmethod
    def fetch_moex() -> list[dict]:
        proxies = {'http': MARKET_PROXY, 'https': MARKET_PROXY} if MARKET_PROXY else None
        response = requests.get(f'{ISS_BASE_URL}/sitenews.json', params={'iss.meta': 'off'},
                                timeout=HTTP_TIMEOUT, proxies=proxies)
        response.raise_for_status()
        block = response.json()['sitenews']
        return [
            {'SOURCE': 'moex.com', 'TITLE': clean(row['title'], 400), 'SUMMARY': '',
             'URL': f'https://www.moex.com/n{row["id"]}', 'PUBLISHED': to_msk(row['published_at'])}
            for row in (dict(zip(block['columns'], values)) for values in block['data'])
            if row.get('title')
        ]

    @classmethod
    def collect(cls) -> int:
        """ Новые новости из всех лент; возвращает число добавленных."""
        items = []
        for url in NEWS_FEEDS:
            try:
                items += cls.fetch_rss(url)
            except (requests.RequestException, ET.ParseError) as error:
                logger.warning(f'Лента {url}: {error}')
        try:
            items += cls.fetch_moex()
        except (requests.RequestException, KeyError, ValueError) as error:
            logger.warning(f'Новости Мосбиржи: {error}')
        rows, seen = [], set()
        for item in items:
            uid = news_uid(item['TITLE'])
            if uid in seen:
                continue
            seen.add(uid)
            rows.append({**item, 'UID': uid, 'CREATED_AT': msk_now()})
        if not rows:
            return 0
        with Session(bind=main_engine) as s:
            known = set(s.execute(
                sa.select(tables.News.UID).where(tables.News.UID.in_([row['UID'] for row in rows]))
            ).scalars())
        fresh = [row for row in rows if row['UID'] not in known]
        if fresh:
            cls.upsert(tables.News, fresh, ['UID'])
        logger.info(f'Новости: получено {len(rows)}, новых {len(fresh)}')
        return len(fresh)


def universe(limit: int = 90) -> dict[str, str]:
    """ {тикер: названия} — ликвидные бумаги (по последнему медианному обороту) для промпта."""
    rows = SQLmain.run_sql_command(
        'SELECT c.SECID, AVG(c.VALUE) AS V FROM candles c '
        'WHERE c."DATE" >= date((SELECT MAX("DATE") FROM candles), \'-30 day\') AND c.SECID != \'IMOEX\' '
        'GROUP BY c.SECID ORDER BY V DESC LIMIT :limit', {'limit': limit}
    )
    names = SQLmain.securities().set_index('SECID')['SHORTNAME'].to_dict()
    result = {}
    for row in rows:
        if (row['V'] or 0) < MIN_TURNOVER / 4:
            continue
        secid = row['SECID']
        result[secid] = ALIASES.get(secid) or re.sub(r'\s*(ао|ап|-п|-ао|ап)$', '', str(names.get(secid) or secid))
    return result


class NewsScorer(SQLmain):
    """ Оценка новостей локальной моделью пачками."""

    def __init__(self, model: str = REVERSE_NEWS_MODEL):
        self.ai = AIReviewer(model=model)
        self.model = model
        self.tickers = universe()
        self.reference = '; '.join(f'{secid} — {names}' for secid, names in self.tickers.items())

    @staticmethod
    def pending(limit: int) -> list[dict]:
        rows = SQLmain.run_sql_command(
            'SELECT id, TITLE, SUMMARY, SOURCE FROM news WHERE SCORED IS NULL OR SCORED = 0 '
            'ORDER BY PUBLISHED DESC LIMIT :limit', {'limit': limit}
        )
        return [dict(row) for row in rows]

    def score_batch(self, batch: list[dict]) -> list[dict]:
        lines = '\n'.join(
            f'{number}. {item["TITLE"]}' + (f' — {item["SUMMARY"][:200]}' if item['SUMMARY'] else '')
            for number, item in enumerate(batch)
        )
        content = self.ai.chat([
            {'role': 'system', 'content': SYSTEM_PROMPT + '\nСправочник: ' + self.reference},
            {'role': 'user', 'content': lines},
        ], max_tokens=35 * len(batch) + 150, extra={'presence_penalty': 0.0, 'temperature': 0.1})
        return self.parse(content, batch)

    def parse(self, content: str, batch: list[dict]) -> list[dict]:
        match = re.search(r'\{.*\}', content, re.S)
        data = json.loads(match.group(0) if match else content)
        rows = []
        for item in data.get('items', []):
            try:
                number = int(item.get('i'))
            except (TypeError, ValueError):
                continue
            if not 0 <= number < len(batch):
                continue
            importance = float(np.clip(float(item.get('w', item.get('importance')) or 0), 0, 1))
            event = str(item.get('e', item.get('event')) or 'other').split('|')[0][:20]
            tickers = item.get('t', item.get('tickers')) or {}
            if isinstance(tickers, list):  # старый формат: одна тональность на все бумаги
                tickers = {ticker: item.get('s', item.get('sentiment')) for ticker in tickers}
            for ticker, sentiment in tickers.items():
                ticker = str(ticker).upper().strip()
                if ticker != MARKET and ticker not in self.tickers:
                    continue
                try:
                    sentiment = float(np.clip(float(sentiment or 0), -1, 1))
                except (TypeError, ValueError):
                    continue
                rows.append({
                    'NEWS_ID': batch[number]['id'], 'SECID': ticker, 'SENTIMENT': sentiment,
                    'IMPORTANCE': importance, 'EVENT': event, 'MODEL': self.model,
                })
        return rows

    def run(self, limit: int = 400, batch_size: int = NEWS_BATCH, deadline: datetime | None = None) -> int:
        """ Оценить до limit неоценённых новостей (сначала свежие); deadline — не начинать
        новую пачку позже этого времени."""
        items = self.pending(limit)
        done = 0
        for start in range(0, len(items), batch_size):
            if deadline and msk_now() >= deadline:
                break
            batch = items[start:start + batch_size]
            try:
                rows = self.score_batch(batch)
            except Exception as error:
                logger.error(f'Оценка новостей не удалась: {error}')
                break
            self.upsert(tables.NewsScore, rows, ['NEWS_ID', 'SECID'])
            with Session(bind=main_engine) as s:
                s.execute(sa.update(tables.News).where(tables.News.id.in_([b['id'] for b in batch]))
                          .values(SCORED=True))
                s.commit()
            done += len(batch)
            self.save_activity('reverse', f'news scoring {done}/{len(items)} at {msk_now():%H:%M}')
            logger.info(f'Оценено новостей {done}/{len(items)}, связей с бумагами {len(rows)}')
        return done


def load_scores() -> pd.DataFrame:
    rows = SQLmain.run_sql_command(
        'SELECT s.SECID, s.SENTIMENT, s.IMPORTANCE, s.EVENT, n.PUBLISHED, n.TITLE '
        'FROM news_scores s JOIN news n ON n.id = s.NEWS_ID'
    )
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        return pd.DataFrame(columns=['SECID', 'SENTIMENT', 'IMPORTANCE', 'EVENT', 'PUBLISHED', 'TITLE'])
    frame['PUBLISHED'] = pd.to_datetime(frame['PUBLISHED'])
    return frame


def signal_dates(published: pd.Series, dates: np.ndarray) -> np.ndarray:
    """ Торговый день, к сигналу которого относится новость (см. описание модуля)."""
    shifted = (published - pd.Timedelta(hours=NEWS_SHIFT_HOURS)).dt.normalize().to_numpy()
    position = np.searchsorted(dates, shifted, side='right') - 1
    return position


def news_features(panel: pd.DataFrame, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """ Колонки news_score, news_count, mkt_news в панели (NaN — новостей о бумаге не было)."""
    scores = load_scores() if scores is None else scores
    panel = panel.copy()
    for column in ('news_score', 'news_count', 'mkt_news'):
        panel[column] = np.nan
    if scores.empty or panel.empty:
        return panel
    dates = np.sort(panel['DATE'].unique())
    scores = scores.assign(DAY=signal_dates(scores['PUBLISHED'], dates))
    scores = scores[scores['DAY'] >= 0]
    scores = scores.assign(DATE=dates[scores['DAY'].to_numpy()],
                           VALUE=scores['SENTIMENT'] * scores['IMPORTANCE'])
    stock = scores[scores['SECID'] != MARKET].groupby(['SECID', 'DATE']).agg(
        news_score=('VALUE', 'sum'), news_count=('VALUE', 'size'))
    stock['news_score'] = stock['news_score'].clip(-1, 1)
    market = scores[scores['SECID'] == MARKET].groupby('DATE')['VALUE'].mean().clip(-1, 1)
    panel = panel.drop(columns=['news_score', 'news_count', 'mkt_news'])
    panel = panel.join(stock, on=['SECID', 'DATE'])
    panel['mkt_news'] = panel['DATE'].map(market)
    # Дни, когда новости уже собирались: нет новостей о бумаге — фон 0, а не «неизвестно».
    covered = panel['DATE'] >= scores['DATE'].min()
    panel.loc[covered, 'news_count'] = panel.loc[covered, 'news_count'].fillna(0)
    panel.loc[covered, 'news_score'] = panel.loc[covered, 'news_score'].fillna(0)
    return panel


def headlines(secid: str, since: datetime, limit: int = 12) -> list[str]:
    """ Оценённые заголовки о бумаге и о рынке с момента since (для ИИ-проверки сигнала)."""
    rows = SQLmain.run_sql_command(
        'SELECT n.PUBLISHED, n.TITLE, s.SECID, s.SENTIMENT, s.IMPORTANCE, s.EVENT FROM news_scores s '
        'JOIN news n ON n.id = s.NEWS_ID WHERE s.SECID IN (:secid, :market) AND n.PUBLISHED >= :since '
        'ORDER BY (s.SECID = :secid) DESC, s.IMPORTANCE DESC, n.PUBLISHED DESC LIMIT :limit',
        {'secid': secid, 'market': MARKET, 'since': since, 'limit': limit},
    )
    return [
        f'[{pd.Timestamp(row["PUBLISHED"]):%d.%m %H:%M}] {"" if row["SECID"] == secid else "(рынок) "}'
        f'{row["TITLE"]} (тон {row["SENTIMENT"]:+.1f}, важн. {row["IMPORTANCE"]:.1f}, {row["EVENT"]})'
        for row in rows
    ]


def recent_since(days: int = 4) -> datetime:
    return msk_now() - timedelta(days=days)
