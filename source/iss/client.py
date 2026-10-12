""" Клиент ISS Мосбиржи (публичный, без авторизации; данные с задержкой 15 минут)."""
import logging
from datetime import date
from time import sleep

import pandas as pd
import requests

from source.settings.settings import (
    handler,
    ISS_BASE_URL,
    MARKET_PROXY,
    HTTP_TIMEOUT,
    HTTP_RETRIES,
    BOARD,
    INDEX_SECID,
)


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

CANDLE_COLUMNS = ['open', 'close', 'high', 'low', 'value', 'volume', 'begin']


class ISSClient:
    """ Минимальный клиент ISS: свечи, список бумаг, текущие котировки."""

    def __init__(self, base_url: str = ISS_BASE_URL, proxy: str | None = MARKET_PROXY):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'iss_filter_algorithm/2.0'
        if proxy:
            self.session.proxies = {'http': proxy, 'https': proxy}

    def get(self, path: str, params: dict | None = None) -> dict:
        params = {'iss.meta': 'off', **(params or {})}
        url = f'{self.base_url}/{path.lstrip("/")}'
        last_error = None
        for attempt in range(1, HTTP_RETRIES + 1):
            try:
                response = self.session.get(url, params=params, timeout=HTTP_TIMEOUT)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as error:
                last_error = error
                logger.warning(f'ISS {path}: попытка {attempt} не удалась ({error})')
                sleep(2 * attempt)
        raise ConnectionError(f'ISS недоступен: {url} ({last_error})')

    @staticmethod
    def block_to_frame(block: dict) -> pd.DataFrame:
        return pd.DataFrame(block.get('data', []), columns=block.get('columns', []))

    def candles(
        self,
        secid: str,
        start: date,
        till: date | None = None,
        index: bool = False,
        interval: int = 24,
    ) -> pd.DataFrame:
        """ Свечи бумаги (или индекса) с постраничной загрузкой: 24 — день, 60 — час."""
        if index:
            path = f'engines/stock/markets/index/securities/{secid}/candles.json'
        else:
            path = (
                f'engines/stock/markets/shares/boards/{BOARD}'
                f'/securities/{secid}/candles.json'
            )
        params = {'interval': interval, 'from': start.isoformat()}
        if till:
            params['till'] = till.isoformat()

        frames = []
        offset = 0
        while True:
            block = self.get(path, {**params, 'start': offset})['candles']
            frame = self.block_to_frame(block)
            if frame.empty:
                break
            frames.append(frame)
            offset += len(frame)
            if len(frame) < 500:
                break

        if not frames:
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'value', 'volume'])
        data = pd.concat(frames, ignore_index=True)[CANDLE_COLUMNS]
        if interval != 24:
            data['begin'] = pd.to_datetime(data['begin'])
            return data[['begin', 'open', 'high', 'low', 'close', 'value', 'volume']]
        data['date'] = pd.to_datetime(data['begin']).dt.normalize()
        return data.drop(columns='begin')[
            ['date', 'open', 'high', 'low', 'close', 'value', 'volume']
        ]

    def index_candles(self, start: date, till: date | None = None) -> pd.DataFrame:
        return self.candles(INDEX_SECID, start, till, index=True)

    def board_snapshot(self) -> pd.DataFrame:
        """ Список акций режима TQBR с текущими котировками."""
        payload = self.get(
            f'engines/stock/markets/shares/boards/{BOARD}/securities.json',
            {
                'iss.only': 'securities,marketdata',
                'securities.columns': 'SECID,SHORTNAME,LOTSIZE,MINSTEP,PREVPRICE,STATUS',
                'marketdata.columns': (
                    'SECID,LAST,OPEN,HIGH,LOW,BID,OFFER,VALTODAY,'
                    'UPDATETIME,TRADINGSTATUS,TRADINGSESSION'
                ),
            },
        )
        securities = self.block_to_frame(payload['securities'])
        marketdata = self.block_to_frame(payload['marketdata'])
        return securities.merge(marketdata, on='SECID', how='left')
