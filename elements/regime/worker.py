import json
import logging
from datetime import datetime

import pandas as pd
from elements.regime.detector import REGIMES, regime_info, regime_series, breadth_from_panel
from source.settings.settings import handler, INDEX_SECID, REGIME_CONFIRM_DAYS
from source.sql.main import SQLmain
import source.sql.tables as tables

logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

COMPONENT_COLUMNS = ['dist_sma200', 'sma_cross', 'r5', 'r63', 'drawdown', 'vol_pct', 'breadth']


class MarketRegime(SQLmain):
    @classmethod
    def history(cls, panel: pd.DataFrame, candles: pd.DataFrame | None = None) -> pd.DataFrame:
        candles = candles if candles is not None else cls.load_candles()
        index = candles[candles['SECID'] == INDEX_SECID][['DATE', 'CLOSE']]
        return regime_series(index, breadth_from_panel(panel), REGIME_CONFIRM_DAYS)

    @classmethod
    def save(cls, frame: pd.DataFrame, last_days: int | None = None) -> int:
        if last_days is None:
            sub = frame
        else:
            sub = frame.tail(last_days)
        rows = []
        for idx, row in sub.iterrows():
            if pd.isna(row['score']):
                continue
            comp = {}
            for col in COMPONENT_COLUMNS:
                val = row[col]
                if not pd.isna(val):
                    comp[col] = round(float(val), 4)
            rows.append({
                'DATE': idx.date(),
                'SCORE': float(row['score']),
                'RAW': str(row['raw']),
                'REGIME': str(row['regime']),
                'COMPONENTS': json.dumps(comp),
                'CREATED_AT': datetime.now()
            })
        cls.upsert(tables.MarketRegime, rows, ['DATE'])
        return len(rows)

    @classmethod
    def previous(cls, before):
        sql = 'SELECT REGIME FROM market_regime WHERE "DATE" < :d ORDER BY "DATE" DESC LIMIT 1'
        res = cls.run_sql_command(sql, {'d': before})
        return res[0]['REGIME'] if res else None

    @classmethod
    def update(cls, panel: pd.DataFrame, notify: bool = True) -> dict:
        frame = cls.history(panel)
        last_date = frame.index[-1]
        prev = cls.previous(last_date.date())
        cls.save(frame)
        current = frame['regime'].iloc[-1]
        info = regime_info(current)
        result = {
            'date': last_date.date(),
            'regime': current,
            'title': info['title'],
            'emoji': info['emoji'],
            'score': float(frame['score'].iloc[-1]),
            'changed': prev is not None and prev != current,
            'previous': prev,
            'text': describe(frame)
        }
        if notify and result['changed']:
            from elements.notification.worker import TelegramNotification as tlg
            tlg.send_message(
                f'{info["emoji"]} Режим рынка сменился: {regime_info(prev)["title"]} → {info["title"]}.\n'
                + result['text']
            )
        logger.info(result['text'])
        return result

    @staticmethod
    def current() -> dict | None:
        try:
            sql = 'SELECT * FROM market_regime ORDER BY "DATE" DESC LIMIT 1'
            res = SQLmain.run_sql_command(sql)
            if not res:
                return None
            row = dict(res[0])
            info = regime_info(row['REGIME'])
            row.update({'title': info['title'], 'emoji': info['emoji']})
            return row
        except Exception:
            return None

    @staticmethod
    def day_regimes(frame: pd.DataFrame) -> pd.Series:
        return frame['regime']


def describe(frame: pd.DataFrame) -> str:
    def fmt(value, spec):
        if pd.isna(value):
            return '—'
        return format(value, spec)

    last = frame.iloc[-1]
    info = regime_info(last['regime'])
    line1 = f'{info["emoji"]} Режим: {info["title"]} (оценка {fmt(last["score"], "+.2f")} из ±1).'
    line2 = (
        f'IMOEX: от максимума за год {fmt(last["drawdown"], "+.0%")}, за 3 мес. {fmt(last["r63"], "+.0%")}, '
        f'волатильность выше {fmt(last["vol_pct"], ".0%")} прошлых дней, '
        f'бумаг в росте (выше средней 50 дн.) {fmt(last["breadth"], ".0%")}.'
    )
    return f'{line1}\n{line2}'


def share_by_regime(regimes: pd.Series) -> dict:
    total = len(regimes)
    shares = {}
    for r in REGIMES:
        key = r['key']
        shares[key] = round((regimes == key).sum() / total, 3) if total > 0 else 0.0
    return shares
