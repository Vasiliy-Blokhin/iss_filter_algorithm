""" Бэктест стратегий на дневной истории (walk-forward для модели)."""
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from elements.strategy.worker import STRATEGIES, select, walk_forward_scores
from source.settings.settings import handler, HOLD_DAYS, MAX_SIGNALS
from source.settings.sectors import SECTORS
from source.sql.main import SQLmain
import source.sql.tables as tables


logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class Backtest(SQLmain):

    @staticmethod
    def trades(scored: pd.DataFrame, sectors: dict = SECTORS) -> pd.DataFrame:
        """ Сделки по дням: топ кандидатов, без повторного входа в открытую бумагу."""
        rows = []
        busy_until = {}
        for current, day in scored.groupby('DATE', sort=True):
            day = day[day['TRADE_RET'].notna()]
            free = day['SECID'].map(lambda secid: busy_until.get(secid, pd.Timestamp.min) <= current)
            day = day[free.astype(bool)]
            for _, row in select(day, sectors).iterrows():
                rows.append(row)
                busy_until[row['SECID']] = row['EXIT_DATE']
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)[
            ['DATE', 'SECID', 'CLOSE', 'PROBABILITY', 'EXPECTED_RETURN',
             'TRADE_RET', 'EXIT_KIND', 'EXIT_DATE']
        ].reset_index(drop=True)

    @staticmethod
    def metrics(trades: pd.DataFrame, all_dates: pd.Series) -> dict:
        if trades.empty:
            return {'TRADES': 0, 'WIN_RATE': 0.0, 'AVG_TRADE': 0.0, 'TOTAL_RETURN': 0.0,
                    'MAX_DRAWDOWN': 0.0, 'SHARPE': 0.0, 'PROFIT_FACTOR': 0.0}
        # Капитал делится на MAX_SIGNALS * HOLD_DAYS равных слотов.
        weight = 1 / (MAX_SIGNALS * HOLD_DAYS)
        daily = trades.groupby('EXIT_DATE')['TRADE_RET'].sum() * weight
        dates = pd.Index(sorted(all_dates.unique()))
        dates = dates[(dates >= trades['DATE'].min()) & (dates <= trades['EXIT_DATE'].max())]
        daily = daily.reindex(dates, fill_value=0.0)
        equity = (1 + daily).cumprod()
        drawdown = (equity / equity.cummax() - 1).min()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
        returns = trades['TRADE_RET']
        losses = -returns[returns < 0].sum()
        return {
            'TRADES': int(len(trades)),
            'WIN_RATE': float((returns > 0).mean()),
            'AVG_TRADE': float(returns.mean()),
            'TOTAL_RETURN': float(equity.iloc[-1] - 1),
            'MAX_DRAWDOWN': float(drawdown),
            'SHARPE': float(sharpe),
            'PROFIT_FACTOR': min(99.0, float(returns[returns > 0].sum() / losses)) if losses > 0 else 99.0,
        }

    @classmethod
    def run(cls, panel: pd.DataFrame, test_days: int = 250, save: bool = True) -> dict:
        """ Сравнение модели и бенчмарков на последних test_days торговых днях."""
        dates = np.sort(panel['DATE'].unique())
        start = pd.Timestamp(dates[max(0, len(dates) - test_days)])
        test = panel[panel['DATE'] >= start]
        results = {}

        scored = {'model': walk_forward_scores(panel, start)}
        for name, scorer in STRATEGIES.items():
            scored[name] = scorer(test)

        for name, frame in scored.items():
            if frame.empty:
                logger.warning(f'{name}: недостаточно данных для бэктеста')
                continue
            trades = cls.trades(frame)
            stats = cls.metrics(trades, test['DATE'])
            results[name] = {'stats': stats, 'trades': trades}
            logger.info(
                f'{name}: сделок {stats["TRADES"]}, win {stats["WIN_RATE"]:.1%}, '
                f'средняя {stats["AVG_TRADE"]:.2%}, итог {stats["TOTAL_RETURN"]:.1%}, '
                f'просадка {stats["MAX_DRAWDOWN"]:.1%}, Sharpe {stats["SHARPE"]:.2f}'
            )
            if save:
                cls.append_data({
                    'CREATED_AT': datetime.now(),
                    'STRATEGY': name,
                    'PERIOD_START': start.date(),
                    'PERIOD_END': pd.Timestamp(dates[-1]).date(),
                    'PARAMS': json.dumps({'hold_days': HOLD_DAYS, 'max_signals': MAX_SIGNALS}),
                    **stats,
                }, tables.BacktestRun)
        return results
