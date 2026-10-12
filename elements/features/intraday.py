""" Дневные признаки из часовых свечей и дивидендов.

Сессии Мосбиржи (МСК): утренняя 07:00–10:00, основная 10:00–18:50, вечерняя 19:00–23:50.
Свеча часа H покрывает H:00–H:59.
"""
import numpy as np
import pandas as pd


INTRADAY_FEATURES = [
    'main_ret', 'morning_ret', 'evening_ret', 'first_hour_ret', 'last_hours_ret',
    'vwap_dist', 'main_close_pos', 'evening_share', 'hour_vol', 'last_hour_share',
]
DIVIDEND_FEATURES = ['div_days', 'div_yield', 'div_after_days']

MAIN_OPEN_HOUR, MAIN_CLOSE_HOUR, EVENING_HOUR = 10, 18, 19


def session_frame(hours: pd.DataFrame) -> pd.DataFrame:
    """ Признаки и цены сессий по каждой бумаге и дню."""
    data = hours.copy()
    data['DATE'] = data['BEGIN'].dt.normalize()
    data['HOUR'] = data['BEGIN'].dt.hour
    data['log_ret'] = np.log(data['CLOSE'] / data['OPEN'])
    keys = ['SECID', 'DATE']
    grouped = data.groupby(keys)

    main = data[(data['HOUR'] >= MAIN_OPEN_HOUR) & (data['HOUR'] <= MAIN_CLOSE_HOUR)]
    main_grouped = main.groupby(keys)
    result = pd.DataFrame({
        'day_open': grouped['OPEN'].first(),
        'day_close': grouped['CLOSE'].last(),
        'value_total': grouped['VALUE'].sum(),
        'volume_total': grouped['VOLUME'].sum(),
        'main_open': main_grouped['OPEN'].first(),
        'main_close': main_grouped['CLOSE'].last(),
        'main_high': main_grouped['HIGH'].max(),
        'main_low': main_grouped['LOW'].min(),
        'hour_vol': main_grouped['log_ret'].std(),
    })
    first_hour = data[data['HOUR'] == MAIN_OPEN_HOUR].set_index(keys)
    result['first_hour_ret'] = first_hour['CLOSE'] / first_hour['OPEN'] - 1
    last_hours = data[data['HOUR'].isin([MAIN_CLOSE_HOUR - 1, MAIN_CLOSE_HOUR])].groupby(keys)
    result['last_hours_ret'] = last_hours['CLOSE'].last() / last_hours['OPEN'].first() - 1
    result['last_hour_share'] = (
        data[data['HOUR'] == MAIN_CLOSE_HOUR].groupby(keys)['VALUE'].sum() / result['value_total']
    )
    evening = data[data['HOUR'] >= EVENING_HOUR].groupby(keys)['VALUE'].sum()
    result['evening_share'] = (evening / result['value_total']).fillna(0)

    result['main_ret'] = result['main_close'] / result['main_open'] - 1
    result['morning_ret'] = result['main_open'] / result['day_open'] - 1
    result['evening_ret'] = result['day_close'] / result['main_close'] - 1
    vwap = result['value_total'] / result['volume_total'].replace(0, np.nan)
    result['vwap_dist'] = result['day_close'] / vwap - 1
    result['main_close_pos'] = (
        (result['main_close'] - result['main_low'])
        / (result['main_high'] - result['main_low']).replace(0, np.nan)
    ).fillna(0.5)
    return result.reset_index()


def dividend_features(panel: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """ Дни до ближайшей отсечки (известной на дату), доходность дивиденда, дни после отсечки.

    Используются только дивиденды, объявленные не позже даты строки (без заглядывания вперёд).
    Последний день с правом на дивиденд при T+1 — рабочий день перед датой реестра.
    """
    panel = panel.copy()
    for column in DIVIDEND_FEATURES:
        panel[column] = np.nan
    if dividends.empty:
        return panel
    dividends = dividends.copy()
    dividends['REGISTRY_DATE'] = pd.to_datetime(dividends['REGISTRY_DATE'])
    dividends['ANNOUNCE_DATE'] = pd.to_datetime(dividends['ANNOUNCE_DATE'])
    # Неизвестная дата объявления — считаем, что объявлен за 30 дней до реестра.
    dividends['ANNOUNCE_DATE'] = dividends['ANNOUNCE_DATE'].fillna(
        dividends['REGISTRY_DATE'] - pd.Timedelta(days=30)
    )
    dividends['LAST_CUM'] = pd.to_datetime(np.busday_offset(
        dividends['REGISTRY_DATE'].to_numpy().astype('datetime64[D]'), -1, roll='backward'
    ))

    parts = []
    for secid, rows in panel.groupby('SECID', sort=False):
        own = dividends[dividends['SECID'] == secid]
        if own.empty:
            parts.append(rows)
            continue
        dates = rows['DATE'].to_numpy()
        days_to = np.full(len(rows), np.nan)
        yields = np.full(len(rows), np.nan)
        days_after = np.full(len(rows), np.nan)
        for div in own.itertuples():
            known = (dates >= np.datetime64(div.ANNOUNCE_DATE)) & (dates <= np.datetime64(div.LAST_CUM))
            distance = np.busday_count(
                dates.astype('datetime64[D]'), np.datetime64(div.LAST_CUM, 'D')
            ).astype(float)
            closer = known & (np.isnan(days_to) | (distance < days_to))
            days_to[closer] = distance[closer]
            yields[closer] = div.VALUE / rows['CLOSE'].to_numpy()[closer]
            after = np.busday_count(np.datetime64(div.LAST_CUM, 'D'), dates.astype('datetime64[D]'))
            recent = (after > 0) & (after <= 10)
            days_after[recent] = after[recent]
        rows = rows.copy()
        rows['div_days'] = days_to
        rows['div_yield'] = yields
        rows['div_after_days'] = days_after
        parts.append(rows)
    return pd.concat(parts)


def ex_dividend_cash(panel: pd.DataFrame, dividends: pd.DataFrame, tax: float = 0.13) -> pd.Series:
    """ Дивиденд (после НДФЛ), положенный при покупке в день t и удержании через отсечку.

    Возвращает по строке сумму на акцию, если LAST_CUM попадает в [t, t+hold-1] — решается
    в симуляции; здесь — словарь (SECID, LAST_CUM) -> выплата.
    """
    if dividends.empty:
        return pd.Series(dtype=float)
    dividends = dividends.copy()
    registry = pd.to_datetime(dividends['REGISTRY_DATE']).to_numpy().astype('datetime64[D]')
    dividends['LAST_CUM'] = pd.to_datetime(np.busday_offset(registry, -1, roll='backward'))
    return dividends.set_index(['SECID', 'LAST_CUM'])['VALUE'] * (1 - tax)
