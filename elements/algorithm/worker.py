""" Описание класса, для работы с БД."""

import logging
from datetime import datetime
import pytz

from source.json_worker.worker import JSONSaveAndRead
from source.settings.settings import (
    handler,
    STATUS_UP,
    STATUS_DOWN,
    STATUS_MEDIUM,
    MAX_SCORE,
    WPTPWP_POINTS,
    LCP_POINTS,
    PMPWP_WP_POINTS,
    LCTLWP_WP_POINTS,
    LCPRCNT_POINTS,
    ABSTRACT_COEFF,
    BORDERS_SCORE,
    LMP_POINTS,
    STOP_TRADING,
    SHARE_GROUPS,
    TYPE_DATA_IMOEX,
    NEEDFUL,
    RUN_TRADING,
    IMOEX_URL,
    STATISTIC_NEED,
    NULL_DATA_ERROR,
    SPLYT_SYMB,
    EMA_POINTS
)
from source.settings.module import interp_4_dote, interp_6_dote
from source.settings.exceptions import NullData
from source.sql.main import SQLmain
import source.sql.tables as tables


# Запуск логгера.
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class Algorithm(JSONSaveAndRead, SQLmain):
    """ Класс для работы с API imoex (iss)."""
    @classmethod
    def data_filter(self):
        """
        Функиция для обработки даных по заданному алгоритму
        и сохранения данных.
        """
        shares_data = self.get_all_data(tables.PrepareData)
        weight_data = self.get_all_data(tables.Weights)
        coeff_result = ABSTRACT_COEFF * 100 / MAX_SCORE

        param_score_list = []
        data = []

        for share in shares_data:
            try:
                if (
                    share['LAST'] == 0
                    or share['LAST'] is None
                ):
                    continue

                weight = weight_data[0]
                current_score = 0
                param_score = {}

                # Баллы по параметрам алгоритма.
                wptpwp_max = WPTPWP_POINTS * weight.get('WPTPWP')
                lcp_max = LCP_POINTS * weight.get('LCP')
                pmpwp_max = PMPWP_WP_POINTS * weight.get('PMPWP_WP')
                lctlw_max = LCTLWP_WP_POINTS * weight.get('LCTLWP_WP')
                lcprcnt_max = LCPRCNT_POINTS * weight.get('LCPRCNT')
                lmp_max = LMP_POINTS * weight.get('LMP')
                ema_max = EMA_POINTS * weight.get('EMA')

                max_weights = sum(
                    [wptpwp_max, lcp_max, pmpwp_max, ema_max,
                        lctlw_max, lcprcnt_max, lmp_max]
                )

                param_score['WPTPWP_MAX'] = wptpwp_max
                param_score['LCP_MAX'] = lcp_max
                param_score['PMPWP_WP_MAX'] = pmpwp_max
                param_score['LCTLWP_WP_MAX'] = lctlw_max
                param_score['LCPRCNT_MAX'] = lcprcnt_max
                param_score['LMP_MAX'] = lmp_max
                param_score['EMA_MAX'] = ema_max

# start__________________________________________________________

                wptpwp = -interp_4_dote(
                    dote_prcnt=share['WAPTOPREVWAPRICEPRCNT'],
                    point_limits=[-wptpwp_max, wptpwp_max],
                    prcnt_limits=[-3, 3],
                    prcnt_start_limit=0.2
                )
                current_score += wptpwp
                param_score['WPTPWP_CUR'] = wptpwp

# _______________________________________________________________

                if share['LCURRENTPRICE'] >= share['LAST']:
                    lcp = lcp_max * weight['LCP']
                    current_score += lcp
                    param_score['LCP_CUR'] = lcp
                elif share['LCURRENTPRICE'] < share['LAST']:
                    lcp = lcp_max * weight['LCP']
                    current_score -= lcp
                    param_score['LCP_CUR'] = -lcp
                else:
                    param_score['LCP_CUR'] = 0

# _______________________________________________________________

                pmpwp_wp = -interp_6_dote(
                    dote_prcnt=share['PRICEMINUSPREVWAPRICE']/share['WAPRICE'],
                    point_limits=[-pmpwp_max, pmpwp_max],
                    prcnt_limits=[-3, -1.5, 1.5, 3],
                    prcnt_start_limit=0.2
                ) * weight['PMPWP_WP']
                current_score += pmpwp_wp
                param_score['PMPWP_WP_CUR'] = pmpwp_wp
# _______________________________________________________________

                lctlwp_wp = -interp_6_dote(
                    dote_prcnt=share['LASTCNGTOLASTWAPRICE']/share['WAPRICE'],
                    point_limits=[-lctlw_max, lctlw_max],
                    prcnt_limits=[-2.5, 1, -1, 2.5],
                    prcnt_start_limit=0.1
                ) * weight['LCTLWP_WP']
                current_score += lctlwp_wp
                param_score['LCTLWP_WP_CUR'] = lctlwp_wp

# _______________________________________________________________

                lcprcnt = interp_4_dote(
                    dote_prcnt=share['LASTCHANGEPRCNT'],
                    point_limits=[-lcprcnt_max, lcprcnt_max],
                    prcnt_limits=[-1, 1],
                    prcnt_start_limit=0.05
                ) * weight['LCPRCNT']
                current_score += lcprcnt
                param_score['LCPRCNT_CUR'] = lcprcnt

# _______________________________________________________________
                lmp = lmp_max * weight['LMP']
                if (
                    share.get('LAST') > share.get('MARKETPRICE')
                ):
                    current_score -= lmp
                    param_score['LMP_CUR'] = -lmp
                elif (
                    share.get('LAST') < share.get('MARKETPRICE')
                ):
                    current_score += lmp
                    param_score['LMP_CUR'] = lmp
                else:
                    param_score['LMP_CUR'] = 0
# _______________________________________________________________
                exp_mov_aver_data = self.get_share_on_secid(
                    table=tables.ExpMovAverages,
                    secid=share['SECID']
                )[0]
                ema = ema_max * weight['EMA']

                pw3 = (
                    exp_mov_aver_data['day_1'] +
                    exp_mov_aver_data['day_2'] +
                    exp_mov_aver_data['day_3']
                ) / 3
                pw5 = (
                    exp_mov_aver_data['day_3'] +
                    exp_mov_aver_data['day_4'] +
                    exp_mov_aver_data['day_5']
                ) / 3
                pw8 = (
                    exp_mov_aver_data['day_5'] +
                    exp_mov_aver_data['day_6'] +
                    exp_mov_aver_data['day_7'] +
                    exp_mov_aver_data['day_8']
                ) / 4

                if (
                    pw3 > pw5 and pw3 > pw8
                    and share['TRENDISSUECAPITALIZATION'] > 0
                ):
                    current_score += ema / 2
                    param_score['EMA_CUR'] = ema / 2
                elif (
                    pw3 < pw5 and pw3 < pw8
                    and share['TRENDISSUECAPITALIZATION'] < 0
                ):
                    current_score -= ema / 2
                    param_score['EMA_CUR'] = -ema / 2
                elif (
                    pw5 > pw3 and pw5 > pw8
                    and share['TRENDISSUECAPITALIZATION'] < 0
                ):
                    current_score -= ema
                    param_score['EMA_CUR'] = -ema
                elif (
                    pw5 < pw3 and pw5 < pw8
                    and share['TRENDISSUECAPITALIZATION'] > 0
                ):
                    current_score += ema
                    param_score['EMA_CUR'] = -ema
                else:
                    param_score['EMA_CUR'] = 0
# end____________________________________________________________

                share['FILTER_SCORE'] = (
                    (
                        100 * current_score / max_weights
                    ) * coeff_result
                )
                if share['FILTER_SCORE'] >= BORDERS_SCORE:
                    share['STATUS_FILTER'] = STATUS_UP

                    for key, value in share.items():
                        if key in STATISTIC_NEED:
                            param_score[key] = value
                    param_score_list.append(param_score)

                elif share['FILTER_SCORE'] <= -BORDERS_SCORE:
                    share['STATUS_FILTER'] = STATUS_DOWN

                else:
                    share['STATUS_FILTER'] = STATUS_MEDIUM

                data.append(share)
            except Exception:
                continue

        try:
            if data == [] or param_score_list == []:
                raise NullData
            self.insert_data(
                data=data,
                table=tables.FilterData
            )
            self.insert_data(
                data=param_score_list,
                table=tables.CurrentScore
            )
        except NullData:
            logger.error(
                'filter data or params score is null'
            )
            return NULL_DATA_ERROR
        except Exception as error:
            logger.error(error)

    @classmethod
    def is_trade_time(self):
        if self.get_all_data(
            table=tables.PrepareData
        )[0]['TRADINGSESSION'] == STOP_TRADING:
            logger.info('Торги приостановлены, работа не ведется')
            return False
        return True

    @staticmethod
    def curent_msc_time():
        timezone = 'Europe/Moscow'
        current_msc_time_h = datetime.now(pytz.timezone(timezone)).hour
        current_msc_time_m = datetime.now(pytz.timezone(timezone)).minute
        return current_msc_time_h + current_msc_time_m / 100

    @staticmethod
    def curent_data():
        date = '{:%d-%m-%y}'.format(datetime.now()).split(SPLYT_SYMB)
        return {
            'day': date[0],
            'month': date[1],
            'year': date[2]
        }

    @classmethod
    def is_not_work_time(self, current_time=None):
        if current_time is None:
            current_time = self.curent_msc_time()

        if (
            (current_time < 9 and current_time > 23)
        ):
            return True

        return False

# prepare data ________________________________________

    @classmethod
    def sorted_data(self, data):
        secid_list = []
        for share in data:
            secid = share.get('SECID')
            if secid in secid_list:
                data.remove(share)
                continue
            secid_list.append(secid)
        return data

    @classmethod
    def data_prepare(self):
        """ Фильтрация данных, полученных с запроса."""
        result = []
        data_list = []
        logger.info('get response info from ISS')
        # Фильтрация полученных данных (из разных "графов").
        for type_data in TYPE_DATA_IMOEX:
            for element in self.get_api_response(
                url=IMOEX_URL
            )[1][type_data]:
                # Оставляет только акции.
                if (
                    type_data == 'securities'
                    and element.get('INSTRID') != 'EQIN'
                ):
                    continue
                if element.get('BOARDID') not in SHARE_GROUPS:
                    continue
                new_dict = {}
                # Добавление только необходимых параметров из списка.
                for key, value in element.items():
                    if key in NEEDFUL:
                        new_dict[key] = value
                data_list.append(new_dict)
            result.append(data_list)

        self.insert_data(
            data=self.union_api_response(*result),
            table=tables.PrepareData
        )

    @classmethod
    def union_api_response(self, data_sec, data_md):
        """ Добавляет доплнительные параметры и сводит всё в одну БД."""
        result = []
        for el_sec in data_sec:
            for el_md in data_md:
                # Сведение БД в одну.
                if el_sec['SECID'] == el_md['SECID']:
                    el_sec.update(el_md)
            # Добавление новых параметров.
            if el_sec['TRADINGSESSION'] is None:
                el_sec['TRADINGSESSION'] = STOP_TRADING
            elif el_sec['TRADINGSESSION'] == '1':
                el_sec['TRADINGSESSION'] = RUN_TRADING

            if el_sec['CURRENCYID'] == 'SUR':
                el_sec['CURRENCYID'] = 'рубль'

            format = '%H:%M:%S (%d.%m)'
            el_sec['DATAUPDATE'] = (
                datetime.now(pytz.timezone('Europe/Moscow'))
            ).strftime(format)

            result.append(el_sec)
        # Проверка и вывод выходных данных.

        return self.sorted_data(result)

# __________________________________________________________
# экспоненциальные скользящие средние 3, 5, 8 дней.

    @classmethod
    def exp_mov_aver_daily_counting(self):
        new_list = []
        for share in self.get_all_data(table=tables.PrepareData):
            data = self.get_share_on_secid(
                table=tables.ExpMovAverages,
                secid=share['SECID']
            )

            new_dict = {}
            new_dict['SECID'] = share['SECID']

            if data == []:
                new_dict['SECID'] = share['SECID']
                for index in range(8, 0, -1):
                    new_dict['day_' + str(index)] = share['PREVWAPRICE']
                new_list.append(new_dict)
                continue

            for index in range(8, 0, -1):
                if (
                    data[0]['day_' + str(index)] is None
                    or index == 1
                ):
                    new_dict['day_' + str(index)] = share['PREVWAPRICE']
                    continue
                new_dict['day_' + str(index)] = data[0]['day_' + str(index - 1)]
            new_list.append(new_dict)

        self.insert_data(data=new_list, table=tables.ExpMovAverages)
# __________________________________________________________
