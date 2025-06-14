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
    SIZE_STAT_BASE,
    TIC_IC_POINTS
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
                tic_ic_max = TIC_IC_POINTS * weight.get('TIC_IC')

                max_weights = sum(
                    [wptpwp_max, lcp_max, pmpwp_max, tic_ic_max,
                        lctlw_max, lcprcnt_max, lmp_max]
                )

                param_score['WPTPWP_MAX'] = wptpwp_max
                param_score['LCP_MAX'] = lcp_max
                param_score['PMPWP_WP_MAX'] = pmpwp_max
                param_score['LCTLWP_WP_MAX'] = lctlw_max
                param_score['LCPRCNT_MAX'] = lcprcnt_max
                param_score['LMP_MAX'] = lmp_max
                param_score['TIC_IC_MAX'] = tic_ic_max
# start__________________________________________________________

                wptpwp = -interp_4_dote(
                    dote_prcnt=share['WAPTOPREVWAPRICEPRCNT'],
                    point_limits=[-wptpwp_max, wptpwp_max],
                    prcnt_limits=[-1.62, 1.62],
                    prcnt_start_limit=0.12
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
                    prcnt_limits=[-1.6, -0.8, 0.8, 1.6],
                    prcnt_start_limit=0.05
                ) * weight['PMPWP_WP']
                current_score += pmpwp_wp
                param_score['PMPWP_WP_CUR'] = pmpwp_wp
# _______________________________________________________________

                lctlwp_wp = -interp_6_dote(
                    dote_prcnt=share['LASTCNGTOLASTWAPRICE']/share['WAPRICE'],
                    point_limits=[-lctlw_max, lctlw_max],
                    prcnt_limits=[-1.6, -0.8, 0.8, 1.6],
                    prcnt_start_limit=0.05
                ) * weight['LCTLWP_WP']
                current_score += lctlwp_wp
                param_score['LCTLWP_WP_CUR'] = lctlwp_wp

# _______________________________________________________________

                lcprcnt = interp_4_dote(
                    dote_prcnt=share['LASTCHANGEPRCNT'],
                    point_limits=[-lcprcnt_max, lcprcnt_max],
                    prcnt_limits=[-1, 1],
                    prcnt_start_limit=0.1
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

                capitalization_diff = (
                    share['TRENDISSUECAPITALIZATION']
                    / share['ISSUECAPITALIZATION']
                )
                tic_ic = interp_4_dote(
                    dote_prcnt=capitalization_diff,
                    point_limits=[-tic_ic_max, tic_ic_max],
                    prcnt_limits=[-1.65, 1.65],
                    prcnt_start_limit=0.1
                ) * weight['TIC_IC']
                current_score += tic_ic
                param_score['TIC_IC_CUR'] = tic_ic
# end____________________________________________________________

                share['FILTER_SCORE'] = (
                    (
                        100 * current_score / max_weights
                    )
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

    @classmethod
    def save_activity_time(self):
        self.insert_data(
            table=tables.Activity,
            data=[{'last_time': self.curent_msc_time()}]
        )

    @classmethod
    def delete_old_stat_base(self):
        try:
            if len(self.get_all_data(
                table=tables.AllStatistic
            )) > SIZE_STAT_BASE:
                last_id = int(self.get_all_data(
                    table=tables.AllStatistic
                )[-1]['id'])

                self.delete_stat(
                    table=tables.AllStatistic,
                    id=last_id-SIZE_STAT_BASE
                )
            logger.info(
                f'''len - {len(self.get_all_data(
                table=tables.AllStatistic
            )) > SIZE_STAT_BASE}\n
                last id - {last_id}\n
                new id - {self.get_all_data(
                    table=tables.AllStatistic
                )[-1]['id']}\n'''
            )
        except Exception as error:
            logger.error(error)
