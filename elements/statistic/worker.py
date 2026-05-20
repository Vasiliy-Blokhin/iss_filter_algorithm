import logging

from source.json_worker.worker import JSONSaveAndRead
from elements.notification.worker import TelegramNotification as tlg
from source.sql.main import SQLmain
import source.sql.tables as tables
from source.settings.settings import (
    STATUS_UP,
    handler,
    COMISSION_COEFF,
)

# Подключение логгера.
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class Statistic(JSONSaveAndRead, SQLmain):
    """Сбор статистики о прогнозе."""
    def __init__(self, url: str | None, file: str, json_classes, json_all_data) -> None:
        super().__init__(url, file)
        self.json_classes = None
        self.json_all_data = json_all_data

    @classmethod
    def prepare_start_data(self):
        try:
            self.insert_data(
                data=self.get_all_data_with_sort_score(
                    table=tables.CurrentScore
                )[0:5],
                table=tables.StartScore
            )
        except Exception as error:
            logger.error(error)

    @classmethod
    def make_statistic(self):
        # Расчет потенциального дохода.
        count_positive = 0
        count_neutral = 0
        count_all = 0
        count_price_before = 0
        count_price_after = 0

        for start in self.get_all_data(tables.StartScore):
            try:
                current = self.get_share_on_secid(
                    table=tables.FilterData,
                    secid=start['SECID']
                )[0]

                if (
                    start['LAST'] is None or start['LAST'] == 0
                    or current['LAST'] is None or current['LAST'] == 0
                ):
                    continue

                if current['LAST'] > start['LAST']:
                    count_positive += 1
                elif current['LAST'] == start['LAST']:
                    count_neutral += 1

                count_all += 1
                count_price_before += float(start['LAST']) * float(start['LOTSIZE'])
                count_price_after += float(current['LAST']) * float(current['LOTSIZE'])

            except Exception as e:
                logger.error(f'{e}')
                continue

        if count_all == 0:
            return {
                'statistic_prcnt': 0,
                'neutral_prcnt': 0,
                'potential_profitability': 0,
                'count_price_after': 0
            }

        statistic_prcnt = float(100 * count_positive / count_all)
        neutral_prcnt = float(100 * count_neutral / count_all)

        comission = count_price_after * COMISSION_COEFF
        potential_profitability = (
            count_price_after - count_price_before - comission
        )
        if potential_profitability > 0:
            potential_profitability *= 0.87

        return {
            'statistic_prcnt': statistic_prcnt,
            'neutral_prcnt': neutral_prcnt,
            'potential_profitability': potential_profitability,
            'count_price_after': count_price_after
        }

    @staticmethod
    def is_null_result(data):
        if (
            data['count_price_after'] == 0
        ):
            return True
        return False

    @classmethod
    def show_statistic(self):
        data = self.get_all_data(table=tables.AllStatistic)

        prof_prcnt = 0
        neutral_prcnt = 0
        pot_prob = 0
        all_price = 0
        for el in data:
            prof_prcnt += el['statistic_prcnt']
            neutral_prcnt += el['neutral_prcnt']
            pot_prob += el['potential_profitability']
            all_price += el['count_price_after']
        prof_prcnt /= len(data)
        neutral_prcnt /= len(data)
        pot_prob /= len(data)
        all_price /= len(data)

        logger.info(
        '\nПроцент дохода/убытка: '
        f'{(100*pot_prob/all_price):.2f} %\n'
        'Процент доходных сделок: '
        f'{prof_prcnt:.2f} %\n'
        'Процент убыточных сделок: '
        f'{(100 - prof_prcnt - neutral_prcnt):.2f} %\n'
        'Количество сделок: '
        f'{len(data)} шт'           
        )


    @classmethod
    def result_statistic(self):
        try:
            current_statistic = self.make_statistic()
            statistic_prcnt = round(
                100 * current_statistic[
                    'potential_profitability'
                ] / current_statistic['count_price_after'], 2
            )
            #if (
            #    self.is_null_result(current_statistic)
            #    or abs(statistic_prcnt) / 100 == COMISSION_COEFF
            #):
            #    return False
            logger.info(f'current stat - {current_statistic}')
            self.append_data(
                data=current_statistic,
                table=tables.AllStatistic
            )
            return True
        except ZeroDivisionError:
            return False
        except Exception as error:
            logger.error(error)
            return False
