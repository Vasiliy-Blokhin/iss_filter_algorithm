import logging
from time import sleep
from random import randrange

from source.json_worker.worker import JSONSaveAndRead
from source.settings.settings import (
    handler,
    WEIGHTS_PARAM,
    MAX,
    MED,
    MIN
)
from source.settings.module import interp_4_dote
from source.sql.main import SQLmain
import source.sql.tables as tables


# Подключение логгера.
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class WEIGHTS(JSONSaveAndRead, SQLmain):

    @classmethod
    def weights_correct(self):
        start_data = self.get_all_data(tables.StartScore)
        weights = self.get_all_data(tables.Weights)[0]

        min_delta_weight = self.get_rand_weight(MIN)
        max_delta_weight = self.get_rand_weight(MAX)
        med_delta_weight = self.get_rand_weight(MED)
        for start_share in start_data:
            try:
                end_share = self.get_share_on_secid(
                    table=tables.CurrentScore,
                    secid=start_share['SECID']
                )[0]

                weights_dict = self.weights_correct_body(
                    start_share,
                    end_share,
                    weights,
                    min_delta_weight,
                    max_delta_weight,
                    med_delta_weight,
                )
                if weights_dict == {}:
                    continue
                self.insert_data(weights_dict, tables.Weights)
            except ZeroDivisionError:
                continue
            except Exception as error:
                logger.error(error)

    @classmethod
    def weights_correct_body(
        self,
        start_share,
        end_share,
        weights,
        min_delta_weight,
        max_delta_weight,
        med_delta_weight,
    ):
        weights_dict = {}
        for weights_name in WEIGHTS_PARAM:
            weights_ratio = (
                    start_share[
                        weights_name + '_CUR'
                    ] / start_share[weights_name + '_MAX']
                )
            if start_share['LAST'] < end_share['LAST']:
                if weights_ratio > 0.7:
                    weights_dict[weights_name] = (
                        weights[weights_name] + max_delta_weight
                    )
                elif weights_ratio < 0.4:
                    weights_dict[weights_name] = (
                        weights[weights_name] + min_delta_weight
                    )
                else:
                    weights_dict[weights_name] = (
                        weights[weights_name] + med_delta_weight
                    )
            elif start_share['LAST'] > end_share['LAST']:
                if weights_ratio > 0.7:
                    weights_dict[weights_name] = (
                        weights[weights_name] + min_delta_weight
                    )
                elif weights_ratio < 0.4:
                    weights_dict[weights_name] = (
                        weights[weights_name] + max_delta_weight
                    )
                else:
                    weights_dict[weights_name] = (
                        weights[weights_name] + med_delta_weight
                    )
            else:
                weights_dict[weights_name] = (
                        weights[weights_name] + med_delta_weight
                    )
        return weights_dict

    # Значения для корректировки весов.
    @classmethod
    def get_rand_weight(self, values: list[int]):
        start = values[0]
        end = values[1]
        result = randrange(start=start, stop=end) / 3000
        return result * self.weight_growth_rate()

    @classmethod
    def counting_all_statistic(self):
        try:
            data = self.get_all_data(tables.AllStatistic)
            all_success = 0
            all_profit = 0
            for el in data:
                try:
                    all_success += el['statistic_prcnt']
                    all_profit += (100 * el[
                        'potential_profitability'
                    ] / el['count_price_after'])
                except ZeroDivisionError:
                    if len(data) == 0:
                        return [0, 0]
                    else:
                        continue
            prcnt_statistic = all_success / len(data)
            median_profit = all_profit / len(data)
            return [prcnt_statistic, median_profit]
        except ZeroDivisionError:
            return [0, 0]

    @classmethod
    def correct_value(self, value):
        if value > 0 and value < 0.5:
            return -0.55
        elif value < 0 and value > -0.5:
            return 0.55

        return value

    @classmethod
    def weight_growth_rate(self):
        try:
            result_statistic = self.counting_all_statistic()
            if result_statistic == [0, 0]:
                return 1

            prcnt_rate = result_statistic[0]
            prcnt_profit = result_statistic[1]

            speed_coeff = interp_4_dote(
                dote_prcnt=prcnt_rate,
                point_limits=[0.5, 5],
                prcnt_limits=[85, 0],
                prcnt_start_limit=0
            )

            if prcnt_profit < 0:
                speed_coeff += 0.5
                speed_coeff *= 1.2
            elif prcnt_profit > 0.05:
                speed_coeff *= 0.5

            return speed_coeff
        except Exception:
            return True
