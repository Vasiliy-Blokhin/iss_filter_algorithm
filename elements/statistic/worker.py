import logging

from source.json_worker.worker import JSONSaveAndRead
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
        self.insert_data(
            data=self.get_all_data(table=tables.CurrentScore),
            table=tables.StartScore
        )

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
                    table=tables.CurrentScore,
                    secid=start['SECID']
                )[0]

                if start['SECID'] == current['SECID']:
                    if (
                        start['LCURRENTPRICE'] is None
                        or start['LCURRENTPRICE'] == 0
                    ):
                        continue

                    if current['STATUS_FILTER'] == STATUS_UP:
                        if current['LCURRENTPRICE'] > start['LCURRENTPRICE']:
                            count_positive += 1
                        elif current['LCURRENTPRICE'] == start['LCURRENTPRICE']:
                            count_neutral += 1

                        count_price_before += start['LCURRENTPRICE'] * start['LOTSIZE']
                        count_price_after += current['LCURRENTPRICE'] * current['LOTSIZE']
                        count_all += 1

            except Exception:
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
            data['potential_profitability'] == 0
            or data['statistic_prcnt'] == 0
            or data['count_price_after'] == 0
            or data['neutral_prcnt'] == 0
        ):
            return True
        return False

    @classmethod
    def result_statistic(self):
        try:
            current_statistic = self.make_statistic()
            logger.info(f'current stat - {current_statistic}')
            statistic_prcnt = round(
                100 * current_statistic[
                    'potential_profitability'
                ] / current_statistic['count_price_after'], 2
            )
            if (
                self.is_null_result(current_statistic)
                or abs(statistic_prcnt) / 100 == COMISSION_COEFF
            ):
                return False
            self.append_data(
                data=current_statistic,
                table=tables.AllStatistic
            )
        except ZeroDivisionError:
            return False
        except Exception as error:
            logger.error(error)
