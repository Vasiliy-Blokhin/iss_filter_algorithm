import logging
from time import sleep
from elements.algorithm.worker import Algorithm as a
from elements.statistic.worker import Statistic as s
from elements.weights.worker import WEIGHTS as w
from elements.notification.worker import TelegramNotification as tlg
from source.settings.settings import (
    handler,
    TIME_UPDATE,
    SET_ITERATION,
    START_VALUE,
    NULL_DATA_ERROR,
    END_INTERATION_MESSAGE,
    ERROR_MESSAGE,
    EMPTY_STATISTIC_MESSAGE,
    DAILY_EXP_MOV_MESSAGE
)
from source.settings.exceptions import NullData
# Запуск логгера.
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


if __name__ == '__main__':
    counter = START_VALUE
    flag_prepare_data = True
    flag_daily_exp_mov_aver = True
    cur_day = a.curent_data()['day']
    a.create_all_tables()
    while True:
        try:
            if a.is_not_work_time():
                logger.info(f'is not work time {a.curent_msc_time()}')
                continue
            logger.info(f'work time: {a.curent_msc_time()}')

            a.data_prepare()
            logger.info('data prepare success')
            if a.is_trade_time():
                if flag_daily_exp_mov_aver:
                    a.exp_mov_aver_daily_counting()
                    flag_daily_exp_mov_aver = False
                    tlg.send_message(text=DAILY_EXP_MOV_MESSAGE)
                if cur_day != a.curent_data()['day']:
                    cur_day = a.curent_data()['day']
                    flag_daily_exp_mov_aver = True

                if a.data_filter() == NULL_DATA_ERROR:
                    flag_prepare_data = True
                    counter = START_VALUE
                    continue
                logger.info('data filter success')

                if flag_prepare_data:
                    s.prepare_start_data()
                    flag_prepare_data = False
                    logger.info('prepare start data success')

                counter += 1
                if counter >= SET_ITERATION:
                    flag_prepare_data = True
                    counter = START_VALUE

                    w.weights_correct()
                    logger.info('weights counter success')
                    if not s.result_statistic():
                        raise NullData
                    tlg.send_message(text=END_INTERATION_MESSAGE)

            else:
                flag_prepare_data = True
                counter = START_VALUE
        except NullData:
            tlg.send_message(text=EMPTY_STATISTIC_MESSAGE)
        except Exception as error:
            tlg.send_message(text=ERROR_MESSAGE + error)
            logger.error(error)
        finally:
            logger.info(f'Итерация # {counter} окончена')
            sleep(TIME_UPDATE)
