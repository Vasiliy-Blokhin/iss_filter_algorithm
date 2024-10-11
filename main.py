import logging
from time import sleep
from elements.algorithm.worker import Algorithm as a
from elements.statistic.worker import Statistic as s
from elements.weights.worker import WEIGHTS as w
from source.settings.settings import (
    handler,
    TIME_UPDATE,
    SET_ITERATION,
    START_VALUE,
    NULL_DATA_ERROR
)
# Запуск логгера.
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


if __name__ == '__main__':
    counter = START_VALUE
    flag_prepare_data = True
    a.create_all_tables()
    while True:
        try:
            if a.is_not_work_time():
                continue

            a.data_prepare()
            logger.info('data prepare success')
            if a.is_trade_time():
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
                    w.weights_correct()
                    logger.info('weights counter success')
                    s.result_statistic()
                    logger.info('counting statistic success')

                    flag_prepare_data = True
                    counter = START_VALUE
            else:
                flag_prepare_data = True
                counter = START_VALUE
        except Exception as error:
            logger.error(error)
        finally:
            logger.info(f'Итерация # {counter} окончена')
            sleep(TIME_UPDATE)
