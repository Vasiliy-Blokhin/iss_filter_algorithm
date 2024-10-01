import logging
from time import sleep
from elements.algorithm.worker import Algorithm as a
from elements.statistic.worker import Statistic as s
from elements.weights.worker import WEIGHTS as w
from source.settings.settings import handler, TIME_UPDATE, SET_ITERATION, START_VALUE

# Запуск логгера.
logger = logging.getLogger(name=__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


if __name__ == '__main__':
    counter = START_VALUE
    flag_prepare_data = True
    a.create_all_tables()
    while True:

        a.data_prepare()
        if True:
            a.data_filter()

            if flag_prepare_data:
                s.prepare_start_data()
                flag_prepare_data = False

            counter += 1
            if counter >= SET_ITERATION:
                w.weights_correct()
                s.result_statistic()

                flag_prepare_data = True
                counter = START_VALUE
        else:
            flag_prepare_data = True
            counter = START_VALUE

        logger.info(f'Итерация # {counter} окончена')
        sleep(TIME_UPDATE)
