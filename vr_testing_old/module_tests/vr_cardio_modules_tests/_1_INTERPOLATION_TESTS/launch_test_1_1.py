"""
Author: SpikaTech
Date: 24/10/2023
Description:
____________
TEST 1_1.

This test focuses on evaluating interpolation performance by not considering an electrode recording 
and calculating the interpolation at that position. Subsequently, the actual and estimated signals are compared.
This process is carried out for all 30 electrodes and the average is calculated.

"""

import sys

import interpolate_n
from loguru import logger
from n_means import save_averages_file

sys.path.append("src/VRCARDIO-Testing/TESTS")
import common_tools


def run_one_perfect_case():
    """
    Function that runs just one perfect case as an example.

    params:none
    returns: none
    """
    session = common_tools.get_subset(1, 5)[0]
    case = common_tools.load_case(session)
    interpolate_n.one_session_stats(case)


def run_selection(number_sessions, bad_electrodes_number):
    """
    Function that generates a selection of cases to run the experiments over them.

    params:
        - number_sessions (int): number of sessions to run.
        - bad_electrodes_number (int): number of MAX wrong electrodes to be included.
        - ecg_das (bool): Boolean to filter the cases within all the electrodes necesary
                          to run the experiment (recomended)

    returns: none
    """
    sessions = common_tools.get_subset(number_sessions, bad_electrodes_number)
    for count, i in enumerate(sessions):
        case = common_tools.load_case(i)
        interpolate_n.one_session_stats(case)
        logger.info("Tested: " + str(count + 1) + " / " + str(len(sessions)))


def create_results_excell():
    """
    Function that generares a summary excell with the results of the
    experiments runned.

    Params: none
    Returns: none
    """
    save_averages_file()


def launch_test_1_1():
    logger.info("Running test 1.1 : Interpolation Cross validation")
    run_selection(400, 15)
    create_results_excell()


if __name__ == "__main__":
    logger.info("Run test")
    # run_one_perfect_case()
    run_selection(400, 15)
    create_results_excell()
