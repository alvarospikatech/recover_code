"""
Author: SpikaTech
Date: 24/10/2023
Description:
_________
TEST 1_0.

This test compares the ECG obtained by DAS with other sintetic ECG calculated 
using the real positions where the electrodes should be positioned.

"""

import sys

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests/_1_INTERPOLATION_TESTS")
from ecg_interpolation import one_session_stats
from loguru import logger

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests")
import common_tools


def run_one_perfect_case():
    """
    Function that runs just one perfect case as an example.

    params:none

    returns: none
    """
    session = common_tools.get_subset(1, 10, True)[0]
    case = common_tools.load_case(session)
    one_session_stats(case)


def run_selection(number_sessions, bad_electrodes_number, ecg_das=True):

    """
    Function that generates a selection of cases to run the experiments over them.

    params:
        - number_sessions (int): number of sessions to run.
        - bad_electrodes_number (int): number of MAX wrong electrodes to be included.
        - ecg_das (bool): Boolean to filter the cases within all the electrodes necesary
                          to run the experiment (recomended)

    returns: none
    """

    sessions = common_tools.get_subset(number_sessions, bad_electrodes_number, ecg_das)
    for i in sessions:
        case = common_tools.load_case(i)
        one_session_stats(case)


def launch_test_1_0():
    logger.info("Runing test 1_0: Interpolation with sintetic ECGs")
    run_selection(400, 15, True)


if __name__ == "__main__":

    logger.info("Run test")
    # run_one_perfect_case()
    run_selection(400, 15, True)

    # one_session_stats(case)
