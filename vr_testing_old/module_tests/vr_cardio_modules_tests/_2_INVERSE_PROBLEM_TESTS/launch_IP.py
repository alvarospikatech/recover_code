"""
Author: SpikaTech
Date: 24/10/2023
Description:
____________
TEST 2_0.

Not really a test but launches all inverse's problem
"""

import sys

from loguru import logger

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests")
import common_tools
from _2_INVERSE_PROBLEM_TESTS.IP_code import run_inverse_problem


def run_one_perfect_case():
    """
    Function that runs just one perfect case as an example.

    params:none

    returns: none
    """
    session = common_tools.get_subset(1, 5)[0]
    logger.info(session)
    case = common_tools.load_case(session)
    run_inverse_problem(case)
    logger.info("Done")


def run_selection(number_sessions, bad_electrodes_number):
    """
    Function that generates a selection of cases to run the experiments over them.

    params:
        - number_sessions (int): number of sessions to run
        - bad_electrodes_number (int): number of MAX wrong electrodes to be included.

    returns: none
    """
    sessions = common_tools.get_subset(number_sessions, bad_electrodes_number)
    for count, i in enumerate(sessions):
        logger.info("Solving IP: " + str(count + 1) + "/" + str(len(sessions)))
        try:
            case = common_tools.load_case(i)
            run_inverse_problem(case, file="neurokit_", over_write=False)
            logger.info("Done")
        
        except:
            logger.info("No best seconds")


if __name__ == "__main__":
    logger.info("Run test")
    # run_one_perfect_case()
    run_selection(400, 15)
