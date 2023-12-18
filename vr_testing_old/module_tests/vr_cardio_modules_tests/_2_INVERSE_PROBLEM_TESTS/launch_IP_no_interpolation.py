"""
Author: SpikaTech
Date: 24/10/2023
Description:
____________
TEST 2_0.

Not really a test but launches all inverse's problem
"""

import sys

import pandas as pd
from loguru import logger

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests")
import common_tools
from _2_INVERSE_PROBLEM_TESTS.IP_code import run_inverse_clasic_managed


def run_selection(number_sessions, bad_electrodes_number):
    """
    Function that generates a selection of cases to run the experiments over them.

    params:
        - number_sessions (int): number of sessions to run
        - bad_electrodes_number (int): number of MAX wrong electrodes to be included.

    returns: none
    """
    file = "nointerp_wavelet_"
    sessions = common_tools.get_subset(number_sessions, bad_electrodes_number)
    for count, i in enumerate(sessions):
        logger.info("Solving IP: " + str(count + 1) + "/" + str(len(sessions)))
        case = common_tools.load_case(i)
        output = run_inverse_clasic_managed(
            torso_points=case["torso_points"],
            torso_faces=case["torso_faces"],
            heart_points=case["heart_points"],
            heart_faces=case["heart_faces"],
            torso_signals=case["electrodes_best_seconds"],
            measured_potentials_position=case["torso_points"][case["mesh_good_electrodes"]],
        )
        df = pd.DataFrame(output)
        path = "src/VRCARDIO-Testing/database_calls/session_data/" + case["user_id"] + "/" + case["session_id"]
        path = path + "/" + file + "local_heart_potentials.csv"

        df.to_csv(path, index=False, header=False)
        logger.info("Done")


if __name__ == "__main__":
    logger.info("Run test")
    # run_one_perfect_case()
    run_selection(400, 15)
