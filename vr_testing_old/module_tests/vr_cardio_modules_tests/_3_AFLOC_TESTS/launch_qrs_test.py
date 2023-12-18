import sys

import numpy as np
import pandas as pd
from af_location_loop import af_location, save_data_file
from loguru import logger

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests/_2_INVERSE_PROBLEM_TESTS")
from IP_code import run_inverse_clasic_managed

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests")
import common_tools
from QRS_detector import multiple_signals_QRS, multiple_signals_QRST


def run_one_perfect_case():
    """
    Function that runs just one perfect case as an example.

    params:none

    returns: none
    """
    session = common_tools.get_victoria_subset(1, 0, af=False)[0]
    logger.info(session)
    case = common_tools.load_case(session)
    QRS = multiple_signals_QRST(case["torso_potentials"], case["t_axis"], case["fs"])

    print(QRS.shape)
    print(case["torso_potentials"].shape)


    common_tools.plot_mesh(
        case["torso_points"],
        case["torso_faces"],
        case["torso_potentials"],
        case["t_axis"],
        aux_signal=QRS,
        fs=case["fs"],
    )

    print("Calculating the inverse:")
    case = common_tools.load_case(session)

    torso_signals = case["torso_potentials"]
    inverse_window = 1 - 0.985 * QRS  # 0.985*QRS
    torso_signals = case["torso_potentials"]  # * inverse_window

    common_tools.plot_mesh(case["torso_points"], case["torso_faces"], QRS, case["t_axis"], aux_signal=QRS)

    lamb = np.linspace(10**-5, 10**-8, 1000).tolist()
    IP_output = case["neurokit_local_heart_potentials"]
    """IP_output = run_inverse_clasic_managed(
        torso_points=case["torso_points"],
        torso_faces=case["torso_faces"],
        heart_points=case["heart_points"],
        heart_faces=case["heart_faces"],
        torso_signals=torso_signals,
        measured_potentials_position = case["torso_points"],
        lamb=lamb,
    )"""

    QRS = multiple_signals_QRST(IP_output, case["t_axis"], case["fs"])
    common_tools.plot_mesh(case["heart_points"], case["heart_faces"], IP_output, case["t_axis"], aux_signal=QRS)

    
    inverse_window = 1 - 0.985 * QRS  # 0.985*QRS
    IP_output = IP_output * inverse_window
    common_tools.plot_mesh(case["heart_points"], case["heart_faces"], IP_output, case["t_axis"], aux_signal=QRS)

    return QRS


run_one_perfect_case()
