import sys

import numpy as np
import pandas as pd
from af_location_loop import af_location, save_data_file
from loguru import logger
from QRS_detector import multiple_signals_QRS, multiple_signals_QRST

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests")
import common_tools


def run_selection(number_sessions, bad_electrodes_number):
    """
    Function that generates a selection of cases to run the experiments over them.

    params:
        - number_sessions (int): number of sessions to run
        - bad_electrodes_number (int): number of MAX wrong electrodes to be included.

    returns: none
    """
    save_list = []
    sessions = common_tools.get_subset(number_sessions, bad_electrodes_number)
    for count, i in enumerate(sessions):
        case = common_tools.load_case(i)

        target = False
        if case["FA_en_ECGs_DAS"] == "SI":
            target = True
        elif case["FA_en_ECGs_DAS"] == "NO":
            target = False
        else:
            target = "NotCount"

        if target != "NotCount":
            try:
                var = "neurokit_local_"
                QRS = multiple_signals_QRST(case[var + "heart_potentials"], case["t_axis"], case["fs"])
                delete_qrs_wind = 1 - 0.95 * QRS
                case[var + "heart_potentials"] = case[var + "heart_potentials"] * delete_qrs_wind
                result, _ = af_location(case, var, 6.5)
            except RecursionError:
                logger.error("RecursionError: maximum recursion depth exceeded in comparison")
            logger.info("Tested: " + str(count + 1) + " / " + str(len(sessions)))

            print("Output: " + str(result))
            print("Victoria: " + str(target))

            save_list.append([case["user_id"], case["session_id"], result, target, i["arrhytmia_prediction"]])

    # Crear un DataFrame de pandas

    save_data_file(save_list)


def find_optimal_freq(number_sessions, bad_electrodes_number):

    posible_range = np.arange(4, 9, 0.2)
    save_excell = {}
    for thres in posible_range:
        save_list = []
        sessions = common_tools.get_subset(number_sessions, bad_electrodes_number)
        for count, i in enumerate(sessions):
            case = common_tools.load_case(i)
            target = False
            if case["FA_en_ECGs_DAS"] == "SI":
                target = True
            elif case["FA_en_ECGs_DAS"] == "NO":
                target = False
            else:
                target = "NotCount"

            if target != "NotCount":

                try:
                    var = "wavelet_local_"
                    QRS = multiple_signals_QRST(case[var + "heart_potentials"], case["t_axis"], case["fs"])
                    delete_qrs_wind = 1 - 0.985 * QRS
                    case[var + "heart_potentials"] = case[var + "heart_potentials"] * delete_qrs_wind
                    result, _ = af_location(case, file=var, threshol_freq=thres)
                except RecursionError:
                    logger.error("RecursionError: maximum recursion depth exceeded in comparison")
                logger.info("Tested: " + str(count + 1) + " / " + str(len(sessions)))

                save_list.append([case["user_id"], case["session_id"], result, target])
                logger.info("threshold: " + str(thres))

        save_list = np.array(save_list)
        save_excell[thres] = common_tools.confusion_matrix_stats(
            predicciones=save_list[:, 2], etiquetas_verdaderas=save_list[:, 3]
        )

    df = pd.DataFrame(save_excell).T
    df.columns = [
        "False neg",
        "False pos",
        "True positive",
        "True negative",
        "Accuracy",
        "Precision",
        "recall",
        "f1",
    ]

    df.to_excel("src/VRCARDIO-Testing/results/3_optimal_af_freq_thres.xlsx")


run_selection(400, 0)
#find_optimal_freq(400, 0)
