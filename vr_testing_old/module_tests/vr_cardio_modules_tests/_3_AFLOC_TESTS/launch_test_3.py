"""
Author: SpikaTech
Date: 24/10/2023
Description:
____________
TEST 3.
This test launchs the test realted to the the AF location. This code compares the results
of the AF Location module (True/False) with the medical evaluation of the signals on the
ECGs.
"""

import sys

import numpy as np
import pandas as pd
from af_location_loop import af_location, save_data_file
from loguru import logger

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests")
import common_tools


def run_one_perfect_case():
    """
    Function that runs just one perfect case as an example.

    params:none

    returns: none
    """
    session = common_tools.get_subset(1, 5)[0]
    logger.info(session)
    case = common_tools.load_case(session)
    logger.info("Detected af?: " + str(af_location(case)))


def run_selection(number_sessions, bad_electrodes_number, type_of_data="neurokit_local"):
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
                result, _ = af_location(case, type_of_data, 6.8)
            except RecursionError:
                logger.error("RecursionError: maximum recursion depth exceeded in comparison")
            logger.info("Tested: " + str(count + 1) + " / " + str(len(sessions)))

            print("Output: " + str(result))
            print("Victoria: " + str(target))

            save_list.append([case["user_id"], case["session_id"], result, target, i["arrhytmia_prediction"]])

    # Crear un DataFrame de pandas

    save_data_file(save_list)


def update_external_excell(external_file):

    """
    Function that updates with the results an especific excell

    Param:
        - external_file (string): Directory/File.xls to update

    Return: None
    """

    df_af = pd.read_excel("src/VRCARDIO-Testing/results/3_results_fa_location.xlsx")
    sheet_name = ""
    try:
        exteranl_df = pd.read_excel(external_file, sheet_name="Sheet1")
        sheet_name = "Sheet1"
    except Exception:

        try:
            exteranl_df = pd.read_excel(external_file, sheet_name="All data")
            sheet_name = "All data"

        except Exception:
            print("Not right sheet founded")

    for _, i in df_af.iterrows():
        fila = exteranl_df[exteranl_df["session_id"] == i["session_id"]]
        if not fila.empty:

            exteranl_df.loc[fila.index, "Automated AF LOCATED"] = i["Detected FA"]
            exteranl_df.loc[fila.index, "arrhytmia_prediction"] = i["arrhytmia_prediction"]

    exteranl_df.to_excel(external_file, sheet_name=sheet_name, index=False)


def find_optimal_freq(number_sessions, bad_electrodes_number):

    posible_range = np.arange(5.4, 7.4, 0.2)
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
                    result, _ = af_location(case, file="neurokit_local_", threshol_freq=thres)
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


if __name__ == "__main__":
    logger.info("Run test")
    # run_one_perfect_case()
    #run_selection(400, 15, "neurokit_local_")

    # update_external_excell("src/VRCARDIO-Testing/results/database_evaluation_VRcardio_Victoria.xlsx")
    find_optimal_freq(400, 15)
