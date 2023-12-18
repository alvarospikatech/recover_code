"""
Author: SpikaTech
Date: 24/10/2023
Description:
"""

import sys

import numpy as np
import pandas as pd
from loguru import logger

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests")
from common_tools import confusion_matrix_stats


def predicted_stats():

    # Nombre del archivo Excel
    archivo_excel = "src/VRCARDIO-Testing/results/database_evaluation_VRcardio_Victoria.xlsx"  # Reemplaza con el nombre de tu archivo Excel

    df = pd.read_excel(archivo_excel)

    victoria_info = df["FA_en_ECGs_DAS"] == "SI"
    predicted_info = df["arrhytmia_prediction"] == "AF"

    zeros = np.zeros(len(predicted_info))

    save_list = list(zip(zeros, zeros, victoria_info, predicted_info))

    save_list_clean = [list(tupla) for tupla in save_list]
    # print(save_list_clean)

    (
        fn_percent,
        fp_percent,
        tp_percent,
        tn_percent,
        aciertos_percent,
        precision,
        recall,
        f1,
    ) = confusion_matrix_stats(save_list_clean[0], save_list[1])

    stats = {
        "False_positives": ["False positives", fp_percent],
        "False neg": ["False negatives", fn_percent],
        "True positives": ["True positives", tp_percent],
        "True negatives": ["True negatives", tn_percent],
        "Precision": ["Precision", precision],
        "Recall": ["Recall", recall],
        "F1": ["F1", f1],
    }

    df1 = pd.DataFrame(save_list, columns=["None", "None", "Victoria judgement", "arrhytmia_prediction"])
    df2 = pd.DataFrame(stats).T
    df2.columns = ["Result", "%"]

    with pd.ExcelWriter("src/VRCARDIO-Testing/results/4_results_predicted.xlsx", engine="xlsxwriter") as writer:

        df1.to_excel(writer, sheet_name="All data", index=False)
        df2.to_excel(writer, sheet_name="stats", index=False)


if __name__ == "__main__":
    logger.info("Run test")
    predicted_stats()
