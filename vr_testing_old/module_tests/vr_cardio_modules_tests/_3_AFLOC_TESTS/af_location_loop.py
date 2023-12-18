"""
Author: SpikaTech
Date: 24/10/2023
Description:
"""
import sys

import numpy as np
import pandas as pd
from atrial_fibrillation_location.run import af_spatial_location

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests")
from common_tools import confusion_matrix_stats


def af_location(session, file="neurokit_local_", threshol_freq=7):

    """
    Function that runs the AF location algorithm for just one case, witch data
    is loaded and stored in one session object.

    param:

        - session (session object): Session object returned in the load_case function located in common_tools.py

    returns:

        - boolean: boolean that indicates that if that case detected an AF.

    """

    heart_points = session["heart_points"]
    heart_faces = session["heart_faces"]
    heart_potentials = session[file + "heart_potentials"]
    sampling_freq = 360

    _, color_map, fo_vicente = af_spatial_location(
        heart_points, heart_faces, heart_potentials, sampling_freq, threshol_freq
    )

    return np.sum(color_map) >= 1, fo_vicente


def save_data_file(save_list):

    save_list = np.array(save_list)
    etiquetas_verdaderas = np.array(save_list)[:, 3]
    predicciones = np.array(save_list[:, 2])

    fn, fp, tp, tn, accuracy, precision, recall, f1 = confusion_matrix_stats(
        etiquetas_verdaderas=etiquetas_verdaderas, predicciones=predicciones
    )

    stats = {
        "False_positives": ["False positives", fp],
        "False neg": ["False negatives", fn],
        "True positives": ["True positives", tp],
        "True negatives": ["True negatives", tn],
        "Accuracy": ["Accuracy", accuracy],
        "Precision": ["Precision", precision],
        "Recall": ["Recall", recall],
        "F1": ["F1", f1],
    }

    df1 = pd.DataFrame(
        save_list,
        columns=["user_id", "session_id", "Detected FA", "target_pathologies", "arrhytmia_prediction"],
    )
    df2 = pd.DataFrame(stats).T
    df2.columns = ["Result", "%"]

    with pd.ExcelWriter("src/VRCARDIO-Testing/results/3_results_fa_location.xlsx", engine="xlsxwriter") as writer:

        df1.to_excel(writer, sheet_name="All data", index=False)
        df2.to_excel(writer, sheet_name="stats", index=False)
