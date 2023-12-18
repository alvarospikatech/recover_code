"""
Author: SpikaTech
Date: 24/10/2023
Description:
"""

import numpy as np
import pandas as pd
from angle_method import das_signal
from loguru import logger
from mesh_interpolation.closest_interpolation import closest_interpolation
from mesh_interpolation.laplacian_interpolation import laplace_interpolation
from mesh_interpolation.lko_3stats import corr, mse, smape
from mesh_interpolation.rbf_interpolation import rbf_interpolation


def one_session_stats(
    case_obj,
    methods=[
        ("closest", closest_interpolation),
        ("rbf_thin_plate", rbf_interpolation),
        ("laplace", laplace_interpolation),
    ],
    K=1,
):

    id_user = case_obj["user_id"]
    logger.info("user_id=", id_user)
    id_session = case_obj["session_id"]
    logger.info("session_id=", id_session)
    points = case_obj["torso_points"]
    faces = case_obj["torso_faces"]
    measures = case_obj["measures"]
    unknown_indexes = case_obj["unknown_indexes"]
    bad_electrodes_indexes = case_obj["mesh_bad_electrodes"]

    size = case_obj["vest"]
    local_dir = "src/VRCARDIO-Testing/database_calls/session_data/" + case_obj["user_id"] + "/" + case_obj["session_id"]

    metrics = [("mse", mse), ("corr", corr), ("smape", smape)]  # , ("cross_corr",cross_corr)]

    runs = {}

    cardiac_axis, das_values = das_signal(local_dir + "/best_seconds_filtered.csv")

    das_values = das_values.values.T
    true_data = das_values

    # true_data = das_values[:, ::3]
    # true_data = das_values[:, 0 : int(das_values.shape[1] / 3)]

    electrodes = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    for method_name, method in methods:

        full_estimation = method(points, faces, measures, unknown_indexes)
        estimated_data = ecg_potentials(full_estimation, size)

        for metric_name, metric in metrics:
            for i, electrode in enumerate(electrodes):

                runs[method_name, metric_name, electrode] = metric(true_data[[i], :], estimated_data[[i], :])

    electrode_list = []
    thin_value = []
    laplace_value = []
    closest_value = []
    metric_list = []

    for i, electrode in enumerate(electrodes):
        for metric, metric_name in metrics:
            electrode_list.append(electrode)
            metric_list.append(metric)
            thin_value.append(runs[("rbf_thin_plate", metric, electrode)])
            closest_value.append(runs[("closest", metric, electrode)])
            laplace_value.append(runs[("laplace", metric, electrode)])

    to_data_frame = [electrode_list, thin_value, laplace_value, closest_value, metric_list]

    df = pd.DataFrame(to_data_frame).T
    df.columns = ["electrode", "thin_plate", "laplace", "closest", "metric"]

    export_one_session_data(df, local_dir)


def ecg_potentials(interpolated_signals, size: str):

    """
    Function that returns all the 12 ecg Leads made post-interpolated

    args:
        - interpolated_signals (np.array): interpolated signals in the nodes x times shape
        - size (string): string that represents the mesh assignated to match the vest gender
                         and sizr. Example male_m: gender male, size:m

    return:
        - numpy array with the 12 leads calculated in a lead x times signal structure.

    """
    # electrodes order: [ra , la, v6, v5, v4, v3, v2, v1, rl, ll]

    option = {
        "male_m": [948, 152, 1004, 1031, 1040, 1039, 1052, 1141, 358, 304],
        "male_l": [502, 1927, 628, 649, 733, 1186, 1108, 1090, 1390, 92],
        "male_xxl": [575, 1905, 1789, 2022, 1802, 1883, 1915, 1884, 185, 287],
        "female_m": [2796, 2203, 1569, 874, 1835, 467, 391, 414, 741, 1174],
        "female_l": [1362, 2558, 859, 1695, 512, 508, 521, 736, 332, 1121],
        "female_xl": [22, 2554, 2488, 370, 1939, 915, 902, 1402, 2711, 842],
    }

    electrode_indexes = option[size]

    leads_potentials = interpolated_signals[electrode_indexes, :]

    I = leads_potentials[1, :] - leads_potentials[0, :]
    II = leads_potentials[9, :] - leads_potentials[0, :]
    III = leads_potentials[9, :] + leads_potentials[1, :]
    aVR = (leads_potentials[1, :] + leads_potentials[0, :]) / 2
    aVL = (leads_potentials[9, :] + leads_potentials[1, :]) / 2
    aVF = (leads_potentials[9, :] + leads_potentials[0, :]) / 2
    V1 = leads_potentials[7, :]
    V2 = leads_potentials[6, :]
    V3 = leads_potentials[5, :]
    V4 = leads_potentials[4, :]
    V5 = leads_potentials[3, :]
    V6 = leads_potentials[2, :]

    twelve_leads = np.array([I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6])

    return twelve_leads


def export_one_session_data(join_results, local_dir):

    """
    Exports the results into two xlsx files: ecg_das_test_3stats and ecg_das_test_3stats_summary.

    args:
        - join_results = resulted data from the experiment
        - local_dir = local dir where this files are been saving, they should be in their session_data/user_id/session_id
                      folder.

    returns: none
    """

    pd.set_option("display.float_format", lambda x: "%.5f" % x)
    join_results.to_excel(local_dir + "/ecg_das_test_3stats.xlsx")  # , index=False)
    columnas_numericas = ["thin_plate", "laplace", "closest"]
    join_results_agg = join_results.groupby(["metric"])[columnas_numericas].agg(["mean", np.std])
    join_results_agg.T.to_csv(local_dir + "/ecg_das_test_3stats_summary.csv")
    join_results_agg.T.to_excel(local_dir + "/ecg_das_test_3stats_summary.xlsx")
    logger.info(join_results_agg)
