"""
Author: SpikaTech
Code to get means of statistical measurements over sessions
for interpolated data
"""
import os

import numpy as np
import pandas as pd
from loguru import logger


def stats_permuted(
    main_folder, verify=True, verify_database_file="src/VRCARDIO-Testing/results/complete_database_info.csv"
):

    """
    Function that generates the RMSE, correlation coeficient and smape for each session and saves all the
    data in 'k_1_validation_3stats_summary.csv' and 'k_1_validation_3stats_summary.csv' files.

    params:
        - main_folder (string): directory with the directory of the main folder where user-sessions are located.
        - verify (bool): Boolean that indicates if a previous comprobation of the case should be done.
        - verify_database_file (string): file where all the database info is registered.

    returns:
        - averages (pandas dataframe): Average electrodes information calculated.
    """

    averages = pd.DataFrame(columns=("mean", "method", "metric"))
    users_id = os.listdir(main_folder)
    for user_id in users_id:
        user_folder = main_folder + "/" + user_id
        sessions_ids = os.listdir(user_folder)
        for session_id in sessions_ids:

            id_session = session_id
            id_user = user_id
            metrics = ["mse", "corr", "smape"]
            methods = ["value_closest", "value_rbf_thin_plate", "value_laplace"]
            partial_results = {}
            if os.path.isfile(
                main_folder + "/" + id_user + "/" + id_session + "/" + "k_1_validation_3stats_summary.csv"
            ):
                if verify:
                    dbfile = pd.read_csv(verify_database_file)
                    db_user = dbfile[dbfile["user_id"] == id_user]
                    db_user_session = db_user[db_user["session_id"] == id_session]

                    stats_file = (
                        main_folder + "/" + id_user + "/" + id_session + "/" + "k_1_validation_3stats_summary.csv"
                    )
                    session_stats = pd.read_csv(stats_file)
                    for method in methods:
                        stats_method = session_stats[session_stats["Unnamed: 0"] == method]
                        mean_method = stats_method[stats_method["Unnamed: 1"] == "mean"]
                        for metric_name in metrics:
                            partial_results[method, metric_name] = mean_method[metric_name].values[0]
                    means = pd.DataFrame(partial_results.items(), columns=["combined_name", "mean"])
                    means[["method", "metric"]] = pd.DataFrame(
                        means["combined_name"].tolist(), columns=["method", "metric"]
                    )
                    means["user_id"] = id_user
                    means["session_id"] = id_session
                    means.drop(columns="combined_name", inplace=True)
                    averages = pd.concat([averages, means], ignore_index=True)

                else:
                    logger.warning("not verify")

            else:
                logger.warning("not file")

    return averages


def save_averages_file():
    """
    Function that generares a summary excell with the results of the
    experiments runned.

    params: none
    returns: none
    """
    averages = stats_permuted("src/VRCARDIO-Testing/database_calls/session_data")
    averages.to_csv("src/VRCARDIO-Testing/results/test1_1_avergage_performance.csv")

    averages.drop(columns="user_id", inplace=True)
    averages.drop(columns="session_id", inplace=True)
    logger.info(
        np.min(averages[averages["metric"] == "corr"]["mean"]), np.max(averages[averages["metric"] == "corr"]["mean"])
    )
    logger.info(
        np.min(averages[averages["metric"] == "mse"]["mean"]), np.max(averages[averages["metric"] == "mse"]["mean"])
    )

    averages = averages[["method", "metric", "mean"]]
    averages.to_excel("src/VRCARDIO-Testing/results/test1_1_avergage_performance.xlsx", index=False)

    global_resume = averages.groupby(["method", "metric"]).agg(["mean", np.std])
    logger.info(global_resume)
    df = global_resume
    df.to_excel("src/VRCARDIO-Testing/results/test1_1_global_resume.xlsx")
