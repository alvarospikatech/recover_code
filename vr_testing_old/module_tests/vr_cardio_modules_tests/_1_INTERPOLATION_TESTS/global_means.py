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
    case_dir, verify=True, verify_database_file="src/VRCARDIO-Testing/results/complete_database_info.csv"
):

    main_folder = "src/VRCARDIO-Testing/database_calls/session_data/"
    users_id = os.listdir(main_folder)
    averages = pd.DataFrame(columns=("mean", "method", "metric"))
    for user_id in users_id:
        user_folder = main_folder + "/" + user_id
        sessions_ids = os.listdir(user_folder)
        for session_id in sessions_ids:

            id_session = session_id
            id_user = user_id
            metrics = ["corr", "mse", "smape"]
            methods = ["thin_plate", "laplace", "closest"]
            partial_results = {}
            if os.path.isfile(case_dir + id_user + "/" + id_session + "/ecg_das_test_3stats_summary.csv"):
                if verify:
                    dbfile = pd.read_csv(verify_database_file)
                    db_user = dbfile[dbfile["user_id"] == id_user]
                    db_user_session = db_user[db_user["session_id"] == id_session]
                    n_bad = np.array(int(db_user_session["Num bad electrodes"]))

                    stats_file = case_dir + id_user + "/" + id_session + "/ecg_das_test_3stats_summary.csv"
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
                    averages = pd.concat([averages, means], ignore_index=True, axis=0)

                else:
                    logger.info("not verify")

            else:
                logger.info(f"User: {id_user} and session: {id_session} hasn't file")

    return averages


def save_averages_file():
    averages = stats_permuted("src/VRCARDIO-Testing/database_calls/session_data/")
    averages.to_csv("src/VRCARDIO-Testing/TESTS/TEST1_0/averages_per_case_session.csv")
    averages.to_excel("src/VRCARDIO-Testing/TESTS/TEST1_0/averages_per_case_session.xlsx")
    averages.drop(columns="user_id", inplace=True)
    averages.drop(columns="session_id", inplace=True)
    join_means = averages.groupby(["method", "metric"]).agg(["mean", np.std])
    logger.info(join_means)
    join_means.to_csv("src/VRCARDIO-Testing/TESTS/TEST1_0/1_global_averages.csv")
    join_means.to_excel("src/VRCARDIO-Testing/TESTS/TEST1_0/1_global_averages.xlsx")


if __name__ == "__main__":

    save_averages_file()
