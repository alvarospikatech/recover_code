"""
Author: SpikaTech
Date: 25/09/2023
Description: Calculate statistics of interpolation by permutation for n individuals
"""
import os

import numpy as np
import pandas as pd
from loguru import logger
from mesh_interpolation.closest_interpolation import closest_interpolation
from mesh_interpolation.laplacian_interpolation import laplace_interpolation
from mesh_interpolation.lko_3stats import lok_validation
from mesh_interpolation.rbf_interpolation import rbf_interpolation


def one_session_stats(
    case_obj,
    over_write=False,
    methods=[
        ("closest", closest_interpolation),
        ("rbf_thin_plate", rbf_interpolation),
        ("laplace", laplace_interpolation),
    ],
    K=1,
):

    results = []
    RUN_ARGS = (
        case_obj["torso_points"],
        case_obj["torso_faces"],
        case_obj["measures"],
        case_obj["unknown_indexes"],
        case_obj["raw_bad_electrodes"],
    )
    K = 1
    local_dir = "src/VRCARDIO-Testing/database_calls/session_data/" + case_obj["user_id"] + "/" + case_obj["session_id"]

    file_names = os.listdir(local_dir)
    condition = "k_1_validation_3stats_summary.csv" in file_names
    if not condition or over_write:

        for method_name, method in methods:
            partial_result = lok_validation(method, *RUN_ARGS, k=K)
            partial_result = partial_result.add_suffix("_" + method_name)
            results.append(partial_result)

            join_results = results[0]
            if len(results) > 1:
                join_results = join_results.join(results[1:])

        join_results.reset_index(inplace=True)
        join_results[["index", "metric"]] = pd.DataFrame(join_results["indexes"].tolist(), columns=["index", "metric"])
        join_results.drop(columns="indexes", inplace=True)
        logger.info(join_results)
        export_one_session_data(join_results, results_dir=local_dir)
        return join_results

    else:
        logger.info("Case already procesed")


def export_one_session_data(join_results, results_dir):
    pd.set_option("display.float_format", lambda x: "%.5f" % x)
    join_results.set_index("index", inplace=True)
    join_results.to_csv(results_dir + "/k_1_validation_3stats.csv")
    join_results_agg = join_results.groupby(["metric"]).agg(["mean", np.std])
    join_results_agg.T.to_csv(results_dir + "/k_1_validation_3stats_summary.csv")
