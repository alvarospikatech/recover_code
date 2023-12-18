"""
Author: SpikaTech
Date: 24/10/2023
Description:
"""

import os

import numpy as np
import pandas as pd
from inverse_problem.inverse_problem_solvers.mesh import Mesh
from inverse_problem.inverse_problem_solvers.mfs.mfs_inverse import mfs_inverse
from inverse_problem.inverse_problem_solvers.mfs.regularization import find_lambda_creso


def run_inverse_problem(session, file="neurokit_", over_write=False):

    """
    Function that runs the inverse problem for the selection selected.

    args:
        - session: session object that contains all the session data necesary for the IP resolution
        - overwrite: boolean that if True, will overwrite heart_potentials .csv , if not, will simply not
                     re-do the inverse problem.

    return:
        - none
    """
    local_dir = "src/VRCARDIO-Testing/database_calls/session_data/" + session["user_id"] + "/" + session["session_id"]
    file_names = os.listdir(local_dir)
    condition = (file + "local_heart_potentials.csv") in file_names
    if not condition or over_write:
        heart_mesh = Mesh(session["heart_faces"], session["heart_points"])
        torso_mesh = Mesh(session["torso_faces"], session["torso_points"])

        scale_factor = 1
        lambda_method = find_lambda_creso  # find_lambda_creso (bien), find_lambda_zc (bien)  , find_lambda_lc (mal), find_lambda_gcv (mal), find_lambda_rgcv(bien)
        scale_method = "normals"

        output = mfs_inverse(
            heart_mesh,
            torso_mesh,
            session["torso_potentials"],
            session["torso_points"],
            scale_factor,
            lambda_method,
            scale_method,
        )

        df = pd.DataFrame(output)
        path = "src/VRCARDIO-Testing/database_calls/session_data/" + session["user_id"] + "/" + session["session_id"]
        path = path + "/" + file + "local_heart_potentials.csv"

        df.to_csv(path, index=False, header=False)


def run_inverse_clasic_managed(
    torso_points,
    torso_faces,
    heart_points,
    heart_faces,
    torso_signals,
    measured_potentials_position,
    lamb=np.linspace(10**-5, 10**-3, 1000).tolist(),
):

    heart_mesh = Mesh(heart_faces, heart_points)
    torso_mesh = Mesh(torso_faces, torso_points)
    scale_factor = 1
    lambda_method = find_lambda_creso  # find_lambda_creso (bien), find_lambda_zc (bien)  , find_lambda_lc (mal), find_lambda_gcv (mal), find_lambda_rgcv(bien)
    scale_method = "normals"

    print(heart_points.shape)
    print(torso_points.shape)
    print(measured_potentials_position.shape)

    output = mfs_inverse(
        heart_mesh, torso_mesh, torso_signals, measured_potentials_position, scale_factor, lambda_method, scale_method,lamb=lamb
    )
    return output
