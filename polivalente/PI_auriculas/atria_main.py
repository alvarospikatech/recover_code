import os
from dataclasses import dataclass
import logging

from copy import copy
import pyvista as pv
import numpy as np
import trimesh
from inverse_problem.inverse_problem_solvers.mfs.mfs_inverse import mfs_inverse
from inverse_problem.inverse_problem_solvers.mesh import Mesh
from inverse_problem.inverse_problem_solvers.mfs.regularization import find_lambda_creso
from plot_mesh import plot_mesh
import pandas as pd

from vrcardio_signal_preprocessing.utils import QRST_deletion


from loguru import logger

if __name__ == '__main__':

    """
    folder_files = "/home/angeles/Cavities/validate_interpolation/VRCARDIO-Testing/src/VRCARDIO-Testing/database_calls/session_data/8d21dbf2-bb38-4556-8936-cadbdfb05231/"
    session1= "25e1dfef-d986-4dd1-a352-049b39703020/"
    session2 = "b9de40d0-a4e0-4b07-bd89-0e951366e3d8/"
    """

    folder_files = "PI_auriculas/"
    session1 =""
    heart = pv.read(folder_files + session1 + "atria_decimated.stl")

    pos_heart = pv.read(folder_files + session1 + "pos_heart.stl")
    centroid_pos = np.mean(pos_heart.points,axis=0)

    heart.points = heart.points /1000
    centroid_og = np.mean(heart.points,axis=0)

    heart.points = heart.points - centroid_og + centroid_pos
    pvtorso = pv.read(folder_files + session1 + "torso.stl")

    mmheart = heart



    pl = pv.Plotter()
    pl.add_mesh(pvtorso, opacity = 0.3)
    pl.add_mesh(mmheart)
    pl.show_grid()
    pl.show()

    mmheart = Mesh((mmheart.faces).reshape(mmheart.n_faces, 4)[:,1:], mmheart.points)
    
    torso_potentials = pd.read_csv(folder_files + session1 + "torso_potentials.csv").T
    torso_potentials = torso_potentials.to_numpy()

    fs = 360
    dur = (torso_potentials.shape[1] - 1) / fs
    interval = 1 / fs
    t_axis = np.arange(0, dur + interval / 2, interval)

    torso_mesh = Mesh(pvtorso.faces.reshape(pvtorso.n_faces, 4)[:, 1:], pvtorso.points*1000)
    #heart_mesh = Mesh(heart.faces.reshape(heart.n_faces, 4)[:, 1:], heart.points*1000)
    heart_mesh = Mesh(mmheart.triplets, mmheart.points*1000)


    #plot_mesh(vertices=torso_mesh.points, triangles= torso_mesh.triplets, signal=torso_potentials, x_axis=t_axis)

    #if_potentials = np.load("/home/angeles/Cavities/validate_interpolation/mesh_interpolation/src/mesh_interpolation/if_torso_potentials+1.npz")
    #plot_mesh(torso_mesh.points, torso_mesh.triplets, if_potentials['arr_0'])
    

    print("QRST DELETION")
    print(pvtorso.points.shape)
    print(torso_potentials.shape)
    torso_potentials_clean, reference_signals = QRST_deletion(torso_potentials,fs=360)

    print(torso_potentials)

    plot_mesh(vertices=torso_mesh.points, triangles= torso_mesh.triplets, signal=torso_potentials, x_axis=t_axis, aux_signal=reference_signals)


    torso_potentials = torso_potentials_clean

    scale_factor = 1

    lambda_method = find_lambda_creso
    scale_method = "normals"


    heart_potentials = mfs_inverse(
        heart_mesh,
        torso_mesh,
        torso_potentials,
        torso_mesh.points, #measured_potentials_position,
        scale_factor,
        lambda_method,
        scale_method,
    )
    heart_potentials_aux = copy(heart_potentials)
    print("min and max vals of heart potentials =", heart_potentials_aux.min(), heart_potentials_aux.max())

    plot_mesh(heart_mesh.points, heart_mesh.triplets, heart_potentials,x_axis=t_axis )

