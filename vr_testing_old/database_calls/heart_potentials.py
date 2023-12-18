"""
Author: SpikaTech
Date: 24/10/2023
Description:
"""

from pathlib import Path

import pandas as pd
import trimesh
from loguru import logger
from pandas.errors import EmptyDataError


def calculate_heart_potentials(session_id: str, download_uri: str):
    """Compute heart potentials from torso potentials and heart mesh."""
    download_uri = Path(download_uri)
    try:
        torso_potentials = pd.read_csv(str(download_uri / "torso_potentials.csv"), header=None, index_col=False)
    except FileNotFoundError:
        # logger.error(
        #    "The torso potentials file does not exist. The session was not recorded or uploaded correctly "
        #    "and it is corrupt."
        # )
        raise FileNotFoundError
    electrode_indexes = (
        pd.read_csv(str(download_uri / "electrode_indexes.csv"), header=None, index_col=False)[0].astype(int).tolist()
    )
    try:
        bad_electrodes = (
            pd.read_csv(str(download_uri / "bad_electrodes.csv"), header=None, index_col=False)[0].astype(int).tolist()
        )
    except EmptyDataError:
        bad_electrodes = []

    # Save the torso potentials in the database

    torso_mesh_path = str(download_uri / "torso.stl")
    heart_mesh_path = str(download_uri / "heart.stl")
    torso = trimesh.load_mesh(torso_mesh_path)
    heart = trimesh.load_mesh(heart_mesh_path)
    vertices = torso.vertices
    torso = Mesh(torso.faces, torso.vertices)
    heart = Mesh(heart.faces, heart.vertices)

    electrode_indexes_without_bad_electrodes = list(electrode_indexes)
    # Sort bad electrodes in descending order
    bad_electrodes.sort(reverse=True)
    # Remove bad electrodes
    for bad_electrode in bad_electrodes:
        del electrode_indexes_without_bad_electrodes[bad_electrode]

    # Pick the indexes of the electrodes to compute the heart potentials.
    torso_potentials = torso_potentials[electrode_indexes_without_bad_electrodes].T
    measured_potentials_position = vertices[electrode_indexes_without_bad_electrodes]
    scale_factor = 1
    # lambda_method = partial(find_lambda_creso, lamb=np.linspace(1e-6, 10, 100).tolist())
    lambda_method = find_lambda_creso
    scale_method = "normals"
    heart_potentials = mfs_inverse(
        heart,
        torso,
        torso_potentials,
        measured_potentials_position,
        scale_factor,
        lambda_method,
        scale_method,
    )
    # save_potentials(heart_potentials.T, session_id, heart_uri, "heart_potentials.csv", "uploadHeartPotential")
    # Calculate the voltage map and the local activation time map. Save them for visualization.
    signals = pd.DataFrame(heart_potentials.T)
    file_path = str(download_uri / "heart_potentials.csv")
    signals.to_csv(file_path, index=False, header=False)
