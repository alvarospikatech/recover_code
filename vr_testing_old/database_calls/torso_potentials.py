"""
Author: SpikaTech
Date: 24/10/2023
Description:
"""

from pathlib import Path

import pandas as pd
import pyvista as pv
from azure_ml.inverse_problem.scripts.torso import torso_potentials as azure_ml_torso
from loguru import logger
from pandas.errors import EmptyDataError


def calculate_torso_potentials(download_uri: str, session_id: str):
    """
    Returns a dataframe with the interpolated signals of the torso.
    Args:
        download_uri (str): The uri to the download folder.
        session_id (str): The session id.
        download_uri (str): The uri to the signals info.
        torso_uri (str): The uri to the torso info.
    Returns:
        pd.DataFrame: A dataframe with the interpolated signals of the torso
        list: Electrode indexes: The indexes of the electrodes
        list : Bad electrodes_indexes: The indexes of the bad electrodes

    """
    download_uri = Path(download_uri)
    try:
        signals = pd.read_csv(str(download_uri / "best_seconds_filtered.csv"), index_col=False, header=None)
    except FileNotFoundError:
        logger.error(
            "The signals file does not exist. The session was not recorded or uploaded correctly " "and it is corrupt."
        )
        raise FileNotFoundError
    try:
        bad_electrodes = (
            pd.read_csv(str(download_uri / "bad_electrodes.csv"), header=None, index_col=False)[0].astype(int).tolist()
        )
    except EmptyDataError:
        bad_electrodes = []
    except FileNotFoundError:
        logger.error(
            "The bad electrodes file does not exist. The session was not recorded or uploaded correctly "
            "and it is corrupt."
        )
        raise FileNotFoundError

    # Read the mesh
    reader = pv.get_reader(str(download_uri / "torso.stl"))
    mesh = reader.read()
    electrode_points_path = str(download_uri / "electrodes.tsv")
    electrode_points = azure_ml_torso._read_electrode_points(electrode_points_path)

    signals, electrode_indexes = azure_ml_torso.get_signals(signals, mesh, electrode_points)

    df_electrode_indexes = pd.DataFrame(electrode_indexes)
    df_electrode_indexes.to_csv(str(download_uri / "electrode_indexes.csv"), index=False, header=False)

    # Interpolate signals
    try:
        interpolated_signals = azure_ml_torso._interpolate_signals(mesh, signals, bad_electrodes, electrode_indexes)
    except ValueError:
        if len(bad_electrodes) > 15:
            logger.error(
                f"There are too many bad electrodes. The session has {len(bad_electrodes)} flat signals. "
                f"The inverse problem cannot be solved with so many bad electrodes."
            )
            raise ValueError

    # Save interpolated signals to csv
    interpolated_signals_dataframe = pd.DataFrame(interpolated_signals)
    file_path = str(download_uri / "torso_potentials.csv")
    interpolated_signals_dataframe.to_csv(file_path, index=False, header=False)
