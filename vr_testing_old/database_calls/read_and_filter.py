"""
Author: SpikaTech
Date: 24/10/2023
Description:
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure_ml.inverse_problem.scripts.signals import signals as azure_ml_signals
from loguru import logger
from procesing_data.ecg_processor import ECGProcessor
from vrcardio_vest.app.filters import apply_filters_per_lead, clean_ecg_from_df

colnames = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
]


def read_and_filter_signals(
    user_id: str,
    session_id: str,
    connection_string: str,
    filtering: str = "wavelet",
    case_dir: str = None,
    fs=250,
    seconds=5,
):

    SAVE_FILES_PATH = os.path.join(case_dir, user_id, session_id)
    """Reads the recorded signals from user_id/session_id"""
    logger.info("Reading and filtering signals")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    case_dir = Path(case_dir)

    # Create a directory for user_id/session_id
    # Check first if the directory exists
    if not os.path.exists(os.path.join(SAVE_FILES_PATH, "temporal_signal")):
        os.mkdir(os.path.join(SAVE_FILES_PATH, "temporal_signal"))

    # Load recorded signals from vest
    # logger.info("start reading signals from vest and filtering them")
    vest_signals = azure_ml_signals.get_signals_dataframe(user_id, session_id, case_dir, blob_service)

    """if not os.path.exists(os.path.join(case_dir, session_id)):
        os.mkdir(os.path.join(case_dir, session_id))"""

    vest_signals_all = vest_signals * 1000
    vest_signals.to_csv(os.path.join(SAVE_FILES_PATH, "signals_raw.csv"), index=False)
    # logger.info("Raw signals retrieved")

    # ECG processing. Summary: outlier removal, baseline wander correction, interference removal and bad electrodes
    # detection
    processor = ECGProcessor(colnames, vest_signals, fs)
    processor.filter_ecgs()

    bad_electrodes = processor.bad_electrodes
    pd.DataFrame(bad_electrodes).to_csv(os.path.join(SAVE_FILES_PATH, "bad_electrodes.csv"), index=False, header=False)

    pd.DataFrame([processor.lower, processor.upper]).to_csv(
        os.path.join(SAVE_FILES_PATH, "lower_and_upper_indexes.csv")
    )
    vest_signals_all[processor.lower : processor.lower + fs * seconds].to_csv(
        os.path.join(SAVE_FILES_PATH, "signals_raw_best_seconds.csv")
    )
    # Get the first 5 seconds of signal without interferences.
    best_seconds_signals = processor.signals[0 : fs * seconds]

    best_seconds_signals.to_csv(os.path.join(SAVE_FILES_PATH, "signals_preprocessed_best_seconds.csv"))

    # Filter the signals
    if filtering == "wavelet":
        # logger.info("Wavelet filtering.")
        # Filter the signal with the wavelet.
        vest_signals_dataframe = pd.DataFrame(apply_filters_per_lead(np.asarray(best_seconds_signals).T, True))
    elif filtering == "neurokit":
        # logger.info("Neurokit filtering.")
        # Filter the signal with neurokit
        vest_signals_dataframe = clean_ecg_from_df(best_seconds_signals)

    # logger.info("finish filtering")

    # Get the statistics of the best seconds filtered
    # best_seconds_stats = vest_signals_dataframe.describe()

    # Write the processed signals in the temporary directory
    signals = vest_signals_dataframe * 1000
    signals.to_csv(os.path.join(SAVE_FILES_PATH, "best_seconds_filtered.csv"), index=False, header=False)
    shutil.rmtree(os.path.join(SAVE_FILES_PATH, "temporal_signal"))
