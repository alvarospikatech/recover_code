"""
Author: SpikaTech
Date: 24/10/2023
Description:
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger
from procesing_data.heart_potentials import calculate_heart_potentials

# from heart_potentials import calculate_heart_potentials
from procesing_data.read_and_filter import read_and_filter_signals
from procesing_data.torso_potentials import calculate_torso_potentials
from azure.storage.blob import BlobServiceClient
from azure_ml.inverse_problem.scripts.signals import signals as azure_ml_signals


def download_raw_signals(user_id: str, session_id: str, case_dir: str):
    load_dotenv(find_dotenv())
    connection_string = os.environ.get("BLOB_CONNECTION_STRING")

    logger.info("Downloading signals")
    blob_service = BlobServiceClient.from_connection_string(connection_string)

    try:
        vest_signals = azure_ml_signals.get_signals_dataframe(user_id, session_id, case_dir, blob_service)
        print(vest_signals)
        vest_signals.to_csv(case_dir + "/" + user_id + "/" + session_id + "/signals_raw.csv", index=False)

    except:
        logger.error("Unable to download the signals")



def download_all_cases_signals(main_folder="src/VRCARDIO-Testing/database_calls/session_data"):
    logger.info("Processing cases...")
    user_counter = 1
    users_id = os.listdir(main_folder)
    for user_id in users_id:
        user_folder = main_folder + "/" + user_id
        sessions_ids = os.listdir(user_folder)
        print(sessions_ids)
        for session_id in sessions_ids:
            case_dir = main_folder
            download_raw_signals(user_id=user_id, session_id=session_id, case_dir=str(case_dir))

        user_counter += 1
        logger.info("Processed " + str(user_counter) + " / " + str(len(users_id)))