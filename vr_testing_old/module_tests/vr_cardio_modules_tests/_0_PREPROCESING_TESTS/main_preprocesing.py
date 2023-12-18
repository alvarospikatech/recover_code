

from loguru import logger
import os

# from heart_potentials import calculate_heart_potentials
from vrcardio_signal_preprocessing.run import process_potentials_and_export , filter_torso_signals, interpolate_torso_signals
from loguru import logger
import sys

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests")
import common_tools


#process_potentials_and_export(case_dir="src/VRCARDIO-Testing/database_calls/session_data/0a44f25a-4361-49ec-8440-9f107db4a98b/8684f796-a1f7-44cc-917e-9e3b10410b19/electrodes.tsv")


def calculate_one_case():
    session = common_tools.get_subset(400, 0)[0]
    logger.info(session)
    #session = {"user_id": "0a44f25a-4361-49ec-8440-9f107db4a98b"  , "session_id": "8684f796-a1f7-44cc-917e-9e3b10410b19"}
    case = common_tools.load_case_minimun(session)

    case_dir = "src/VRCARDIO-Testing/database_calls/session_data/" + case["user_id"] + "/" + case["session_id"]

    results = filter_torso_signals( filtering_type="neurokit",save_files= True,case_dir=case_dir ,fs= 250, das_signals = case["signals_raw"])

    results = interpolate_torso_signals(save_files= True, 
                                        case_dir=case_dir,
                                        clean_electrodes_signals=results["best_seconds_filtered"],
                                        mesh_points=case["torso_points"],
                                        mesh_faces=case["torso_faces"],
                                        bad_electrodes= results["bad_electrodes"],
                                        electrode_points=case["electrodes"])
    print(results)
    print("Final")



def calculate_for_all_cases():
    main_folder="src/VRCARDIO-Testing/database_calls/session_data"
    logger.info("Processing cases...")
    user_counter = 1
    users_id = os.listdir(main_folder)
    for user_id in users_id:
        user_folder = main_folder + "/" + user_id
        sessions_ids = os.listdir(user_folder)
        print(sessions_ids)
        for session_id in sessions_ids:
            case_dir = user_folder + "/" + session_id
            results = filter_torso_signals( filtering_type="neurokit",save_files= True,case_dir=case_dir ,fs= 250)
            results = interpolate_torso_signals(save_files= True, 
                                                case_dir=case_dir)

        user_counter += 1
        logger.info("Processed " + str(user_counter) + " / " + str(len(users_id)))


#calculate_one_case()
#calculate_for_all_cases()