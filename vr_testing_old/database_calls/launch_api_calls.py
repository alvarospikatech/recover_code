"""
Author: SpikaTech
Date: 24/10/2023
Description:
"""

from __future__ import print_function

import copy
import os
import time

import numpy as np
import openpyxl
import pandas as pd
import requests
import swagger_client
from loguru import logger
from openpyxl import load_workbook
from procesing_data.read_and_filter import colnames
from swagger_client.rest import ApiException


def configurate():
    """
    This function just configures all the HTTP basic authorization.

    param: none

    returns: none
    """

    configuration = swagger_client.Configuration()
    api_instance = swagger_client.TokenApi(swagger_client.ApiClient(configuration))
    api_response = api_instance.token_create({"username": "admin", "password": "SpikaTechSL2022"})

    api_instance2 = swagger_client.TokenApi(swagger_client.ApiClient())
    api_response2 = api_instance2.token_refresh_create({"refresh": api_response.refresh})

    configuration2 = swagger_client.Configuration()
    configuration2.api_key = {"Authorization": api_response2.access, "Refresh": api_response.refresh}
    configuration2.api_key_prefix = {"Authorization": "Bearer "}

    api_instance = swagger_client.VrCardioApi(swagger_client.ApiClient(configuration2))
    return api_instance


def session_list():
    """
    This function downloads al the information related with the sesion object of the api
    (session id, description ,created at, updated at, last record, age, height, weight,
    sex, physical status, prevoius patlogogies, previous treatment, target pathologies
    position, predictions, corrupted, score, user, device)

    params: None

    returns:
        - data (list): A list with all the sessions information downloaded from the api.
    """

    logger.info("Conecting with the API and downloading session info")
    api_instance = configurate()
    page = 1
    next_page = 1
    data = []
    while page == next_page:
        try:
            api_response = api_instance.vr_cardio_sessions_list(page=next_page)
            next_page = int(api_response.to_dict().get("next"))
            data.extend(api_response.to_dict().get("results"))
            page += 1

        except Exception:
            page = next_page
            break

    return data


def vest_list():
    """
    This function downloads all the information related with the vest object of the api

    params: None

    returns:
        - vest_sizes (list): A list with all the relevant vest information downloaded from the api.
    """

    logger.info("Conecting with the API and downloading vest info")
    api_instance = configurate()
    page = 1
    next_page = 1
    data = []
    while page == next_page:

        try:
            api_response = api_instance.vr_cardio_devices_list(page=next_page)
            next_page = int(api_response.to_dict().get("next"))
            data.extend(api_response.to_dict().get("results"))
            page += 1

        except Exception:
            page = next_page
            break

    vest_sizes = {}
    for i in data:
        vest_sizes[i["id"]] = [i["vest"]["size"], i["vest"]["genre"]]

    return vest_sizes


def download_sesions():
    """
    Function that uses different api calls to download all the user/session information relevant
    for further processing (session procesing). It download all the sessions at "session_data"
    folder where, each subfolder are the user id and each subsubfolder are the sessions related to
    the user.

    params: None

    returns: None
    """

    logger.info("Conecting with the API and downloading session data, this may take a while")
    logger.info("Session data can be found in session_data folder")

    api_instance = configurate()
    folder = "src/VRCARDIO-Testing/database_calls/session_data/"
    data = session_list()
    if not os.path.exists("src/VRCARDIO-Testing/database_calls/session_data"):
        os.mkdir("src/VRCARDIO-Testing/database_calls/session_data")

    for count, i in enumerate(data):

        user_folder = folder + i["user"]
        session_folder = user_folder + "/" + i["session_id"]
        if not os.path.exists(user_folder):
            os.mkdir(user_folder)

        if not os.path.exists(session_folder):
            os.mkdir(session_folder)

        api_response = api_instance.vr_cardio_sessions_read(i["session_id"]).to_dict()

        # print(api_response["heart_potential"]["heart_potential"])
        try:
            heart_potentials = requests.get(api_response["heart_potential"]["heart_potential"])
            open(session_folder + "/api_heart_potentials.csv", "wb").write(heart_potentials.content)
        except Exception:
            pass

        heart = requests.get(api_response["heart_potential"]["heart_mesh"])
        open(session_folder + "/heart.stl", "wb").write(heart.content)

        torso = requests.get(api_response["torso_potential"]["torso_mesh"])
        open(session_folder + "/torso.stl", "wb").write(torso.content)

        electrodes = requests.get(api_response["torso_potential"]["electrodes"])
        open(session_folder + "/electrodes.tsv", "wb").write(electrodes.content)

        logger.info("Downloaded:  " + str(count + 1) + "/" + str(len(data)))


def create_basic_excell():

    """
    This function generates an excell on "results" folder with all the user/session relevant
    data downloaded from the api with stadistics related with the number of cases avaible.
    This data is a little bit uncomplete as its only the provided by the server and some other
    important factors such as the number of bad electrodes arent provided. For that, another
    excell is being developed wich will need to have processed all the cases.

    params: None

    returns: None
    """

    def miscelanous(representate_array, names_array):
        general = []
        general_counts = []
        general_percent = []
        percent = []
        for count, i in enumerate(representate_array):
            info, counts = np.unique(i, return_counts=True)
            percent = np.char.add((np.round(100 * counts / len(i), 2)).astype(str), "%")
            general = general + [""] + [names_array[count]] + info.tolist()
            general_counts = general_counts + ["", ""] + counts.tolist()
            general_percent = general_percent + ["", ""] + percent.tolist()

        return [general, general_counts, general_percent]

    def weight_dataframe(weight_count, expanded=False):
        weights, counts = np.unique(weight_count, return_counts=True)
        if expanded:
            expanded_weights = np.arange(0, 200)
            expanded_counts = np.zeros(200)
            for c, i in enumerate(weights):
                if i == "ND":
                    i = 200
                expanded_counts[int(i)] = counts[c]

            expanded_weights = expanded_weights.astype(str)
            expanded_weights[200] = "ND"
            expanded_counts = expanded_counts[50:]
            expanded_weights = expanded_weights[50:]
            weight_hist = zip(expanded_weights, expanded_counts, 100 * expanded_counts / np.sum(expanded_counts))
        else:
            weight_hist = zip(weights, counts, 100 * counts / np.sum(counts))

        return [weight_hist, ["weight", "Number of times", "%"]]

    def height_dataframe(height_count, expanded=False):
        heights, counts = np.unique(height_count, return_counts=True)
        if expanded:
            expanded_heights = np.arange(0, 252)
            expanded_counts = np.zeros(252)
            for c, i in enumerate(heights):
                if i == "ND":
                    i = 251
                expanded_counts[int(i)] = counts[c]

            expanded_heights = expanded_heights.astype(str)
            expanded_heights[251] = "ND"
            expanded_counts = expanded_counts[50:]
            expanded_heights = expanded_heights[50:]
            height_hist = zip(expanded_heights, expanded_counts, 100 * expanded_counts / np.sum(expanded_counts))

        else:
            height_hist = zip(heights, counts, 100 * counts / np.sum(counts))

        return [height_hist, ["height", "Number of times", "%"]]

    def age_dataframe(age_count, expanded=False):
        ages, counts = np.unique(age_count, return_counts=True)
        if expanded:
            expanded_ages = np.arange(0, 102)
            expanded_counts = np.zeros(102)

            for c, i in enumerate(ages):
                if i == "ND":
                    i = 101
                expanded_counts[int(i)] = counts[c]

            expanded_ages = expanded_ages.astype(str)
            expanded_ages[101] = "ND"
            age_hist = zip(expanded_ages, expanded_counts, 100 * expanded_counts / np.sum(expanded_counts))

        else:
            age_hist = zip(ages, counts, 100 * counts / np.sum(counts))

        return [age_hist, ["Age", "Number of times", "%"]]

    sinnoh_bool = False
    archivos_en_directorio = os.listdir("src/VRCARDIO-Testing/results")
    if "sinnoh_1.xlsx" in archivos_en_directorio:
        sinnoh_df = pd.read_excel("src/VRCARDIO-Testing/results/sinnoh_1.xlsx", index_col=None)
        sinnoh_bool = True

    logger.info("Conecting with the API and downloading all the info...")

    data = session_list()
    clean_data = []
    vest_listed = vest_list()
    user_data = []
    error_list = [None, "", " "]

    api_instance = configurate()

    for i in data:

        user_id = i["user"] if i["user"] != 0 else "ND"
        age = i["age"] if i["age"] != 0 else "ND"
        height = i["height"] if i["height"] != 0 else "ND"
        weight = i["weight"] if i["weight"] != 0 else "ND"
        sex = i["sex"] if i["sex"] != 0 else "ND"

        session = {}
        if sinnoh_bool:

            try:
                row = sinnoh_df[sinnoh_df["ID"] == user_id]
                session["user_name"] = row["DESCRIPTION"].values[0]
                # print(row["DESCRIPTION"].values)

            except:
                session["user_name"] = "ERROR: Not name found for this patient"

        else:
            session["user_name"] = "Unable to get Patient real name"

        session["user_id"] = user_id
        session["age"] = age
        session["height"] = height
        session["weight"] = weight
        session["sex"] = sex

        session["session_id"] = i["session_id"]
        session["physical_status"] = i["physical_status"].lower()
        if i["target_pathologies"] in ["None", "no", "none", "No"]:
            i["target_pathologies"] = "SR (Sinus Rhythm)"
        session["target_pathologies"] = i["target_pathologies"] if not i["target_pathologies"] in error_list else "ND"
        session["position"] = i["position"].lower() if not i["position"] in error_list else "ND"
        session["predictions"] = i["predictions"].lower() if not i["predictions"] in error_list else "ND"
        session["corrupted"] = i["corrupted"]
        session["score"] = i["score"]

        session["vest size"] = vest_listed[i["device"]][0]
        session["vest gender"] = vest_listed[i["device"]][1]

        new_petition = api_instance.vr_cardio_sessions_read(session_id=session["session_id"]).to_dict()
        prediction = new_petition["arrhytmia_prediction"]
        print(prediction)

        session["arrhytmia_prediction"] = prediction

        session["vest size"] = prediction

        user_data.append(session)

    actual_uid = ""
    clean_data = []

    age_count = []
    height_count = []
    weight_count = []

    gender_count = []
    vest_gender_count = []
    vest_size_count = []
    pat_count = []
    phisical_count = []
    position_count = []
    corrupted_count = []
    prediction_count = []

    full_df = copy.deepcopy(user_data)

    for i in user_data:
        if actual_uid == i["user_id"]:
            i["user_name"] = ""
            i["user_id"] = ""
            i["age"] = ""
            i["height"] = ""
            i["weight"] = ""
            i["sex"] = ""
        else:
            actual_uid = i["user_id"]
            age_count.append(i["age"])
            height_count.append(i["height"])
            if len(str(i["weight"])) == 2 and i["weight"] != "ND":
                weight_count.append("0" + str(i["weight"]))
            else:
                weight_count.append(str(i["weight"]))

            gender_count.append(i["sex"])
            vest_gender_count.append(i["vest gender"])
            vest_size_count.append(i["vest size"])
            pat_count.append(i["target_pathologies"])
            phisical_count.append(i["physical_status"])
            position_count.append(i["position"])
            corrupted_count.append(i["corrupted"])
            prediction_count.append(i["predictions"])

        clean_data.append(i)

    miscelanous_data = miscelanous(
        [
            gender_count,
            vest_gender_count,
            vest_size_count,
            phisical_count,
            position_count,
            pat_count,
            corrupted_count,
            prediction_count,
        ],
        ["Gender", "Vest Gender", "Vest size", "Phisical status", "Position", "Pathologies", "Corrupted", "Prediction"],
    )

    if not os.path.exists("src/VRCARDIO-Testing/results"):
        os.mkdir("src/VRCARDIO-Testing/results")

    df = pd.DataFrame(full_df)
    df.to_csv("src/VRCARDIO-Testing/results/database_info.csv", index=False)

    xcell_name = "src/VRCARDIO-Testing/results/simple_database_excell.xlsx"
    df1 = pd.DataFrame(clean_data)
    age_df = age_dataframe(age_count)
    df2 = pd.DataFrame(age_df[0], columns=age_df[1])

    height_df = height_dataframe(height_count)
    df3 = pd.DataFrame(height_df[0], columns=height_df[1])

    weight_df = weight_dataframe(weight_count)
    df4 = pd.DataFrame(weight_df[0], columns=weight_df[1])

    df5 = pd.DataFrame(miscelanous_data).T
    df5.columns = ["", "Number", "%"]

    with pd.ExcelWriter(xcell_name, engine="xlsxwriter") as writer:

        df1.to_excel(writer, sheet_name="All data", index=False)
        df2.to_excel(writer, sheet_name="Ages", index=False)
        df3.to_excel(writer, sheet_name="Height", index=False)
        df4.to_excel(writer, sheet_name="Weight", index=False)
        df5.to_excel(writer, "Miscelaneous", index=False)

    logger.info("Excell downloaded, avaible in 'results' folder")


def create_complex_excell():

    ecg_das_electrodes = ["B4", "A4", "B8", "A3", "B6", "B7", "B2", "C4", "C7"]

    def miscelanous(representate_array, names_array):
        general = []
        general_counts = []
        general_percent = []
        percent = []
        for count, i in enumerate(representate_array):
            info, counts = np.unique(i, return_counts=True)
            percent = np.char.add((np.round(100 * counts / len(i), 2)).astype(str), "%")
            general = general + [""] + [names_array[count]] + info.tolist()
            general_counts = general_counts + ["", ""] + counts.tolist()
            general_percent = general_percent + ["", ""] + percent.tolist()

        return [general, general_counts, general_percent]

    logger.warning("This excell uses the 'bad indexes' obtained after procesing all the data")
    logger.info("Please be sure that all the cases are stored in local and procesed or run 'update_and_process_all()'")

    main_folder = "src/VRCARDIO-Testing/database_calls/session_data"
    users_id = os.listdir(main_folder)
    session_list = []
    bad_indexes_number = []
    bad_indexes_list = []
    bad_indexes_count = []

    ecg_das = []

    for user_id in users_id:
        user_folder = main_folder + "/" + user_id
        sessions_ids = os.listdir(user_folder)
        for session_id in sessions_ids:
            session_list.append(session_id)

            error_bool = False

            try:
                csv_data = pd.read_csv(user_folder + "/" + session_id + "/bad_electrodes.csv")

            except:
                error_bool = True
                ecg_das.append("No")

            if error_bool == False:
                csv_data = csv_data.values.flatten()

                if len(csv_data) > 15 and len(csv_data) < 50:
                    csv_data = 30
                    bad_indexes_number.append(30)
                    bad_indexes_list.append("ALL")
                    ecg_das.append("No")

                elif len(csv_data) >= 98:
                    bad_indexes_number.append(30)
                    bad_indexes_list.append("NO BEST SECONDS ERROR")
                    ecg_das.append("No")

                else:
                    bad_indexes_number.append(len(csv_data))

                    names = []
                    for i in csv_data:
                        names.append(colnames[i])

                    common_electrodes = [elemento for elemento in ecg_das_electrodes if elemento in names]

                    if common_electrodes == []:
                        ecg_das.append("Si")

                    else:
                        ecg_das.append("No")

                    bad_indexes_list.append(" ".join(names))
                    bad_indexes_count += names

            else:
                csv_data = ["NO BAD_ELECTRODES FILE DETECTED"]
                bad_indexes_number.append(30)
                bad_indexes_list.append("NO BAD_ELECTRODES FILE DETECTED")
                error_bool = False

    miscelanous_data = miscelanous(
        [bad_indexes_number, bad_indexes_count, ecg_das], ["Bad indexes number", "Bad indexes", "ECG DAS"]
    )

    # Load the Excel file into a DataFrame
    df_csv = pd.read_csv("src/VRCARDIO-Testing/results/database_info.csv")
    df = pd.read_excel("src/VRCARDIO-Testing/results/simple_database_excell.xlsx", sheet_name="All data")
    df2 = pd.read_excel("src/VRCARDIO-Testing/results/simple_database_excell.xlsx", sheet_name="Ages")
    df3 = pd.read_excel("src/VRCARDIO-Testing/results/simple_database_excell.xlsx", sheet_name="Height")
    df4 = pd.read_excel("src/VRCARDIO-Testing/results/simple_database_excell.xlsx", sheet_name="Weight")
    df5 = pd.read_excel("src/VRCARDIO-Testing/results/simple_database_excell.xlsx", sheet_name="Miscelaneous")

    df5.columns = ["", "Number", "%"]
    df_misc = pd.DataFrame(miscelanous_data).T
    df_misc.columns = ["", "Number", "%"]

    df5 = df5._append(df_misc, ignore_index=True)

    df5 = pd.DataFrame(df5)
    # df5.columns = ["", "Number" , "%"]
    # Value you want to search for

    for count, i in enumerate(session_list):
        value_to_search = i
        new_column1 = "Num bad electrodes"
        new_column2 = "Bad electrodes"
        new_column3 = "Ecg DAS"

        # Find the row containing the specific value
        row_containing_value = df[df["session_id"] == value_to_search]

        # Assign a value to the new column in that row using .loc
        df.loc[row_containing_value.index, new_column1] = bad_indexes_number[count]
        df.loc[row_containing_value.index, new_column2] = bad_indexes_list[count]
        df.loc[row_containing_value.index, new_column3] = ecg_das[count]

        df_csv.loc[row_containing_value.index, new_column1] = bad_indexes_number[count]
        df_csv.loc[row_containing_value.index, new_column2] = bad_indexes_list[count]
        df_csv.loc[row_containing_value.index, new_column3] = ecg_das[count]

    # Print the updated DataFrame
    # df.to_excel("src/VRCARDIO-Testing/results/complex_database_excell.xlsx", sheet_name="All data", index=False)
    with pd.ExcelWriter("src/VRCARDIO-Testing/results/complex_database_excell.xlsx", engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="All data", index=False)
        df2.to_excel(writer, sheet_name="Ages", index=False)
        df3.to_excel(writer, sheet_name="Height", index=False)
        df4.to_excel(writer, sheet_name="Weight", index=False)
        df5.to_excel(writer, "Miscelaneous", index=False)

    df_csv.to_csv("src/VRCARDIO-Testing/results/complete_database_info.csv", index=False)
