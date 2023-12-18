import sys

import numpy as np
import pandas as pd
from af_location_loop import af_location, save_data_file
from loguru import logger

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests")
import common_tools


def vicente_petition():
    """ """

    fa_list = [
        "15519483-ac18-426e-9d76-d0d5ab0fccb7",
        "b9de40d0-a4e0-4b07-bd89-0e951366e3d8",
        "25e1dfef-d986-4dd1-a352-049b39703020",
        "546db965-d752-4844-a808-97436d4a5ccc",
        "214a42fb-3e8e-49bf-8404-9ac7f3b175c5",
        "3b4af681-62ba-4335-aa63-70b81e007587",
    ]

    sinus_list = [
        "d9cdb542-0ef4-4e39-9dc0-b56df77dbbac",
        "03300a49-fce3-4eb5-8e01-3b8992c74690",
        "47ce40c8-54d9-4ab9-a9af-a2cdf1fe6b2b",
        "7ebf6252-1078-4a0c-afe1-3b77e7fd5a12",
        "a6b1e25c-bcba-471f-be16-092ec30f481c",
        "0cf3306a-d4c9-42b7-99b1-b1a332ea2cc6",
        "248a1165-ecbf-4253-ae5f-efc7174fa32b",
        "1643eec3-33d6-4086-a7b4-3ce5c562f085",
        "7a4c7a05-a39e-4e9a-883a-13d071407e67",
        "72c67837-5eeb-4093-bbf1-4c89141c92b3",
        "94ae8044-b21a-4c5d-bd5c-6b216b2d85c9",
        "ace6a643-fb51-49fe-8389-fec3bd0e0086",
        "d187eda5-e74d-486b-92b2-823f87b9d689",
        "0af0be97-5bd4-47f9-b8ce-0f5f25f857b0",
        "f90f6c64-d338-41cf-b9ee-ea0a21345c90",
        "8fc65ea4-b0d7-4d1b-a590-7b61c5dbe501",
        "ebb964c6-7699-4865-954f-42336c59a9ac",
        "decb5935-9151-45ba-8e4e-014b9b5925a5",
        "ab94e35a-a757-422a-be22-07d2e6ce1f89",
        "8aaff2f7-d746-429a-a95a-213cacf740e6",
        "40efbcd3-7179-4ee4-ab4d-4748d8918309",
        "57db4854-80c5-4d62-bd41-0fc8c229531b",
        "ff1b40c5-b9db-4555-b3cb-2e2b9b62ea30",
        "1dd2d5aa-8dd0-4ed8-a137-2825d9affd3f",
        "3b7ece73-2a68-4d09-817c-d8a658ac4501",
        "99f849fe-fd30-40ca-933e-7e3ad4a2eb2b",
        "0b863266-4dd3-450b-8983-4980439afb34",
        "fc29de8f-c2fc-40f3-a582-b26ce98f2a83",
        "1602bf27-5faf-45e7-9934-344127df1ce7",
        "dfba02b2-1e5a-42bd-8206-50d4e25f4aee",
        "6546aef8-9134-474c-ab34-b20f0dae7f5f",
        "0513df14-cd0b-4e60-acfb-429037f03871",
        "9100b104-c33e-48ec-9817-7109a47e1285",
        "ddcfa901-0a7a-43a0-a879-2b37abcf8bbe",
        "e5f1d953-5346-4bab-970e-24021c3d6a65",
        "ff82193d-497e-4b38-896f-e189e664b3bf",
        "bc68f723-32cb-4945-b6a0-604ac9829019",
    ]

    todo_list = sinus_list + fa_list
    todo_list = list(set(todo_list))

    todo_list = common_tools.get_subset(400, 0)
    todo_list = todo_list + [{"session_id":"15519483-ac18-426e-9d76-d0d5ab0fccb7"},{"session_id": "546db965-d752-4844-a808-97436d4a5ccc"}]

    for id in todo_list:

        session = common_tools.get_especific_session(id["session_id"])
        logger.info(session)
        case = common_tools.load_case(session)
        try:
            _, fo_array = af_location(case, file="neurokit_local_")

            fo_array = np.array([str(sublist) for sublist in fo_array])
            # fo_array = np.random.choice(fo_array, size=1200, replace=False)  #Esta linea se queda con algunas muestras
            id_array = np.full(fo_array.shape, case["session_id"])
            victoria_array = np.full(fo_array.shape, case["FA_en_ECGs_DAS"])

            vicente_df = pd.DataFrame([id_array, victoria_array, fo_array]).T
            vicente_df.columns = ["session_id", "FA en ECG", "posibles fo"]
            nombre_archivo = "vicente_session.xlsx"

            # Guardar el DataFrame en un archivo Excel

            try:
                df = pd.read_excel("src/VRCARDIO-Testing/results" + "/" + nombre_archivo)

                df = pd.concat([df, vicente_df])
                df.to_excel("src/VRCARDIO-Testing/results" + "/" + nombre_archivo, index=False)

            except:
                vicente_df.to_excel(
                    "src/VRCARDIO-Testing/results" + "/" + nombre_archivo, index=False
                )  # El argumento index=False evita que se escriba el índice en el archivo

        except:
            print("Error")
            pass

vicente_petition()
