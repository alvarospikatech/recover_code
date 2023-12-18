"""
Author: SpikaTech
Date: 24/10/2023
Description:
This File contains some common functions related to all the test made.Especifically
 - Loads all the patients and session data in one complete object.
 - Can calculate the confussion matrix (TODO, that code is actually in af_location_loop.py)

"""

import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from loguru import logger
from scipy.signal import butter, filtfilt, hilbert


def get_especific_session(session_id):
    df = pd.read_csv("src/VRCARDIO-Testing/results/complete_database_info.csv")
    # print(df.columns)
    session = df[df["session_id"] == session_id]
    return session.to_dict(orient="records")[0]


def get_subset(num_sessions, num_bad_elect, ecg_das=False):
    # df = pd.read_csv("src/VRCARDIO-Testing/results/complete_database_info.csv")
    df = pd.read_csv("src/VRCARDIO-Testing/results/complete_database_info.csv")
    to_add_count = 0
    conditions_number = [ecg_das].count(True) + 1
    sessions = []
    for index, row in df.iterrows():

        if row["Num bad electrodes"] <= num_bad_elect:
            to_add_count += 1

        if row["Ecg DAS"] == "Si" and ecg_das:
            to_add_count += 1

        if to_add_count == conditions_number:
            sessions.append(row)

        to_add_count = 0

    sessions = sessions[0:num_sessions]
    if len(sessions) < num_sessions:
        logger.warning("Not enought cases, returning the maximun posible")

    return sessions


def get_victoria_subset(num_sessions, num_bad_elect, ecg_das=False, af=False):
    # df = pd.read_csv("src/VRCARDIO-Testing/results/complete_database_info.csv")
    df = pd.read_excel("src/VRCARDIO-Testing/results/database_evaluation_VRcardio_Victoria.xlsx", sheet_name="Sheet1")
    to_add_count = 0
    conditions_number = [ecg_das, af].count(True) + 1
    sessions = []
    for index, row in df.iterrows():

        if row["Num bad electrodes"] <= num_bad_elect:
            to_add_count += 1

        if row["Ecg DAS"] == "Si" and ecg_das:
            to_add_count += 1

        if row["FA_en_ECGs_DAS"] == "SI" and af:
            to_add_count += 1

        if to_add_count == conditions_number:
            sessions.append(row)

        to_add_count = 0

    sessions = sessions[0:num_sessions]
    if len(sessions) < num_sessions:
        logger.warning("Not enought cases, returning the maximun posible")

    return sessions


def isInMiliMeters(vertexBody, facesBody, vertexHeart, facesHeart):
    def meshBodyHeart(vertexBody, facesBody, vertexHeart, facesHeart, scale):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.plot_trisurf(
            vertexBody[:, 0], vertexBody[:, 1], vertexBody[:, 2], triangles=facesBody, color="green", alpha=0.2
        )
        ax.plot_trisurf(vertexHeart[:, 0], vertexHeart[:, 1], vertexHeart[:, 2], triangles=facesHeart, color="red")

        ax.set_xlim3d(-scale, scale)
        ax.set_ylim3d(-scale, scale)
        ax.set_zlim3d(-scale, scale)

        plt.show()

    # centro_de_masa = vertexBody.mean(axis=0)
    # vertexBody = vertexBody - centro_de_masa
    # centro_de_masa = vertexHeart.mean(axis=0)
    # vertexHeart = vertexHeart - centro_de_masa
    meshBodyHeart(vertexBody, facesBody, vertexHeart, facesHeart, 500)


def confusion_matrix_stats(etiquetas_verdaderas, predicciones):

    indices = [i for i, elemento in enumerate(etiquetas_verdaderas) if elemento == "NotCount"]

    for indice in reversed(indices):
        etiquetas_verdaderas = np.delete(etiquetas_verdaderas, indice)
        predicciones = np.delete(predicciones, indice)

    etiquetas_verdaderas = etiquetas_verdaderas == "True"
    predicciones = predicciones == "True"

    false_positive_counter = 0
    false_negative_counter = 0
    true_positive_counter = 0
    true_negative_counter = 0
    aciertos = 0
    for i in range(0, len(etiquetas_verdaderas)):

        if etiquetas_verdaderas[i] == False and predicciones[i] == True:
            false_positive_counter += 1

        if etiquetas_verdaderas[i] == True and predicciones[i] == False:
            false_negative_counter += 1

        if predicciones[i] == True and etiquetas_verdaderas[i] == True:
            true_positive_counter += 1

        if predicciones[i] == False and etiquetas_verdaderas[i] == False:
            true_negative_counter += 1

        if predicciones[i] == etiquetas_verdaderas[i]:
            aciertos += 1

    fp = false_positive_counter  # / len(save_list)
    fn = false_negative_counter  # / len(save_list)
    tp = true_positive_counter  # / len(save_list)
    tn = true_negative_counter  # / len(save_list)

    print("               Real positive | Real negative")
    print("Pred Positive:" + str([tp, fp]))
    print("Pred Negative:" + str([fn, tn]))

    try:
        precision = tp / (tp + fp)
    except:
        precision = 0

    try:
        recall = tp / (tp + fn)
    except:
        recall = 0

    try:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    except:
        accuracy = 0

    try:
        f1 = 2 * tp / (2 * tp + fp + fn)
    except:
        f1 = 0

    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Accuracy: " + str(accuracy))
    print("F1: " + str(f1))

    return [fn, fp, tp, tn, accuracy, precision, recall, f1]


def load_case(case):

    """
    Function that loads all the relevant information and datasets (signals and meshes) in a compact
    object formats that allows an easy use interoperation of the data in the tests. The object contains:

    - torso_vertices (np.array): Torso vertex positions in a n_points X [x,y,z] coordinates format
    - torso_faces (np.array): As a Face is made by 3 points, it returns a list with the torso_vertices
                              that made a face (n_faces X [idx_1,idx_2,idx_3])

    - heart_vertices (np.array): Heart vertex positions in a n_points X [x,y,z] coordinates format
    - heart_faces (np.array): As a Face is made by 3 points, it returns a list with the heart_vertices
                              that made a face (n_faces X [idx_1,idx_2,idx_3])


    - electrodes (np array)

    - raw_bad_electrodes (np.array): contains the content of bad_electrodes.csv, a file with the electrodes
                                     indexes of the bad electrodes that should not be considered.

    - best_seconds
    - mesh_electrodes
    - mesh_bad_electrodes
    - mesh_good_electrodes:
    - vest : Talla del chaleco + Genero del chaleco   (male_m) (female_m)
    - unknown_indexes
    - Unknow electrodes --> Lista con los electrodos a interpolar (incluyendo los de los electrodos malos)
    - measures:
    - shorted_measures:
    - torso_potentials:
    - api_heart_potentials:
    - local_heart_potentials:

    - FA_en_ECGs_DAS

    """

    main_folder = "src/VRCARDIO-Testing/database_calls/session_data"
    session_folder = main_folder + "/" + case["user_id"] + "/" + case["session_id"] + "/"

    session_obj = {}
    session_obj["session_id"] = case["session_id"]
    session_obj["user_id"] = case["user_id"]
    session_obj["arrhytmia_prediction"] = case["arrhytmia_prediction"]

    session_obj["raw_bad_electrodes"] = pd.read_csv(
        session_folder + "bad_electrodes.csv", header=None, index_col=False
    ).values.flatten()
    session_obj["electrodes"] = pd.read_csv(
        session_folder + "electrodes.tsv", sep="\t", header=None, index_col=False
    ).values

    try:
        session_obj["best_seconds"] = pd.read_csv(
            session_folder + "best_seconds_filtered.csv", header=None, index_col=False
        ).values.T

    except:
        print("NO BEST SECONDS")


    try:
        session_obj["mesh_electrodes"] = pd.read_csv(
            session_folder + "electrode_indexes.csv", header=None, index_col=False
        ).values.flatten()

        session_obj["mesh_bad_electrodes"] = session_obj["mesh_electrodes"][[session_obj["raw_bad_electrodes"]]].flatten()
        mask = np.isin(session_obj["mesh_electrodes"], session_obj["mesh_bad_electrodes"])
        session_obj["mesh_good_electrodes"] = session_obj["mesh_electrodes"][~mask]

        torso_mesh_index = np.arange(0, torsoMesh_nodes.shape[0])

        mask = np.isin(torso_mesh_index, session_obj["mesh_good_electrodes"])
        session_obj["unknown_indexes"] = torso_mesh_index[~mask]

    except:
        pass

    
    try:
        # Devuelve los mejores segundos solo para los electrodos buenos
        session_obj["electrodes_best_seconds"] = session_obj["best_seconds"][~mask]
    except:
        pass

    heart_load = trimesh.load_mesh(session_folder + "/heart.stl")
    heartMesh_nodes = np.array(heart_load.vertices) * 1000  # Checkear si las mallas estan en m
    heartMesh_faces = np.array(heart_load.faces)

    torso_load = trimesh.load_mesh(session_folder + "/torso.stl")
    torsoMesh_nodes = np.array(torso_load.vertices) * 1000  # Checkear si las tallas estan en m
    torsoMesh_faces = np.array(torso_load.faces)

    session_obj["heart_faces"] = heartMesh_faces
    session_obj["heart_points"] = heartMesh_nodes

    session_obj["torso_faces"] = torsoMesh_faces
    session_obj["torso_points"] = torsoMesh_nodes

    """
    isInMiliMeters(
        session_obj["torso_points"], session_obj["torso_faces"], session_obj["heart_points"], session_obj["heart_faces"]
    )
    """
    vest_gender = case["vest gender"]
    if vest_gender == "M":
        vest_gender = "male"
    elif vest_gender == "W":
        vest_gender = "female"
    elif vest_gender == "U":
        vest_gender = "unisex"

    vest_size = case["vest size"].lower()
    vest = vest_gender + "_" + vest_size

    session_obj["vest"] = vest

    

    measures = np.zeros((torsoMesh_nodes.shape[0], session_obj["best_seconds"].shape[1]))

    for idx, i in enumerate(session_obj["best_seconds"]):
        measures[session_obj["mesh_electrodes"][idx]] = i

    session_obj["measures"] = measures

    session_obj["shorted_measures"] = measures[:, 0 : int(measures.shape[1] / 3)]

    session_obj["torso_potentials"] = pd.read_csv(
        session_folder + "torso_potentials.csv", header=None, index_col=False
    ).values.T

    try:
        session_obj["api_heart_potentials"] = pd.read_csv(
            session_folder + "api_heart_potentials.csv", header=None, index_col=False
        ).values.T

    except Exception:
        pass

    try:
        session_obj["wavelet_local_heart_potentials"] = pd.read_csv(
            session_folder + "wavelet_local_heart_potentials.csv", header=None, index_col=False
        ).values

    except Exception:
        pass

    try:
        session_obj["neurokit_local_heart_potentials"] = pd.read_csv(
            session_folder + "neurokit_local_heart_potentials.csv", header=None, index_col=False
        ).values

    except Exception:
        pass

    try:
        session_obj["nointerp_neurokit_local_heart_potentials"] = pd.read_csv(
            session_folder + "nointerp_neurokit_local_heart_potentials.csv", header=None, index_col=False
        ).values

    except Exception:
        pass

    try:
        session_obj["nointerp_wavelet_local_heart_potentials"] = pd.read_csv(
            session_folder + "nointerp_wavelet_local_heart_potentials.csv", header=None, index_col=False
        ).values

    except Exception:
        pass

    try:
        archivo_excel = "src/VRCARDIO-Testing/results/database_evaluation_VRcardio_Victoria.xlsx"
        df = pd.read_excel(
            archivo_excel,
        )
        fila = df[df["session_id"] == session_obj["session_id"]]
        session_obj["FA_en_ECGs_DAS"] = fila["FA_en_ECGs_DAS"].values[0]

    except Exception:
        print("Not victoria file found or no data")
        session_obj["FA_en_ECGs_DAS"] = "No Data"

    session_obj["fs"] = 360

    dur = (session_obj["torso_potentials"].shape[1] - 1) / session_obj["fs"]
    interval = 1 / session_obj["fs"]
    t_axis = np.arange(0, dur + interval / 2, interval)

    session_obj["t_axis"] = t_axis

    return session_obj



def load_case_minimun(case):
    main_folder = "src/VRCARDIO-Testing/database_calls/session_data"
    session_folder = main_folder + "/" + case["user_id"] + "/" + case["session_id"] + "/"

    session_obj = {}
    session_obj["user_id"] = case["user_id"]
    session_obj["session_id"] = case["session_id"]
    heart_load = trimesh.load_mesh(session_folder + "/heart.stl")
    heartMesh_nodes = np.array(heart_load.vertices) * 1000  # Checkear si las mallas estan en m
    heartMesh_faces = np.array(heart_load.faces)

    torso_load = trimesh.load_mesh(session_folder + "/torso.stl")
    torsoMesh_nodes = np.array(torso_load.vertices) * 1000  # Checkear si las tallas estan en m
    torsoMesh_faces = np.array(torso_load.faces)

    session_obj["heart_faces"] = heartMesh_faces
    session_obj["heart_points"] = heartMesh_nodes

    session_obj["torso_faces"] = torsoMesh_faces
    session_obj["torso_points"] = torsoMesh_nodes

    session_obj["signals_raw"] = pd.read_csv(
            session_folder + "signals_raw.csv", index_col=False
    )


    n_points = 1
    electrodes = pd.read_csv(session_folder + "electrodes.tsv", sep="\t", header=None)
    electrode_points = [re.findall(r"[^[]*\[([^]]*)\]", s)[::2] for s in electrodes[2].values.tolist()[1:]]
    electrode_points = [s.split() for s in np.array(electrode_points).ravel()]
    electrode_points = np.array(electrode_points).astype("float")
    electrode_points = electrode_points[:: 4 - n_points]

    session_obj["electrodes"] = electrode_points


    return session_obj


def representate_signal(t, signal, fs, aux_signal=[False], position=None):
    def fourier_tf(s, fs):
        fft = np.fft.fft(s) / len(s)
        phase = np.angle(fft)
        freqs = np.fft.fftfreq(len(fft), 1 / fs)
        phase = np.unwrap(phase)  # Desenvolver la fase
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        return freqs, fft, phase

    def spectral_density(s, fs):
        s = 2 * np.correlate(s, s, mode="full") / len(s)
        psd = np.fft.fft(s) / len(s)
        psd_magnitude = np.real(np.abs(psd.copy()))
        freqs = np.fft.fftfreq(len(psd), 1 / fs)

        return freqs, psd_magnitude

    f_tf, fft, phase = fourier_tf(signal, fs)
    f_psd, psd = spectral_density(signal, fs)

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, signal, label="Señal 1")
    if not (np.array_equal(aux_signal, np.array([False]))):
        plt.plot(t, aux_signal, linestyle="-.", label="Aux")

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.legend()

    if position != None:
        plt.axvline(position, color="black", linestyle="--", label="current position")

    # Graficar las transformadas en el dominio de la frecuencia
    plt.subplot(3, 1, 2)
    plt.plot(f_psd, psd, label="Magnitud")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")
    plt.xlim(0, 50)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(f_tf, phase, label="Fase")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("fase")
    plt.xlim(0, 50)
    plt.ylim(-np.pi, np.pi)
    plt.legend()

    plt.yticks([-np.pi, 0, np.pi], ["-π", "0", "π"])

    plt.show()


def plot_mesh(vertices, triangles, signal, x_axis, aux_signal=["none"], catheter_pos=None, step=1, fs=380):
    def inner_representate_signal(t, signal, fs, aux_signal=["none"], position=None, plt=None):
        def normalize_func(x):
            return x / np.max(np.abs(x))

        def butter_lowpass(cutoff, fs, order=4):
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype="low", analog=False)
            return b, a

        def butter_lowpass_filter(data, cutoff, fs, order=4):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = filtfilt(b, a, data)
            return y

        def fourier_tf(s, fs):
            fft = np.fft.fft(s) / len(s)
            phase = np.angle(fft)
            freqs = np.fft.fftfreq(len(fft), 1 / fs)
            phase = np.unwrap(phase)  # Desenvolver la fase
            phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi

            return freqs, fft, phase

        def spectral_density(s, fs):
            s = 2 * np.correlate(s, s, mode="full") / len(s)
            psd = np.fft.fft(s) / len(s)
            psd_magnitude = np.real(np.abs(psd.copy()))
            freqs = np.fft.fftfreq(len(psd), 1 / fs)

            return freqs, psd_magnitude

        f_tf, fft, phase = fourier_tf(signal, fs)
        f_psd, psd = spectral_density(signal, fs)

        #plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        plt.plot(t, signal, label="Señal 1")
        if not (np.array_equal(aux_signal, np.array([False]))):
            amplitude = np.max(signal)
            plt.plot(t, aux_signal * amplitude, linestyle="-.", label="Aux")


        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.legend()

        if position != None:
            plt.axvline(position, color="black", linestyle="--", label="current position")

        # Graficar las transformadas en el dominio de la frecuencia
        plt.subplot(3, 1, 2)
        plt.plot(f_psd, psd, label="Magnitud")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Amplitud")
        plt.xlim(0, 40)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(f_tf, phase, label="Fase")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("fase")
        plt.xlim(0, 40)
        plt.ylim(-np.pi, np.pi)
        plt.legend()

        plt.yticks([-np.pi, 0, np.pi], ["-π", "0", "π"])


    if aux_signal[0][0] == "none":
        aux_signal = np.zeros(signal.shape)
        aux_signal[aux_signal == 0] = False

    raw_signal = signal.copy()
    signal = raw_signal

    faces = np.hstack((np.full((triangles.shape[0], 1), 3), triangles))
    mesh = pv.PolyData(vertices, faces, n_faces=triangles.shape[0])
    mesh.point_data["scalars"] = signal[:, 0]

    pl = pv.Plotter()
    pl.add_mesh(mesh, opacity=1, name="data", cmap="gist_rainbow", clim=[signal.min(), signal.max()])
    pl.add_axes()

    if catheter_pos is not None:
        actor = pl.add_points(
            catheter_pos,
            style="points",
            point_size=10,
        )

    position = 0
    text_actor = pl.add_text(text="", position="upper_right", color="blue", font_size=10)
    menu_actor = pl.add_text(
        """
    Right click: add point
    p: plot signals from the marked points
    g: clear points
    s: move forward
    a: move backward""",
        position="upper_right",
        color="blue",
        font_size=10,
    )

    def update_text(position):
        text_actor.SetText(0, f"Position: {position} / {signal.shape[1]}")

    def move_signal(forward=True):
        nonlocal position
        if forward:
            if position + step >= signal.shape[1]:
                position = 0
            else:
                position = position + step
        else:
            if position - step <= 0:
                position = signal.shape[1] - 1
            else:
                position = position - step

        update_text(position)
        pl.update_scalars(signal[:, position], mesh=mesh)

    points = []
    actors = []

    def callback(point):
        idx = np.argmin(np.linalg.norm(vertices - point, axis=1))
        points.append(idx)

        actor = pl.add_point_labels(point, [f"{len(points)}"])
        actors.append(actor)

    def clear():
        for actor in actors:
            pl.remove_actor(actor)

        points.clear()
        actors.clear()

    def do_plots():
        if len(points) == 0:
            return

        plt.figure(figsize=(12, 6))
        for i, pidx in enumerate(points):
            """
            plt.plot(x_axis, raw_signal[pidx, :], label=f"signal {i}")
            if not (np.array_equal(aux_signal, np.array([False]))) :
                plt.plot(x_axis, aux_signal[pidx, :], linestyle="-.", label=f"Aux Signal {i}")
            """

            inner_representate_signal(x_axis, raw_signal[pidx, :], fs, aux_signal[pidx, :], position, plt)
            
        plt.show()

    update_text(position)
    pl.add_key_event("p", do_plots)
    pl.add_key_event("g", clear)
    pl.add_key_event("s", lambda: move_signal(True))
    pl.add_key_event("a", lambda: move_signal(False))
    pl.enable_surface_picking(callback=callback, left_clicking=True, show_point=False)
    pl.show()
