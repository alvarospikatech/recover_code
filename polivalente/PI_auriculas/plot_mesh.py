import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from loguru import logger
from scipy.signal import butter, filtfilt, hilbert

def plot_mesh(vertices, triangles, signal, x_axis, aux_signal=np.array(["none"]), catheter_pos=None, step=1, fs=380):
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

    if aux_signal == ["none"]:
        aux_signal = np.zeros(signal.shape)



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