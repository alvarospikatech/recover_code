"""
Author: SpikaTech
Date: 19/07/2023
Description: This file conatis a usefull group of functions to easy represent and analyse the results 
obtained in the Maf,Maf+Spatial analisys. This are just development tools to show the output
and should not be included in the final implementation.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pyvista as pv
from scipy.signal import butter, filtfilt, hilbert , get_window
import pywt
from kymatio import Scattering1D


def show_signal_matrix(signal_matrix, axis):
    """
    Function that allows both, time or frecuency matrix signals representations in a 3D plane.
    params:
        - signal_matrix (np matrix) : Group of signals packed in a matrix
        - axis (np array): temporal o frecuency axis for just one signal of the matrix.
    returns: none
    """
    # Crear figura y eje 3D
    n, _ = signal_matrix.shape
    x, y = np.meshgrid(axis, range(n))
    z = signal_matrix
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z)
    plt.show()


def calculate_spectrogram_matrix(potentials,fs):

    potentials_spectrogram = []
    Nx = int(1 * fs) #Numero de muestras de la sample, vamos a targetear 0.2s
    w = signal.get_window("hamming", Nx, fftbins=True)
    print(Nx)
    for x in potentials:
        f_axis, t_axis, Sxx = signal.spectrogram(x, fs,window=w, return_onesided=True)
        potentials_spectrogram.append(Sxx)

    return np.array(potentials_spectrogram),f_axis,t_axis

def calculate_modelogram_matrix(potentials,fs,t):

    potentials_spectrogram = []
    Nx = int(1 * fs) #Numero de muestras de la sample, vamos a targetear 0.2s
    w = signal.get_window("hamming", Nx, fftbins=True)
    max_cuts = potentials.shape[1] / Nx
    max_cuts = int(max_cuts + int(max_cuts/int(max_cuts)))
    t_axis = np.arange(0,max_cuts)
    print(max_cuts)
    print(potentials.shape)
    print(Nx)
    print("___")
    for i in np.arange(1,max_cuts+1):
        print(i)
        if i < max_cuts:
            potentials_window = np.zeros((potentials.shape[0],Nx))
            potentials_window = potentials[:,(i-1)*Nx : i *Nx ]
            print(potentials_window.shape)
        else:
            print("LAST")
            potentials_window = np.zeros((potentials.shape[0],Nx))
            last_idx =  potentials.shape[1] - (i-1)*Nx
            print(last_idx)
            if last_idx == Nx:
                potentials_window = potentials[:,(i-1)*Nx : i *Nx ]

            else:
                print(potentials_window.shape)
                potentials_window[:,0:last_idx] = potentials[:,(i-1)*Nx : ]
                print(potentials_window.shape)

        window_spectrogram = []
        for x in potentials_window:
            f_axis, _, Sxx = signal.spectrogram(x, fs,window=w, return_onesided=True)
            window_spectrogram.append(Sxx.reshape(Sxx.shape[0]))


        print(np.array(window_spectrogram).shape)
        potentials_spectrogram.append(window_spectrogram)

    print(len(potentials_spectrogram))
    potentials_spectrogram = np.array(potentials_spectrogram)
    potentials_spectrogram = np.transpose(potentials_spectrogram, (1, 2, 0))
    return potentials_spectrogram,f_axis,t_axis


def calculate_spectrum_matrix(t, potentials):

    """
    function that recives the temporal matrix signal and converts it into a frecuncy matrix.

    params:
      - t (np.array): temporal axis for one signal
      - potentials(np.matrix): Temporal signal matrix
    """
    spectrum_matrix = []
    for i in np.arange(0, potentials.shape[0]):
        signal = potentials[i]
        fft_signal = np.real( np.fft.fft(signal))
        # Calcula las frecuencias correspondientes
        dt = t[1] - t[0]  # Paso de tiempo
        frequencies = np.fft.fftfreq(len(signal), dt)
        spectrum_matrix.append(fft_signal)

    
    """
    show_signal_matrix(
        np.array(spectrum_matrix)[:, 0 : int(len(frequencies) / 2) - 1400],
        frequencies[0 : int(len(frequencies) / 2) - 1400],
    )
    """

    return frequencies[0 : int(len(frequencies) / 2)], np.array(spectrum_matrix)[:,0 : int(len(frequencies) / 2)]


def show_nicole_plot(t_axis, y, y_noise, y_est, f_axis, y_noised_espectrum, y_est_espectrum):
    """
    Function that shows a group of diferent representations of the results thats has been done multiple times along
    the diferent papers for sinthetic data testing.
        - First plot: Real signal compared with estimated signal in time
        - Second plot: FFT of the contaminated signal
        - Residuals
        - TF of the estimated signal

    params:
        - t_axis (np.array): temporal axis of the diferent signals
        - y (np.array): clean generated signal
        - y_noise (np.array): Clean signal contaminated with noise
        - y_est (np.array): signal estimated using MAF
        - f_axis (np.array): frecuency axis for the FFT signals
        - y_noised_espectrum (np.array): Contaminated signal FFT
        - y_est_espectrum (np.array): Estimated signal FFT

    """

    _, axs = plt.subplots(2, 2)
    # Plot 1 -> Señal real + estimada
    axs[0, 0].plot(t_axis, y)
    axs[0, 0].plot(t_axis, y_est)
    axs[0, 0].set_title("Signal & Estimated signal")

    # Plot 2 -> TF señal real
    axs[0, 1].plot(f_axis, y_noised_espectrum)
    axs[0, 1].set_title("Contaminated signal spectrum")

    # Plot 3 -> Residuo
    axs[1, 0].plot(t_axis, y_noise - y_est)
    axs[1, 0].set_title("Residuals")

    # Plot 4 -> TF señal estimada.
    axs[1, 1].plot(f_axis, y_est_espectrum)
    axs[1, 1].axvline(15, color="black", linestyle="--", label="current position")
    axs[1, 1].set_title("Estimated signal spectrum")

    # Ajustar el espacio entre los subplots
    plt.tight_layout()

    # Mostrar la figura
    plt.show()


def moving_spectrum(vertices, triangles, signal, f_axis, catheter_pos=None, step=1):

    """
    This code is a modification of a previous used figure that allows to show a 3D representation
    along the frecuency axis of the spectrum in a 3D mesh.

     - vertices: vertex of the mesh
     - triangles: faces of the mesh
     - signal: frecuency signal matrix calculated with the FAM
     - f_axis: frecuency axis
    """
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
        text_actor.SetText(0, f"Hz: { np.round(f_axis[position],2)} / {np.round(f_axis[-1],2)}")

    def move_signal(forward=True):
        nonlocal position
        if forward:
            if position + step >= f_axis.shape[0]:
                position = 0
            else:
                position = position + step
        else:
            if position - step <= 0:
                position = f_axis.shape[0] - 1
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

        for i, pidx in enumerate(points):
            plt.plot(f_axis, raw_signal[pidx, :], label=f"signal {i}")
            print("HEY")
            print(pidx)

        plt.axvline(f_axis[position], color="black", linestyle="--", label="current position")
        plt.legend()
        plt.show()

    update_text(position)
    pl.add_key_event("p", do_plots)
    pl.add_key_event("g", clear)
    pl.add_key_event("s", lambda: move_signal(True))
    pl.add_key_event("a", lambda: move_signal(False))
    pl.enable_surface_picking(callback=callback, left_clicking=True, show_point=False)
    pl.show()



def spectrogram_3D(vertices, triangles, signal_input, f_axis, t_axis,max_frec = 200):

    """
    This code is a modification of a previous used figure that allows to show a 3D representation
    along the frecuency axis of the spectrum in a 3D mesh.

     - vertices: vertex of the mesh
     - triangles: faces of the mesh
     - signal_input: frecuency signal matrix calculated with the FAM
     - f_axis: frecuency axis
    """
    catheter_pos=None
    step=1
    raw_signal = signal_input.copy()
    signal_input = raw_signal

    faces = np.hstack((np.full((triangles.shape[0], 1), 3), triangles))
    mesh = pv.PolyData(vertices, faces, n_faces=triangles.shape[0])
    dif = np.abs(f_axis - max_frec)
    index = np.argmin(dif)
    signal_input = signal_input[:,0:index,:]
    f_axis =f_axis[0:index]
    mesh.point_data["scalars"] = signal_input[:, 0,0]

    #pv.set_jupyter_backend('trame')
    pl = pv.Plotter()
    pl.add_mesh(mesh, opacity=1, name="data", cmap="gist_rainbow", clim=[signal_input.min(), signal_input.max()])
    pl.add_axes()

    if catheter_pos is not None:
        actor = pl.add_points(
            catheter_pos,
            style="points",
            point_size=10,
        )

    position = 0
    moment = 0
    text_actor = pl.add_text(text="", position="upper_right", color="blue", font_size=10)
    menu_actor = pl.add_text(
        """
    Right click: add point
    p: plot signal_inputs from the marked points
    g: clear points
    s: move forward spectrum
    a: move backward spectrum
    x: move forward time
    z: move backwards time""",
        position="upper_right",
        color="blue",
        font_size=10,
    )

    def update_text(position,moment):
        text_actor.SetText(0, f"Hz: { np.round(f_axis[position],2)} / {np.round(f_axis[-1],2)}")
        text_actor.SetText(1, "window (seg): " + str(moment))

    def move_spectrum(forward=True):
        nonlocal position
        if forward:
            if position + step >= f_axis.shape[0]:
                position = 0
            else:
                position = position + step
        else:
            if position - step <= 0:
                position = f_axis.shape[0] - 1
            else:
                position = position - step

        update_text(position,moment)
        pl.update_scalars(signal_input[:, position,moment], mesh=mesh)

    
    def move_timeWindow(forward=True):
        nonlocal moment
        if forward:
            if moment+ step >= t_axis.shape[0]:
                moment= 0
            else:
                moment= moment+ step
        else:
            if moment- step <= 0:
                moment= t_axis.shape[0] - 1
            else:
                moment= moment- step

        update_text(position,moment)
        pl.update_scalars(signal_input[:, position,moment], mesh=mesh)

    points = []
    actors = []

    def callback(point):
        idx = np.argmin(np.linalg.norm(vertices - point, axis=1))
        print("HEYYYY")
        print(idx)
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

        
        #f, t, Sxx = signal.spectrogram(raw_signal[points[-1], :], f_sampling)
        dif = np.abs(f_axis - max_frec)
        index = np.argmin(dif)

        print(t_axis.shape)
        print(f_axis.shape)
        print(signal_input[points[-1]].shape)
        print("______________________________________")
        plt.pcolormesh(t_axis, f_axis[0:index], signal_input[points[-1]][0:index,:], shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    update_text(position,moment)
    
    pl.add_key_event("p", do_plots)
    pl.add_key_event("g", clear)
    pl.add_key_event("s", lambda: move_spectrum(True))
    pl.add_key_event("a", lambda: move_spectrum(False))
    pl.add_key_event("x", lambda: move_timeWindow(True))
    pl.add_key_event("z", lambda: move_timeWindow(False))
    pl.enable_surface_picking(callback=callback, left_clicking=True, show_point=False)
    pl.show()



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
        

        def cepstrum(s):
            # Calcula la FFT de la señal
            spectrum = np.fft.fft(s)
            log_spectrum = np.log(np.abs(spectrum) + 1e-10)
            cepstrum_result = np.fft.ifft(log_spectrum)
    
            return cepstrum_result, np.angle(spectrum)
        
        def cepstrumgram(signal, sampling_rate, window_size):
            # Calcula la longitud de la señal y el número de muestras por ventana
            signal_length = len(signal)
            window_samples = int(sampling_rate * window_size)

            # Calcula el número de ventanas
            num_windows = signal_length // window_samples

            # Inicializa el cepstrumgrama
            cepstrumgram_result = np.zeros((num_windows, window_samples))

            # Aplica la ventana de Hamming
            window = get_window('hamming', window_samples)

            print(cepstrumgram_result.shape)
            for i in range(num_windows):
                # Obtiene la porción de la señal para la ventana actual
                start_index = i * window_samples
                end_index = start_index + window_samples

                # Manejo de casos en los que la última ventana no coincide completamente
                

                windowed_signal = signal[start_index:end_index] * window
                if len(windowed_signal) < window_samples:
                    break

                else:
                    # Calcula el cepstrum para la ventana actual
                    cepstrum_result, _ = cepstrum(windowed_signal)

                    # Almacena el cepstrum en el cepstrumgrama
                    cepstrumgram_result[i, :] = np.abs(cepstrum_result)

            return cepstrumgram_result

        def stacionary_wt(s):
            level = 3  # 4  # Número de niveles de descomposición
            coeffs = pywt.swt(s, "rbio3.1", level=level, trim_approx=True, norm=True)  # rbio3.3
            coeffs = np.array(coeffs)
            return coeffs
        
        def wavelet_scattering(s,t):
            # Parámetros del scattering transform
            J = 3  # Niveles de descomposición
            Q = 8  # Número de sub-bandas por nivel

            # Inicializa el objeto Scattering1D
            scattering = Scattering1D(J, t.shape[0] ,Q)
            # Calcula el scattering transform
            scattering_result = scattering(s)
            return scattering_result

        f_tf, fft, phase = fourier_tf(signal, fs)
        f_psd, psd = spectral_density(signal, fs)

        #plt.figure(figsize=(12, 6))
        plt.subplot(3, 3, 1)
        plt.plot(t, signal, label="Señal 1")
        if not (np.array_equal(aux_signal, np.array([False]))):
            amplitude = np.max(signal)
            plt.plot(t, aux_signal * amplitude, linestyle="-.", label="Aux")


        plt.xlabel("Tiempo (s)")
        plt.ylabel("Time Amplitud")
        plt.legend()

        if position != None:
            plt.axvline(position, color="black", linestyle="--", label="current position")

        # Graficar las transformadas en el dominio de la frecuencia
        plt.subplot(3, 3, 4)
        plt.plot(f_psd, psd, label="Magnitud")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("spectrum Amplitude")
        plt.xlim(0, 40)
        plt.legend()

        plt.subplot(3, 3, 7)
        plt.plot(f_tf, phase, label="Fase")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("fase")
        plt.xlim(0, 40)
        plt.ylim(-np.pi, np.pi)
        plt.legend()

        plt.yticks([-np.pi, 0, np.pi], ["-π", "0", "π"])


        plt.subplot(3, 3, 2)
        cepstrum_grama = cepstrumgram(signal,fs,1)
        cepstrum_grama  = np.clip(np.abs(cepstrum_grama), 0, np.percentile(np.abs(cepstrum_grama), 90))
        plt.imshow(cepstrum_grama, aspect='auto', cmap='viridis')
        plt.ylabel("cepstrumgrama")

        plt.subplot(3, 3, 5)
        cepstrum_mag , cepstrum_phase = cepstrum(signal)
        cepstrum_mag = cepstrum_mag[0: int(len(cepstrum_mag)/2)]
        plt.plot(cepstrum_mag, label="Magnitud")
        plt.ylabel("cepstrum")

        plt.subplot(3, 3, 8)
        cepstrum_phase = cepstrum_phase[0: int(len(cepstrum_phase)/2)]
        plt.plot(cepstrum_phase, label="Magnitud")
        plt.ylabel("Cepstrum phase")

        plt.subplot(3, 3, 9)
        coeffs = stacionary_wt(signal)
        plt.imshow(np.array(coeffs), aspect='auto', cmap='viridis') #, extent=[0, 5]
        plt.ylabel("Wavelet")


        plt.subplot(3, 3, 3)
        result = wavelet_scattering(signal,t)
        plt.imshow(result, aspect='auto', cmap='viridis') #, extent=[0, 5]
        plt.ylabel("Wavelet scattering")

        plt.subplot(3, 3, 6)
        result = wavelet_scattering(np.abs(cepstrum_mag),t[0: int(len(t)/2)])
        plt.imshow(result, aspect='auto', cmap='viridis') #, extent=[0, 5]
        plt.ylabel("Wavelet scattrum")
        plt.xlim(0, len(result))
        #plt.ylim(0, len(result))


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
            print("HEYYYY")
            print(pidx)
            inner_representate_signal(x_axis, raw_signal[pidx, :], fs, aux_signal[pidx, :], position, plt)
            
        plt.show()

    update_text(position)
    pl.add_key_event("p", do_plots)
    pl.add_key_event("g", clear)
    pl.add_key_event("s", lambda: move_signal(True))
    pl.add_key_event("a", lambda: move_signal(False))
    pl.enable_surface_picking(callback=callback, left_clicking=True, show_point=False)
    pl.show()
