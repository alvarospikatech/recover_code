import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pywt


def passband_filter(input_signal, low_fc, high_fc, fs):
    # Definir las frecuencias de corte del filtro pasabanda

    input_signal = input_signal / np.max(np.abs(input_signal))
    # Calcular las frecuencias normalizadas
    low_fc_norm = low_fc / (fs / 2)
    high_fc_norm = high_fc / (fs / 2)

    # Crear el filtro pasabanda
    order = 2
    sos = signal.butter(order, [low_fc_norm, high_fc_norm], btype="band", output="sos")

    # Aplicar el filtro ala señal de entrada
    output_signal = signal.sosfilt(sos, input_signal)
    output_signal[output_signal < 0.1] = 0

    frequencies, response = signal.sosfreqz(sos, worN=8000, fs=fs)
    numerator, denominator = signal.sos2tf(sos)
    w, group_delay_values = signal.group_delay((numerator, denominator), w=frequencies, fs=fs)

    # Compensation
    generalized_delay = np.max(group_delay_values) / 2
    output_signal = np.roll(output_signal, -1 * int(generalized_delay))

    """# Graficar la señal y el retardo de grupo
    plt.subplot(3, 1, 1)
    plt.plot(frequencies, 20 * np.log10(np.abs(response)))
    plt.title('Respuesta en Frecuencia del Filtro Pasa Banda')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [dB]')
    plt.grid()

    # Retardo de grupo
    plt.subplot(3, 1, 2)
    plt.plot(w, group_delay_values)
    plt.title('Retardo de Grupo del Filtro Pasa Banda')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Retardo de Grupo [muestras]')
    plt.grid()

    plt.subplot(3, 1, 3)

    plt.plot( frequencies, -1/ np.gradient(np.angle(response), frequencies))
    plt.title('Retardo de Grupo del Filtro Pasa Banda')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Retardo de Grupo [muestras]')
    plt.grid()


    plt.tight_layout()
    plt.show()"""

    return output_signal


def running_means(input):
    # Especifica el ancho de la ventana de la media móvil
    window = 30
    media_movil = np.convolve(input, np.ones(window) / window, mode="same")
    return media_movil


def detect_areas(input, front_margin, back_margin):

    input = input / np.max(np.abs(input))
    # derivada = np.abs(np.gradient(input))
    idx = np.where((input != 0))[0]
    output = np.zeros(input.shape)
    if len(idx) != len(output):
        output[idx] = 1

    """5/0

    if front_margin > 0:
        front_margin_samples = int(len(input) * front_margin / 100)
        prev_margin = np.array([])
        print("se deberia expandir por la derecha")

        added = [idx[0] - front_margin_samples, idx[0]]
        print(added)
        if idx[0] - front_margin_samples > 0:
            #aumentar margen
            print("Ha entrado")
            prev_margin = np.arange(idx[0] - front_margin_samples, idx[0], 1)
            idx = np.hstack((prev_margin, idx))
        else:
            prev_margin = np.arange(0, idx[0], 1)
            idx = np.hstack((prev_margin, idx))

    if front_margin > 0:
        back_margin_samples = int(len(input) * back_margin / 100)
        post_margin = np.array([])
        if idx[-1] + np.abs(back_margin_samples) < len(input):
            post_margin = np.arange(idx[-1], idx[-1] + back_margin_samples, 1)
            idx = np.hstack((idx, post_margin))

        output = np.zeros(input.shape)

    if len(idx) != len(output):
        output[idx] = 1
"""
    return output


def bandwidth_detector(input_signal, t, fs, lw_fc, hi_fc, front_margin, back_margin):
    filtered_signal = passband_filter(input_signal, lw_fc, hi_fc, fs)
    derivated_signal = np.gradient(filtered_signal, t)
    rectified_signal = np.abs(derivated_signal)
    smoothed_signal = running_means(rectified_signal)
    QRS = detect_areas(smoothed_signal, back_margin, front_margin)

    return QRS


def QRS_detector(input_signal, t, fs):
    return bandwidth_detector(input_signal, t, fs, 18, 45, 10, 30)


def QRST_detector(input_signal, t, fs):
    QRS = QRS_detector(input_signal, t, fs)
    window = 1 - 0.99 * QRS
    signal_wo_QRS = input_signal * window

    t_wave = T_detector_wavelet(signal_wo_QRS, t, fs)

    """plt.figure(figsize=(12, 6))
    plt.plot(t, QRS, label='Señal original', color='b')
    plt.plot(t, input_signal, label='Señal original', color='r')
    plt.plot(t,t_wave, label='Señal original', color='g')
    plt.show()"""

    QRS_T = QRS + t_wave
    QRS_T[QRS_T > 1] = 1



    return QRS_T


def T_detector(clean_sig, t, fs, witdh=0.11):

    thres = 0.6 * np.max(np.abs(clean_sig))
    peaks, _ = signal.find_peaks(np.abs(clean_sig), height=thres)

    rep_peaks = np.zeros(clean_sig.shape)
    rep_peaks[peaks] = 1


    square_pulse = np.ones(int(witdh * fs))
    t_wave = np.convolve(rep_peaks, square_pulse, mode="same")
    t_wave[t_wave > 0] = 1

    return t_wave


def T_detector_wavelet(clean_sig, t, fs, witdh=0.11):

    #clean_sig = passband_filter(clean_sig, 0.3, 20, fs)

    level = 3#4  # Número de niveles de descomposición
    coeffs = pywt.swt(clean_sig, 'rbio3.3', level=level, trim_approx=True , norm=True) #rbio3.3
    
    coeffs = np.array(coeffs)
    """plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, clean_sig, label='Señal original', color='b')
    plt.xlim(0,5)
    plt.legend()

    
    plt.subplot(2, 1, 2)
    plt.imshow(np.array(coeffs), aspect='auto', cmap='viridis') #, extent=[0, 5]
    plt.show()"""

    filtered_coeffs = coeffs.copy()
    filtered_coeffs[np.abs(filtered_coeffs) < np.max(np.abs(filtered_coeffs))*0.5] = 0

    negative_filtered_coeffs = coeffs.copy()
    negative_filtered_coeffs[np.abs(filtered_coeffs) > np.max(np.abs(filtered_coeffs))*0.5] = 0

    # Reconstruir la señal después del thresholding


    """plt.figure(figsize=(12, 6))
    plt.subplot(4, 1, 1)
    plt.plot(t, clean_sig, label='Señal original', color='b')
    plt.xlim(0,5)
    plt.legend()

    
    plt.subplot(4, 1, 2)
    plt.imshow(np.array(coeffs), aspect='auto', cmap='viridis') #, extent=[0, 5]
    plt.show()"""


    reconstructed_signal = pywt.iswt(filtered_coeffs, 'rbio3.3') #rbio3.3


    # Graficar la señal original y la reconstruida
    """plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, clean_sig, label='Señal original', color='b')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(reconstructed_signal, label=f'Señal Reconstruida (Umbral={0.85})', color='r')
    plt.legend()


    plt.tight_layout()
    plt.show()"""




    if (np.max(np.abs(reconstructed_signal)) != 0):
        t_wave = np.abs(reconstructed_signal) / np.max(np.abs(reconstructed_signal))
        t_wave = running_means(t_wave)
        t_wave[t_wave > 0] = 1
    else:
        t_wave = np.zeros(reconstructed_signal.shape)
    

    round2_sig = pywt.iswt(negative_filtered_coeffs, 'rbio3.3') #np.max(np.abs(clean_sig)) * reconstructed_signal/np.abs(np.max(reconstructed_signal))

    coeffs = pywt.swt(round2_sig, 'db20', level=level, trim_approx=True , norm=True) 
    coeffs = np.array(coeffs)

    """
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, round2_sig, label='Señal original', color='b')
    plt.xlim(0,5)
    plt.legend()

    
    plt.subplot(2, 1, 2)
    plt.imshow(np.array(coeffs), aspect='auto', cmap='viridis') #, extent=[0, 5]
    plt.show()
    """


    coeffs[np.abs(coeffs) < np.max(np.abs(coeffs))*0.4] = 0


    reconstructed_signal = pywt.iswt(coeffs, 'db20') #rbio3.3

    if (np.max(np.abs(reconstructed_signal)) != 0):
        t_wave2 = np.abs(reconstructed_signal) / np.max(np.abs(reconstructed_signal))
        t_wave2 = running_means(t_wave)
        t_wave2[t_wave2 > 0] = 1
    else:
        t_wave2 = np.zeros(reconstructed_signal.shape)


    return t_wave + t_wave2


def multiple_signals_QRS(input_signal, t, fs):

    multiple_output = np.zeros(input_signal.shape)
    for idx, i in enumerate(input_signal):
        multiple_output[idx] = QRS_detector(i, t, fs)

        """plt.plot(t, i , label=f"signal {i}")
        plt.plot(t, multiple_output[idx] ,linestyle="-.", label=f"Aux Signal {i}")
        plt.show()"""

    return multiple_output


def multiple_signals_QRST(input_signal, t, fs):

    multiple_output = np.zeros(input_signal.shape)
    for idx, i in enumerate(input_signal):
        multiple_output[idx] = QRST_detector(i, t, fs)

        

    return multiple_output
