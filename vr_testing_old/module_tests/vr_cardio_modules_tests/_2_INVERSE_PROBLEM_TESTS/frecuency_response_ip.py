import sys

import IP_code
import load_data_controlled
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests")
import common_tools


def simplified_impulse_response(f_axis, real_response):
    def filter_func(x, a, b, c):

        return c * (1 / 1 + a * x + b * (x**2))

    params, params_covariance = curve_fit(filter_func, f_axis, real_response, p0=[2, 1, -125])

    plt.figure(figsize=(8, 6))
    plt.plot(f_axis, real_response, "b", label="Datos con ruido")
    plt.plot(f_axis, filter_func(f_axis, params[0], params[1], params[2]), "r", label="Ajuste no lineal")
    plt.legend(loc="best")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

    print("Parámetros del ajuste:")
    print("a =", params[0])
    print("b =", params[1])
    print("c =", params[2])


def simplified_impulse_response_2(f_axis, real_response, representate=True):

    window = 50
    wind_idx = np.arange(window, real_response.shape[0], window)
    before_idx = 0
    estimated_response = np.zeros(real_response.shape)
    for i in wind_idx:
        mean = np.mean(real_response[before_idx:i])
        estimated_response[before_idx] = mean
        before_idx = i

    zero_indices = np.where(estimated_response == 0)[0]
    nonzero_indices = np.where(estimated_response != 0)[0]
    nonzero_values = estimated_response[nonzero_indices]
    interpolator = interp1d(nonzero_indices, nonzero_values, kind="linear", fill_value="extrapolate")
    interpolated_values = interpolator(zero_indices)
    # Reemplaza los valores 0 con los valores interpolados
    estimated_response[zero_indices] = interpolated_values

    if representate:
        plt.figure(figsize=(8, 6))
        plt.plot(f_axis, real_response, "b", label="Datos con ruido")
        plt.plot(f_axis, estimated_response, "r", label="Ajuste no lineal")
        plt.legend(loc="best")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

    return estimated_response


def spectral_density(s, fs):

    #En realidad es un diagrama de bode
    fft = np.fft.fft(s) / len(s)
    fft_phase = np.real(np.angle(fft.copy()))
    freqs_fft = np.fft.fftfreq(len(fft), 1 / fs)

    s = 2 * np.correlate(s, s, mode="full") / len(s)
    psd = np.fft.fft(s) / len(s)
    psd_magnitude = np.real(np.abs(psd.copy()))
    psd_phase = np.real(np.angle(psd.copy()))
    freqs = np.fft.fftfreq(len(psd), 1 / fs)

    return freqs,freqs_fft, psd_magnitude, fft_phase


def white_noise_generator(time_dur, fs):

    t = np.linspace(0, time_dur, int(time_dur * fs), endpoint=False)
    desired_power_dbw = 1000
    power_watts = 10 ** (desired_power_dbw / 10)
    samples = np.random.normal(0, 1, len(t))
    scaling_factor = np.sqrt(power_watts / np.mean(samples**2))
    # scaling_factor = power_watts**2
    noise_signal = samples  # * scaling_factor

    # print("POWER:"  + str(10 * np.log10(np.mean(noise_signal**2))))

    return noise_signal, t


def delta_generator(time_dur, fs):

    t = np.linspace(0, time_dur, int(time_dur * fs), endpoint=False)
    delta = np.zeros(t.shape)
    delta[0] = 1

    return delta, t


def filter_signal(s, fs):
    frecuencia_corte = 240  # Frecuencia de corte del filtro en Hz
    orden_filtro = 6
    b, a = signal.butter(orden_filtro, frecuencia_corte / (fs / 2), "low")

    # Aplicar el filtro a la señal de entrada
    signal_output = signal.lfilter(b, a, s)
    return signal_output


def frec_response_inverse(torso_points, torso_faces, heart_points, heart_faces, lambda_value, representate=True):

    fs = 500
    time_dur = 10
    torso_signal = []
    torso_signal_freq = []
    t_axis = []
    for i in torso_points:
        s, t = white_noise_generator(time_dur, fs)
        #s, t = delta_generator(time_dur, fs)

        s = filter_signal(s, fs)
        torso_signal.append(s)
        t_axis = t

        f,_, psd_mag, phase = spectral_density(s, fs)
        idx_0 = int(len(f) / 2)
        f = f[0:idx_0]
        psd_mag = 10 * np.log10(psd_mag[0:idx_0])
        f_axis = f
        torso_signal_freq.append(psd_mag)

    torso_signal = np.array(torso_signal)
    torso_signal_freq = np.array(torso_signal_freq)

    if representate:
        common_tools.plot_mesh(
            vertices=torso_points, triangles=torso_faces, signal=torso_signal, x_axis=t, aux_signal=[False]
        )

    lamb = np.linspace(lambda_value, lambda_value, 1).tolist()
    print(lamb)
    IP_output = IP_code.run_inverse_clasic_managed(
        torso_points=torso_points,
        torso_faces=torso_faces,
        heart_points=heart_points,
        heart_faces=heart_faces,
        measured_potentials_position=torso_points,
        torso_signals=torso_signal,
        lamb=lamb,
    )

    spd_inverse = []
    phase_inverse = []
    f_axis = []
    f_axis_phase = []
    for i in IP_output:
        f,f_phase, psd_mag, phase = spectral_density(i, fs)
        idx_0 = int(len(f) / 2)
        f = f[0:idx_0]
        psd_mag = psd_mag[0:idx_0] #/ np.mean(psd_mag[0:idx_0])
        f_axis = f

        idx_0 = int(len(f_phase) / 2)
        f_phase = f_phase[0:idx_0]
        phase = phase[0:idx_0]
        f_axis_phase = f_phase

        spd_inverse.append(psd_mag)
        phase_inverse.append(phase)

    spd_inverse = np.array(spd_inverse)
    phase_inverse = np.array(phase_inverse)

    if representate:
        common_tools.plot_mesh(heart_points, heart_faces, IP_output, t)

    general_frec_resp = np.mean(spd_inverse, axis=0)
    general_phase_resp = np.mean(phase_inverse,axis=0)

    return f_axis,f_phase, simplified_impulse_response_2(f_axis, general_frec_resp, representate) , simplified_impulse_response_2(f_axis_phase, general_phase_resp, representate)


def launch_torso_freq_response():

    dataset = "utah_2018_avp"
    (
        m_vm,
        heart_faces,
        heart_vertices,
        m_ve,
        torso_faces,
        torso_vertices,
        torso_faces_cl,
        torso_vertices_cl,
    ) = load_data_controlled.loadDataSet(dataset)
    lambda_value = 10**-6
    frec_response_inverse(
        torso_vertices, torso_faces, heart_vertices, heart_faces, lambda_value=lambda_value, representate=True
    )


def launch_multiple_torso_freq_response():

    dataset = "utah_2002_03"

    if dataset.startswith("utah"):
        (
            m_vm,
            heart_faces,
            heart_vertices,
            m_ve,
            torso_faces,
            torso_vertices,
            torso_faces_cl,
            torso_vertices_cl,
        ) = load_data_controlled.loadDataSet(dataset)
    else:
        session = common_tools.get_subset(1, 5)[0]
        case = common_tools.load_case(session)
        heart_faces = case["heart_faces"]
        heart_vertices = case["heart_points"]
        torso_faces = case["torso_faces"]
        torso_vertices = case["torso_points"]

    lambda_value = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # np.arange(-8,3,1)

    print(lambda_value)
    responses = []

    plt.figure(figsize=(8, 6))
    for i in lambda_value:
        print("Calculating for: 10^" + str(i))
        f_axis,f_phase, r ,phase= frec_response_inverse(
            torso_vertices, torso_faces, heart_vertices, heart_faces, lambda_value=10.0**i, representate=False
        )
        plt.subplot(2, 1, 1)
        plt.legend(loc="upper right")
        plt.xlim(0, 250)
        plt.xlabel("freq")
        plt.ylabel("dB")
        plt.grid(True)
        #plt.plot(f_axis, 10 * np.log10(r), label="10^" + str(i))
        plt.plot(f_axis,r, label="10^" + str(i))

        plt.subplot(2, 1, 2)
        plt.plot(f_phase,phase)
        plt.xlim(0, 250)

        responses.append(np.max(np.abs(r)))

    
    plt.savefig("src/VRCARDIO-Testing/results/" + dataset + "_freq_res.png")
    plt.show()

    """plt.plot(np.array(lambda_value), responses)
    plt.legend(loc="upper right")
    plt.ylim(0, 3)
    plt.xlabel("10^")
    plt.ylabel("dB")
    plt.grid(True)
    # plt.savefig("src/VRCARDIO-Testing/results/" + dataset + "_freq_res.png")
    plt.show()"""


launch_multiple_torso_freq_response()
#launch_torso_freq_response()
