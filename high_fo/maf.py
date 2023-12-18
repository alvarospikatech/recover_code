"""
Author: SpikaTech
Date: 19/07/2023
Description: This file contains all the functions related to all the frencuency modeling
for the first proposal of AF detection and location.This should be considered as the part 1
of the proposal as its expected to be the input of the spatial analysis.
"""

import itertools
import math
import random
from collections import Counter

import statistics
import numpy as np
import scipy.signal as sig
from scipy import signal
import matplotlib.pyplot as plt

import pywt
from kymatio import Scattering1D


class Maf:

    """
    maf: Multicomponent Atrial Fibrilation. Algorithm that detects the atrial fibrilation by:
    """

    def improved_fo_detector(self, potentials, fs, overlap_windows=True):

        
        dur = (potentials.shape[1] - 1) / fs
        interval = 1 / fs
        t_axis = np.arange(0, dur + interval / 2, interval)
        fo_posible = []
        fo_posible_dark = []
        fo = []  # a full estimated spectrum for each node
        fo_vicente = []
        for i in np.arange(0, potentials.shape[0]):
            Nx = int(1 * fs)  # Numero de muestras de la sample, vamos a targetear 0.2s
            overlap = int(Nx/2)
            w = signal.get_window("hamming", Nx, fftbins=True)
            max_cuts = potentials.shape[1] / Nx
            max_cuts = int(max_cuts + int(max_cuts / int(max_cuts)))
            fo_posible = []
            fo_posible_dark = []
            for j in np.arange(1, max_cuts + 1):
                # print(t_axis[(i-1)*Nx : i *Nx])
                if j < max_cuts:

                    w_t_axis = t_axis[(j - 1) * Nx : j * Nx].copy()
                    w_potentials = potentials[i][(j - 1) * Nx : j * Nx].copy() * w

                    f = self.calculate_armonic_freqs(w_t_axis, w_potentials, 60)[0]
                   

                    if f != 1 and f < 12:
                        fo_posible.append(f)

                    else:
                        fo_posible.append("x")
                        
                    if overlap_windows and ((j * Nx) + overlap < t_axis.shape[0]):
                        
                        w_t_axis = t_axis[((j - 1) * Nx) + overlap : (j * Nx) + overlap].copy()
                        w_potentials = potentials[i][((j - 1) * Nx) + overlap : (j * Nx) + overlap] * w
                        f = self.calculate_armonic_freqs(w_t_axis, w_potentials, 60)[0]
                        
                        if f != 1 and f < 12:
                            fo_posible.append(f)
                            fo_posible_dark.append(f)

            

            
            #print("Overlapped fo: " + str(fo_posible_dark))
            #print("Full stack: " + str(fo_posible))
            fo_vicente.append(fo_posible)
            fo_posible = [x for x in fo_posible if x != "x"]
            contador = Counter(fo_posible)
            added_fo = 1
            mode = "moda"
            if (mode == "median"):
                try:
                    median = statistics.median(fo_posible)
                    added_fo = median
                except:
                    added_fo = 0

            elif (mode == "per_5"):
                try:
                    perc_5 = np.percentile(fo_posible,5)
                    added_fo = perc_5
                except:
                    added_fo = 0

            else:
                try:
                    repeated_value = contador.most_common(1)[0][0]
                    num_reps = contador.most_common(1)[0][1]
                    if num_reps > 1:
                        added_fo = repeated_value
                    else:
                        added_fo = np.mean(fo_posible)
                except:
                    added_fo = 0
                    

            fo.append(added_fo)
            
        return np.array(fo) , fo_vicente
    

    def obtain_peaks_map(self, potentials, fs):

        def filter_low_pass(temporal_signal,fs):
            fc = 5.0

            # Calcular la frecuencia normalizada
            fc_norm = fc / (0.5 * fs)

            orden = 1
            b, a = sig.butter(orden, fc_norm, btype='low', analog=False, output='ba')

            filtered = sig.filtfilt(b, a, temporal_signal)

            return filtered

        peaks = []
        for i in np.arange(0, potentials.shape[0]):
            temporal_sig = filter_low_pass(potentials[i,:], fs)
            n_peaks, _ = sig.find_peaks(temporal_sig)
            n_peaks = n_peaks.shape[0] / (5 *fs)
            peaks.append(n_peaks)

        peaks = np.array(peaks) #.reshape(-1,1)

        return peaks
    

    def wavelet_scatrum(self, potentials, t):

        def wavelet_scattering(s,t):
            # Parámetros del scattering transform
            J = 3  # Niveles de descomposición
            Q = 8  # Número de sub-bandas por nivel

            # Inicializa el objeto Scattering1D
            scattering = Scattering1D(J, t.shape[0] ,Q)
            # Calcula el scattering transform
            scattering_result = scattering(s)
            return scattering_result
        
        def cepstrum(s):
            # Calcula la FFT de la señal
            spectrum = np.fft.fft(s)
            log_spectrum = np.log(np.abs(spectrum) + 1e-10)
            cepstrum_result = np.fft.ifft(log_spectrum)
    
            return cepstrum_result, np.angle(spectrum)

        all_coeficients = []
        for i in np.arange(0, potentials.shape[0]):
            sig = potentials[i,:]
            cepstrum_mag , cepstrum_phase = cepstrum(sig)
            cepstrum_mag = cepstrum_mag[0: int(len(cepstrum_mag)/2)]
            coeficients = wavelet_scattering(np.abs(cepstrum_mag),t[0: int(len(t)/2)])

            short_coeficients = coeficients[0:coeficients.shape[0],0:coeficients.shape[0] ]
            
            short_coeficients = short_coeficients.flatten()

            all_coeficients.append(short_coeficients)

        all_coeficients = np.array(all_coeficients) #.reshape(-1,1)

        return all_coeficients


    def full_mesh_signal_proposal(self, potentials, fs):
        """
        Function that calculates the maf for a full 3D mesh potentials in a nodes x time estructure.

        params:
            - potentials (np array): temporal signal matrix asociates with the mesh
            - fs (np array): frecuency sampling
            - clean (boolean):Boolean that determines if some extra signal cleaning its needed. This process
                        its not from the original code and includes some extra-improvements.

        returns:
            - maf_spectrum (np array): np.array with the estimated signal frecuency matrix.
            - maf_time_signal (np array): np.array with the estimated signal time matrix.
            - f_axis (np array): frecuency axis
            - t_axis (np array): Temporal axis
        """
        fo_map, fo_vicente = self.improved_fo_detector(potentials, fs)
        fo_map = np.array(fo_map)
        peaks_map = self.obtain_peaks_map(potentials,fs)

        return fo_map, peaks_map

    def calculate_armonic_freqs(self, t, y, f_max):
        """
        Function to extract the fo and the frecuency of its armonics from the original signal.

        params:
            - t (np.array): temporal axis of the y signal
            - y (np.array): temporal input signal.

        returns:
            - f_list(np array) :array with all the estimated frecs of its components.
        """

        def frecuency_redundancy(estimated_components):
            """
            Sub-function that finds frecuency distance redundancy paterns between armonics that
            should equal to the fundamental frecuency. This is calculated by subtracting the
            nearest component frecuencies in a spectrum-likefunction, if a value is highly repeated
            ,this will be understanded as the fo*k pattern, and the fo would replacethe first peak value.

            params:
                - estimated_components (list): Array with the stimated frecuencies of all the components

            returns:
                - fo (int) = estimated fundamental frecuency

            """

            estimated_components = np.round(estimated_components, 2)
            frecs_diff = []
            repeated_flag = False
            for i in np.arange(1, len(estimated_components)):
                calc = np.abs(estimated_components[i] - estimated_components[i - 1])
                frecs_diff.append(calc)
            frecs_diff = [x for x in frecs_diff if x >= 1]
            count = {}
            for i in frecs_diff:
                if i in count:
                    count[i] += 1
                    repeated_flag = True
                else:
                    count[i] = 1

            if repeated_flag:
                arr_ordenado = sorted(frecs_diff, key=lambda x: count[x], reverse=True)
                try:
                    return arr_ordenado[0]
                except IndexError:
                    return 1

            else:
                try:
                    return estimated_components[0]
                except IndexError:
                    return 1

        def high_pass_filter(f_axis, spectrum, a):
            """
            Sub-function that slices the spectrum from "a" hz (or the closest frecuency axis value) to the
            end, acting like a high pass filter without being a clasical digitial filter in its definition.
            This is used for cleaning under 1hz values (a = 1) in order to avoid <1 hz frecuencies as a fo
            detection under this value its condered an error

            params:
                - f_axis(np.array): frecuency axis
                - spectrum(np.array): spectrum function
                - a (int): cut frecuency

            returns:
                - f_axis(np.array): sliced frecuency axis
                - spectrum(np.array): sliced spectrum

            """
            dif = np.abs(f_axis - a)
            index = np.argmin(dif)
            f_axis = f_axis[index:]
            spectrum = spectrum[index:]

            return f_axis, spectrum

        # 1 & 2. Proposal --> fft attemps
        f_axis, fft, components = self.fft(y, t)
        fft = np.abs(fft) ** 2 / np.max(np.abs(fft) ** 2)
        f_axis, fft = high_pass_filter(f_axis, fft, 1.5)

        window = np.hanning(4)
        envelope = np.convolve(window/window.sum(), fft, mode='same')

        """plt.figure(figsize=(12, 6))
        plt.plot(f_axis, fft)
        plt.plot(f_axis, envelope)
        plt.xlim(0,20)
        plt.show()"""
        fft = envelope

        components, _ = sig.find_peaks(fft, height=np.max(fft) / 4)
        try:
            f_peak_max = f_axis[components][-1]
            # 1.Propose --> First peak of the fft
            fo_0 = f_axis[components][0]
            # 2.Propose --> Finding frecuency redundancy of the fft
            fo_0_2 = frecuency_redundancy(f_axis[components])
        except IndexError:
            fo_0 = 1
            fo_0_2 = 1
            f_peak_max = f_max

        #        #3.Proposal --> Temporal correlation.
        #        y_X = np.correlate(y, y, mode='full')
        #        y_X_shorted = y_X[int(len(y_X)/2) + 20: ]
        #        y_X = y_X[int(len(y_X)/2): ]
        #        t_axis_corr = t[20:]
        #        peaks,_ = sig.find_peaks(y_X_shorted ,height= np.max(y_X_shorted)/1.75)
        #        try:
        #            f_peak_max = f_max
        #            fo_1 = np.abs(1/ (t_axis_corr[peaks[0]] - t_axis_corr[peaks[1]]))
        #        except IndexError:
        #            fo_1 = 1
        #
        #        #4 & 5. Proposal --> Spectral density attemps (fft on the autocorr)
        #
        #        f_axis, fft ,components = self.smooth_fft(y_X,t)
        #        fft = fft**2 / np.max(fft**2)
        #        plt.plot(f_axis, fft)
        #        components,_ = sig.find_peaks(fft ,height= np.max(fft)/4)
        #        try:
        #            f_peak_max = f_axis[components][-1]
        #            #4.Propose --> First peak of the Spectral density
        #            fo_2 = f_axis[components][0]
        #            #5.Propose --> frecuency redundancy of the spectral density
        #            fo_3 = frecuency_redundancy(f_axis[components])
        #        except IndexError:
        #            fo_2 = 1
        #            fo_3 = 1
        #            f_peak_max = f_max
        #
        #        #6 & 7. Proposal --> Shorted Spectrak density attemps
        #
        #        f_axis, fft ,components = self.smooth_fft(y_X_shorted,t_axis_corr)
        #        f_axis,fft = high_pass_filter(f_axis, fft, 1.5)
        #        fft = fft**2 / np.max(fft**2)
        #        components,_ = sig.find_peaks(fft ,height= np.max(fft)/4)
        #        try:
        #            f_peak_max = f_axis[components][-1]
        #            #6.Propose --> First peak of the Spectral density (shorted)
        #            fo_4 = f_axis[components][0]
        #            #7.Propose --> frecuency redundancy of the spectral density (shorted)
        #            fo_5 = frecuency_redundancy(f_axis[components])
        #        except IndexError:
        #            fo_4 = 1
        #            fo_5 = 1
        #            f_peak_max = f_max

        fo = np.round(fo_0_2, 2)

        # ABA, "Ancho de banda adaptativo". Uncomment these two next lines to activate it.
        # Its suposed to discard high componets that hasnt been detected before in the fo
        # process. This shouldnt have an impact on permormance as controlles experiments showed
        # that works exactly the same with and just a fo output regardless of the rest of the
        # components, but in uncontrolled cases has shown to be a key factor on performance.
        # if f_peak_max < f_max:
        #    f_max = f_peak_max

        f_list = np.arange(fo, f_max, fo)
        # print(f_list)
        try:
            f_list[0]
        except IndexError:
            f_list = [1]

        #print(f_list)
        return f_list

    def signal_model_matrix(self, frequencies: list, t: list, delta: float, max_frequency=60):
        """
        function that prepares the multicomponent signal model that will be regularizated later
        - frequencies (np.array): List of frecuencies of each component that the model will have
        - t (np.array): time axis that will have the signal???????
        - delta (int): indicates the diferent fluctuations of the signal.
        """
        fgrid = [list(np.arange(frequency, max_frequency, frequency)) for frequency in frequencies]
        grupo = [list(itertools.repeat(i + 1, len(fgrid[i]))) for i, element in enumerate(fgrid)]
        fgrid = [item for sublist in fgrid for item in sublist]
        grupo = [item for sublist in grupo for item in sublist]

        fgrid_aux = [item - delta for item in fgrid] + fgrid + [item + delta for item in fgrid]
        grupo = grupo + grupo + grupo

        fgrid_grupo = dict(zip(fgrid_aux, grupo))
        fgrid_grupo = {key: val for key, val in sorted(fgrid_grupo.items(), key=lambda ele: ele[0])}
        fgrid = list(fgrid_grupo.keys())
        grupo = list(fgrid_grupo.values())

        t_mat = list(itertools.repeat(t, len(fgrid)))
        t_mat = np.array(t_mat)
        t_mat_transpose = np.transpose(t_mat)
        t_mat_transpose_multiply = t_mat_transpose * 2 * math.pi
        t_mat_transpose_multiply = np.multiply(t_mat_transpose_multiply, fgrid)

        cosines = [[math.cos(item) for item in sublist] for sublist in t_mat_transpose_multiply]
        sines = [[math.sin(item) for item in sublist] for sublist in t_mat_transpose_multiply]
        x = np.concatenate((cosines, sines), axis=1)
        grupo = grupo + grupo

        return x, fgrid, grupo

    def fft(self, signal, t):
        """
        Function that calculates de FFT.

        params:
            - signal (np.array): input_signal signal to calculate its FFT
            - t (np.array): temporal axis of the signal

        returns:
            - frequencies (np.array): frencuencial axis of the FFT
            - clean_fft (np.array): FFT calculated with some artifacts eliminated
            - components_freqs (np.array): Exact frecuency of the resulting estimated components.

        """

        fft_signal = np.real(np.fft.fft(signal))
        fft_signal = fft_signal[0 : int(len(fft_signal) / 2)]
        dt = t[1] - t[0]
        frequencies = np.fft.fftfreq(len(signal), dt)
        frequencies = frequencies[0 : int(len(frequencies) / 2)]
        components, _ = sig.find_peaks(fft_signal, height=np.max(fft_signal) / 5)
        components_freqs = frequencies[components]

        return frequencies, fft_signal, components_freqs

    def complex_fft(self, signal, t):
        """
        Function that calculates de FFT.

        params:
            - signal (np.array): input_signal signal to calculate its FFT
            - t (np.array): temporal axis of the signal

        returns:
            - frequencies (np.array): frencuencial axis of the FFT
            - clean_fft (np.array): FFT calculated with some artifacts eliminated
            - components_freqs (np.array): Exact frecuency of the resulting estimated components.

        """

        fft_signal = np.fft.fft(signal)
        fft_signal = fft_signal[0 : int(len(fft_signal) / 2)]
        dt = t[1] - t[0]
        frequencies = np.fft.fftfreq(len(signal), dt)
        frequencies = frequencies[0 : int(len(frequencies) / 2)]
        components, _ = sig.find_peaks(fft_signal, height=np.max(fft_signal) / 5)
        components_freqs = frequencies[components]

        return frequencies, fft_signal, components_freqs

    #
    #    Improvements
    #
    def smooth_fft(self, signal, t):
        """
        Signal that calculates de fft and cleans the peaks of the fourier transform to make it more precise.

        params:
            - signal (np.array): Input signal to calculate its fft
            - t (np.array): temporal axis of the signal

        returns:
            - frequencies (np.array): frencuencial axis of the fft
            - clean_fft (np.array): fft calculated with some artifacts eliminated
            - components_freqs (np.array): Exact frecuency of the resulting estimated components.

        """
        fft_signal = np.real(np.fft.fft(signal))
        fft_signal = fft_signal[0 : int(len(fft_signal) / 2)]
        diferential_t = t[1] - t[0]
        frequencies = np.fft.fftfreq(len(signal), diferential_t)
        frequencies = frequencies[0 : int(len(frequencies) / 2)]
        components, _ = sig.find_peaks(fft_signal, height=np.max(fft_signal) / 5)
        components_freqs = frequencies[components]
        clean_fft = np.zeros(len(frequencies))
        for i in components:
            clean_fft[i] = fft_signal[i]

        return frequencies, clean_fft, components_freqs

    #   Test

    def test_signal_generator(self, ao, fo, nc, fs, dur, snr, fluc):
        """
        function that generates test heart signals (inputs) to test the code. To produce an AF make sure that
        you chose a fo/nc enought to reach more that 15hz components

        params:

            - ao (int): Amplitude of the components, this will be modified with a random number generator.
            - fo (int): Fundamental frecuency or first component frecuency
            - nc (int): Number of components, their frecuencies will be calculated using an harmonic series from
                  the fo. (2fo, 3fo, 4fo...)
            - fs (float): Sampling frecuency
            - dur (float): Signal duration (in seconds)
            - snr (float): Signal to noise ratio, This will be calculated as 10log(signal_power/Noise_power) and will not
                   have considerations about the bandwidth as it usually has.

            - fluc (int): It will add some unharmonic components, wich translates in no-linal distortion of the signal.
                    its distortion will be spaced in frencuency fluc*(1/fs).

        returns:

            - y (np array): Generated clean (temporal) signal
            - y_noisy (np array): Generated clean (temporal) signal with noise added
            - t (np array): temporal axis
            - freqs (np array): frecuency axis
            - amp (float): amplitude of the signal

        """
        A = ao * random.randint(1, nc)
        a_plus_delta_f = ao * random.randint(1, nc)
        a_minus_delta_f = ao * random.randint(1, nc)

        phi = 2 * np.pi * random.randint(1, nc)
        phi_mas_delta_f = 2 * np.pi * random.randint(1, nc)
        phi_menos_delta_f = 2 * np.pi * random.randint(1, nc)

        ts = 1 / fs  # segundos
        # np.linspace(start=0,stop=dur, num=int(dur * fs) ) #t = 0:ts:dur-ts; % Segundos
        t = np.arange(0, dur, ts)
        y = 0
        y_menos_delta_f = 0
        y_mas_delta_f = 0
        delta_f = 1 / fs
        freqs = fo * np.arange(1, nc + 1, 1)

        y_matrix = []
        y = []
        for i, _ in enumerate(t):
            row = []
            for j, _ in enumerate(freqs):
                row.append(A * np.cos(2 * np.pi * (freqs[j]) * t[i] + phi))

            y_matrix.append(row)
            y.append(np.sum(row))

        y = np.array(y)
        amp = A

        if fluc != 0:
            freqs_menos_delta_f = freqs - (delta_f * fluc)
            freqs_mas_delta_f = freqs + (delta_f * fluc)
            y_menos_delta_f = []
            y_mas_delta_f = []
            for i, _ in enumerate(t):
                row_menos = []
                row_mas = []
                for j in range(0, len(freqs)):
                    row_menos.append(
                        a_minus_delta_f * np.cos(2 * np.pi * (freqs_menos_delta_f[j]) * t[i] + phi_menos_delta_f)
                    )
                    row_mas.append(a_plus_delta_f * np.cos(2 * np.pi * (freqs_mas_delta_f[j]) * t[i] + phi_mas_delta_f))

                y_menos_delta_f.append(np.sum(row_menos))
                y_mas_delta_f.append(np.sum(row_mas))

            y = y + y_menos_delta_f + y_mas_delta_f
            # ANTIGUO -> freqs,idx = np.sort([freqs,freqs_menos_delta_f,freqs_mas_delta_f])
            freqs = np.concatenate((freqs, freqs_menos_delta_f, freqs_mas_delta_f))
            # idx = np.argsort(freqs)
            # freqs_sorted = freqs[idx]
            # amp = [A,a_minus_delta_f,a_plus_delta_f]
            # amp = amp[idx]

        noise_power = np.mean(y**2) / (10 ** (snr / 10))
        noise = np.array([random.normalvariate(0, np.sqrt(noise_power)) for _ in range(len(y))])
        # noise2 = (noise_power * np.random.normal(0,1,y.shape[0]))

        y_noisy = y + noise

        return y, y_noisy, t, freqs, amp
