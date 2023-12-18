import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def white_noise_generator(fs):

    time_dur = 5
    t = np.linspace(0, time_dur, int(time_dur * fs), endpoint=False)
    desired_power_dbw = 10
    power_watts = 10**(desired_power_dbw / 10)
    samples = np.random.normal(0, 1, len(t))
    scaling_factor = np.sqrt(power_watts / np.mean(samples**2))
    noise_signal = samples * scaling_factor

    return noise_signal,t


def generate_cardiac_impulse(frec,fs):
    # Parámetros de la señal 1
    amp = 1.0
    duracion = 3#np.round(1/frec , 3)  # segundos

    t = np.linspace(0, duracion + (1/fs), int(duracion * fs), endpoint=False)

    señal1 = amp * np.cos(2 * np.pi * frec * t) + amp/2 * np.cos(2* 2 * np.pi * frec * t) + amp/3 * np.cos(3* 2 * np.pi * frec * t) + amp/4 * np.cos(4* 2 * np.pi * frec * t)
    señal2 = amp * np.cos(2 * np.pi * frec * t)

    temp_amplitude_red =  np.exp(-2* t)

    #señal1 = señal1 * temp_amplitude_red

    return señal2,t


def generate_cardiac_train(lpm,fs):
    
    time_dur = 5
    t = np.linspace(0, time_dur - (1/fs), int(time_dur * fs), endpoint=False)
    signal = np.zeros(t.shape)
    fo = np.round(lpm/60 , 3)
    to = np.round(1/fo ,3)

    print("selected fo: " + str(fo))
    before_pos = 0
    pulses = np.arange(0,time_dur,to)[1:]
    for i in pulses:
        index = np.argmin(np.abs(t - i))
        add_impulse = generate_cardiac_impulse(fo,fs)[0]
        
        if signal[before_pos: index].shape  == add_impulse.shape:
            signal[before_pos: index] = add_impulse
        else:
            signal[before_pos: index-1] = add_impulse
        before_pos = index

    return signal,t


def densidad_de_potencia(s ,fs):
        
    fft = np.fft.fft(s) / len(s)
    phase = np.angle(fft)
    s = 2* np.correlate(s, s, mode='full') / len(s)
    psd = np.fft.fft(s) / len(s)
    psd_magnitude = np.real(np.abs(psd.copy()))
    freqs = np.fft.fftfreq(len(psd), 1 / fs)

    return freqs,psd_magnitude,phase

frecuencia_muestreo = 250.0  # Hz
señal1,t = generate_cardiac_impulse(4,frecuencia_muestreo) #white_noise_generator(frecuencia_muestreo) #

frecuencias,psd_mag,psd_phase =  densidad_de_potencia(señal1 ,frecuencia_muestreo)


#frecuencias = sorted(frecuencias)
idx_0 = int(len(frecuencias)/2)
frecuencias = frecuencias[0:idx_0]
psd_mag = psd_mag[0:idx_0] * 2
psd_phase = psd_phase[0:idx_0]
# Graficar las señales en el dominio del tiempo
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, señal1, label='Señal 1')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()

# Graficar las transformadas en el dominio de la frecuencia
plt.subplot(3, 1, 2)
plt.plot(frecuencias,psd_mag, label='Magnitud')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.xlim(0, 15)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(frecuencias, psd_phase, label='Fase')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('fase')
plt.xlim(0, 15)
plt.legend()

plt.tight_layout()
plt.show()