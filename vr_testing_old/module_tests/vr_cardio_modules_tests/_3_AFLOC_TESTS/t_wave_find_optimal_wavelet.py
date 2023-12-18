import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pywt
import sys

sys.path.append("src/VRCARDIO-Testing/vr_cardio_modules_tests")
import common_tools
from QRS_detector import QRS_detector


def find_optimal_onesig(t,clean_sig, idx_comparing1 , idx_comparing2 , wavelet):

    #clean_sig = passband_filter(clean_sig, 0.3, 20, fs)

    QRS = QRS_detector(clean_sig, t, 360)
    window = 1 - 0.985 * QRS
    clean_sig = clean_sig * window

    reference_signal_comparing1 = np.zeros(clean_sig.shape)
    reference_signal_comparing1[idx_comparing1] = 1

    reference_signal_comparing2 = np.zeros(clean_sig.shape)
    reference_signal_comparing2[idx_comparing2] = 1


    window = 20
    reference_signal_comparing1 = np.convolve(reference_signal_comparing1, np.ones(window) / window, mode="same")
    reference_signal_comparing1 = reference_signal_comparing1 / np.max(np.abs(reference_signal_comparing1))
    reference_signal_comparing2 = np.convolve(reference_signal_comparing2, np.ones(window) / window, mode="same")
    reference_signal_comparing2 = reference_signal_comparing2 / np.max(np.abs(reference_signal_comparing2))

    idx_comparing1 = np.where(reference_signal_comparing1 != 0)[0]
    idx_comparing2 = np.where(reference_signal_comparing2 != 0)[0]

    level = 3#4  # Número de niveles de descomposición
    coeffs = pywt.swt(clean_sig, wavelet, level=level,trim_approx=True)
    # Recuperar las señales aproximadas y detalladas
    #approximations = [a for a, _ in coeffs]
    #details = [d for _, d in coeffs]

    
    coeffs = np.array(coeffs)
    #approximations = np.array(approximations)
    #details = np.array(details)

    clean_sig = np.zeros(clean_sig.shape)
    clean_sig[idx_comparing1] = 3

    #print(coeffs.shape)
    coefs_comparing1 = np.max(coeffs[0,idx_comparing1])
    coefs_comparing2= np.max(coeffs[0,idx_comparing2])

    difference = np.abs(coefs_comparing1 - coefs_comparing2)

    return difference
    





def run_one_case():


    """ wavelet_list = []
    wavelet_fam_list = pywt.families()
    for i in wavelet_fam_list:
        wavelet_list += pywt.wavelist(i , kind='discrete')
    """

    wavelet_list = pywt.wavelist(kind='discrete')
    print(wavelet_list)


    session = common_tools.get_victoria_subset(1, 0, af=False)[0]
    case = common_tools.load_case(session)



    
    results = []
    for j in wavelet_list:
        difference_list = []
        for i in case[ "torso_potentials"]:
            difference_list.append(find_optimal_onesig(case["t_axis"],i, 390 , 500 , wavelet = j))

        difference_list = np.array(difference_list)
        print("- Wavelet: "  +j + " Value: "  + str(np.mean(difference_list)))
        results.append(np.mean(difference_list))


    print(" ")
    print("BEST RESULT:")
    idx = np.argmax(np.array(results))
    print("Wavelet: "  +wavelet_list[idx] + " Value: "  + str(results[idx]))

run_one_case()