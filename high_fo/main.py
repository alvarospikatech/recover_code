import MAF
import numpy as np
import representations

MAF_obj = MAF.Maf()
#MAF_obj.freqModel = [5, 10 ,15,20]

#fo=5 ,Nc= 4 --> SNR=10
t_axis,y, y_ruidosa, y_est, f_axis, y_ruidosa_FFT , y1_FFT = MAF_obj.monosignal_example( ao=1,fo=5,nc=4,fs=100,dur=3,snr=0,fluc=0 ,clean=True)

y = y / np.max(y)
y_ruidosa = y_ruidosa/ np.max(y_ruidosa)
y_est = y_est / np.max(y_est)

y_ruidosa_FFT = y_ruidosa_FFT / np.max(y_ruidosa_FFT)
y1_FFT = y1_FFT / np.max(y1_FFT)


representations.show_nicole_plot(t_axis,y, y_ruidosa, y_est, f_axis, y_ruidosa_FFT, y1_FFT)