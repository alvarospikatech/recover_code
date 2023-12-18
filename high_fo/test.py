import maf
import numpy as np
import pandas as pd

MAF_obj = maf.Maf()
freqModel = [5,10,15,20,25,30,35,40,45,50,55,60]

#fo=5 ,Nc= 4 --> SNR=10

array_fluc = np.arange(0,100)
array_SNR = np.flip(np.arange(-7,-5))

result = []
for i in array_SNR:
    print(str(i) + ". _______________________________________")
    error, prob = MAF_obj.test( Ao=1,fo=5,Nc=4,Fs=100,dur=3,SNR=i,fluc=0,f=freqModel)
    result.append([i,error,prob])


df = pd.DataFrame(result, columns=['SNR', "Error Adaptative Clean", "prob Adaptative clean"])

nombre_archivo_excel = 'test.xlsx'
df.to_excel(nombre_archivo_excel, index=False, engine='openpyxl')