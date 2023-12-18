import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

ekg_conversion_dict = {
    'EKG_1': 'A1',
    'EKG_2': 'A2',
    'EKG_3': 'A3',
    'EKG_4': 'A4',
    'EKG_5': 'A5',
    'EKG_6': 'A6',
    'EKG_7': 'B1',
    'EKG_8': 'B2',
    'EKG_9': 'B3',
    'EKG_10': 'B4',
    'EKG_11': 'B5',
    'EKG_12': 'B6',
    'EKG_13': 'B7',
    'EKG_14': 'B8',
    'EKG_15': 'C1',
    'EKG_16': 'C2',
    'EKG_17': 'C3',
    'EKG_18': 'C4',
    'EKG_19': 'C5',
    'EKG_20': 'C6',
    'EKG_21': 'C7',
    'EKG_22': 'C8',
    'EKG_23': 'D1',
    'EKG_24': 'D2',
    'EKG_25': 'D3',
    'EKG_26': 'D4',
    'EKG_27': 'D5',
    'EKG_28': 'D6',
    'EKG_29': 'D7',
    'EKG_30': 'D8'}



df = pd.read_csv(r"/home/robotate/VRCARDIO-DAS/ouputs/211123/csv/DCN/data_DCN_211123_normal_volts.csv") # cargamos tabla
df.rename(columns=ekg_conversion_dict, inplace=True)
#df = df.loc[~(df==0).all(axis=1)] # quitamos cero
SAMPLING_RATE=500 # declaramos frecuencia de muestreo
#df = clean_ecg_from_df(df, sampling_rate=SAMPLING_RATE) # limpiamos la señal
#df = df[20000:22000]

# Determine the number of rows and columns for the subplot grid
n_rows = 12
n_cols = 3

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Loop through the columns and plot each signal
for i, column in enumerate(df.columns):
    axes[i].plot(df[column])
    axes[i].set_title(f'Signal {column}')
    axes[i].set_xlabel('Time (ms)')
    axes[i].set_ylabel('Amplitude')

# Remove any remaining empty subplots
for i in range(len(df.columns), n_rows * n_cols):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()