"""
Author: SpikaTech
Date: 24/10/2023
Description:
"""

import neurokit2 as nk
import numpy as np
import pandas as pd
from loguru import logger


class ECGFeatureExtractor:
    def __init__(self):
        self.fs = 250
        pass

    def _peaks_detector(self, ecg_signal: np.ndarray, r_peaks: dict) -> pd.DataFrame:
        """
        Function to delineate the QRS complex.
        Args:
            ecg_signal (np.ndarray): array with the amplitude values from ecg signal.
            r_peaks (dict): dictionary with "ECG_R_Peaks" as key and r peaks indexes as value.

        Returns:
            pd.DataFrame: dataframe in which occurrences o peaks, onsets and offsets marked as "1" in a list of zeros.
        """
        ecg_signal = pd.Series(ecg_signal)
        ecg_signal = ecg_signal.append(pd.Series([ecg_signal.iloc[-1]]))
        signals, waves = nk.ecg_delineate(ecg_cleaned=ecg_signal, rpeaks=r_peaks, sampling_rate=self.fs, method="peak")
        signals = signals.reset_index()
        return signals

    def _r_peaks_detector(self, ecg_signal: np.ndarray) -> dict:
        """
        Detects R peaks in an ecg signal using neurokit package.
        Args:
            ecg_signal (np.ndarray): array with the amplitude values from ecg signal.

        Returns:
            dict: dict with the index position for each r peak detected.
        """
        r_peaks_dict = nk.ecg.ecg_findpeaks(ecg_signal, sampling_rate=self.fs)
        return r_peaks_dict

    @staticmethod
    def _s_peaks_detector(qrs_signals: pd.DataFrame) -> list:
        """
        Detects s peaks in an ecg signal using neurokit package.
        Args:
            qrs_signals (pd.DataFrame): dataframe with peaks as columns. Occurrences marked as 1.

        Returns:
            list: list with the index position for each s peak detected.
        """
        s_peaks_index = qrs_signals[qrs_signals["ECG_S_Peaks"] == 1]["index"].to_list()
        return s_peaks_index

    @staticmethod
    def _q_peaks_detector(qrs_signals: pd.DataFrame) -> list:
        """
        Detects q peaks in an ecg signal using neurokit package.
        Args:
            qrs_signals: dataframe with peaks as columns. Occurrences marked as 1.

        Returns:
            list: list with the index position for each q peak detected.
        """
        q_peaks_index = qrs_signals[qrs_signals["ECG_Q_Peaks"] == 1]["index"].to_list()
        return q_peaks_index

    @staticmethod
    def _get_qrs_complexes(q_peaks_index: list, r_peaks_index: np.ndarray, s_peaks_index: list):
        """
        Function to link each Q peak with its R and S peak to create QRS complexes.
        We use R peaks as a quality control: if a Q and S peak are part of the same QRS complex,
        there can only be one R peak between them.
        Args:
            q_peaks_index (list): list with the index position for each q peak detected.
            r_peaks_index (np.ndarray): list with the index position for each s peak detected.
            s_peaks_index (list): list with the index position for each r peak detected.

        Returns:
            list: list with QRS complexes
        """
        qrs_complexes = []
        for i in range(len(q_peaks_index)):
            q_peak = q_peaks_index[i]
            s_peaks_possibles = [s_peak for s_peak in s_peaks_index if s_peak > q_peak]
            for s_peak in s_peaks_possibles:
                qs_interval = (q_peak, s_peak)
                r_peaks_in_interval = [r_peak for r_peak in r_peaks_index if qs_interval[0] < r_peak < qs_interval[1]]
                if len(r_peaks_in_interval) == 1:
                    r_peak = r_peaks_in_interval[0]
                    qrs_complex = [q_peak, r_peak, s_peak]
                    qrs_complexes.append(qrs_complex)
                    break
        return qrs_complexes

    def get_features(self, ecg_signal):
        error = False
        r_peaks_dict = self._r_peaks_detector(ecg_signal)
        if len(r_peaks_dict["ECG_R_Peaks"]) == 0:
            logger.warning(f"No R Peaks found in ecg signal.")
            error = True
        else:
            signals_qrs = self._peaks_detector(ecg_signal, r_peaks_dict)
            q_peaks_index = self._q_peaks_detector(signals_qrs)
            s_peaks_index = self._s_peaks_detector(signals_qrs)

            if len(s_peaks_index) == 0:
                logger.warning(f"Unable to find S Peaks in ecg signal.")
                error = True
            elif len(q_peaks_index) == 0:
                logger.warning(f"Unable to find Q Peaks in ecg signal.")
                error = True
            else:
                qrs_complexes = self._get_qrs_complexes(q_peaks_index, r_peaks_dict["ECG_R_Peaks"], s_peaks_index)
                if len(qrs_complexes) == 0:
                    logger.warning(
                        f"No peak position Q is greater than no peak position S  with a R peak"
                        f" between themin ecg signal."
                    )
                    error = True
        if error:
            qrs_complexes = []
        return qrs_complexes
