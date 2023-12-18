"""
Author: SpikaTech
Date: 24/10/2023
Description:
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from azure_ml.inverse_problem.scripts.signals.signals_analyzer import SignalAnalyzer
from loguru import logger
from scipy.interpolate import CubicSpline
from scipy.signal import butter, detrend, filtfilt
from sklearn import linear_model
from vrcardio_vest.app.filters import apply_filters_per_lead


class ECGProcessor:
    def __init__(self, column_names: list, signals: pd.DataFrame, fs: int):
        self.column_names = column_names
        self.fs = fs
        signals.columns = column_names
        self.signals = signals
        self.tolerance = 10e-6
        self.win_len = 250
        self.static_threshold = 0.01
        self.dynamic_threshold = 6
        self.outliers_proportion = 0.25
        self.segments_proportion = 0.8
        self.wavelet_filter_size = 1000
        self.analyzer = SignalAnalyzer()
        self.bad_electrodes = []
        self.lower_iqr_threshold = 0.1
        self.upper_iqr_threshold = 0.9

        self.bad_electrodes = None
        self.lower = None
        self.upper = None

    def remove_outliers_and_interpolate(self, outliers_dict):
        """
        Remove outliers and interpolate the values in the ECG array.
        Args:
            outliers_dict (dict): Dictiopnary containing outliers as {'a1': list of indices, 'a2': list of indices...}
        """
        # logger.info("Replace outliers by cubic spline interpolation.")
        ecg_data_no_outliers = self.signals.copy()
        for channel in self.column_names:
            all_outliers = np.unique(outliers_dict.get(channel, [])).astype(int)
            if len(all_outliers) > 0:  # Only interpolate if outliers are present for the channel
                ecg_data_no_outliers[channel] = self.interpolate_outliers(channel, all_outliers)
        self.signals = ecg_data_no_outliers

    def interpolate_outliers(self, channel, outlier_indices, method="cubic"):
        """
        Interpolate outliers in a given signal using the specified method.

        Parameters:
        - channel: The channel that we are processing.
        - outlier_indices (list): List of indices at which outliers are present
        - method (str): Interpolation method ('linear', 'cubic', etc.)

        Returns:
        - pd.Series: Signal with outliers interpolated
        """
        # Remove outliers
        signal_no_outliers = self.signals[channel].drop(index=outlier_indices)

        # Interpolation function
        if method == "cubic":
            cs = CubicSpline(signal_no_outliers.index, signal_no_outliers.values)
            interpolated_values = cs(outlier_indices)
        else:
            raise ValueError("Unsupported interpolation method.")

        # Replace outliers with interpolated values
        signal_interpolated = self.signals[channel].copy()
        signal_interpolated.iloc[outlier_indices] = interpolated_values

        return signal_interpolated

    def detect_remaining_outliers_by_iqr_threshold(self, lower_percentile, upper_percentile, iqr_factor=1.5):
        outliers_dict = dict()
        for col in self.column_names:
            q1 = self.signals[col].quantile(lower_percentile)
            q3 = self.signals[col].quantile(upper_percentile)
            iqr = q3 - q1
            lower_bound = q1 - iqr_factor * iqr
            upper_bound = q3 + iqr_factor * iqr
            outliers = self.signals[col][(self.signals[col] < lower_bound) | (self.signals[col] > upper_bound)]
            outliers_dict[col] = outliers.index.tolist()
        return outliers_dict

    def filter_ecgs(self):
        # Step 1 Remove outliers ransac
        # logger.info("First outlier removal. Performing ransac outlier correction.")
        outliers_dict, bad_electrodes_1 = self.get_outliers_ransac()
        # Step 2 Replace outliers by interpolation
        self.remove_outliers_and_interpolate(outliers_dict)

        # Step 3 Correct baseline drift. Center the signals and correct baseline wander.
        # logger.info("Correct baseline drift. Center the signals and correct baseline wander.")
        self.remove_baseline_wander()

        # Step 4 Second outlier removal (0.1 and 0.9 percentiles IQR)
        """logger.info(
            f"Second outlier removal. Percentiles IQR: lower percentile{self.lower_iqr_threshold} "
            f"and upper percentile {self.upper_iqr_threshold}"
        )"""
        remaining_outliers = self.detect_remaining_outliers_by_iqr_threshold(
            self.lower_iqr_threshold, self.upper_iqr_threshold
        )
        self.remove_outliers_and_interpolate(remaining_outliers)

        # Step 5. Identify flat electrodes and electrodes full of noise. Identify interferences indexes.
        # Step 6. Find the longest clean range without interferences
        # Both steps are together in analyzer.analyze_interferences
        # logger.info("Starting analyze interferences.")
        (
            noisy_electrodes,
            flat_electrodes,
            interference_percentage,
            valid,
            largest_start,
            largest_end,
        ) = self.analyzer.analyze_interferences(self.signals)
        flat_electrodes_idx = [i for i, item in enumerate(self.column_names) if item in flat_electrodes]
        noisy_electrodes_idx = [i for i, item in enumerate(self.column_names) if item in noisy_electrodes.keys()]
        bad_electrodes = list(set(flat_electrodes_idx) | set(noisy_electrodes_idx))
        self.bad_electrodes = bad_electrodes
        self.lower = largest_start
        self.upper = largest_end
        # logger.info(f"Bad electrodes = {bad_electrodes}")
        # Handle exceptions
        if len(bad_electrodes) > 15:
            logger.warning("More than half of the electrodes are bad electrodes.")
            self.bad_electrodes = np.arange(0, 30).tolist()
            self.signals = pd.DataFrame([])
        elif not valid:
            logger.warning("There are not more that 5 continuous seconds without interferences.")
            self.bad_electrodes = np.arange(0, 101).tolist()
            self.signals = pd.DataFrame([])
        else:
            self.signals = self.signals[largest_start:largest_end]

    def get_outliers_ransac(self):

        self.signals.reset_index(inplace=True)
        X = self.signals.index.values.reshape(-1, 1)

        bad_electrodes = []
        outliers = []

        bad_segments = []

        for signal_id, signal_name in enumerate(self.column_names):
            signal = self.signals.loc[:, signal_name].values
            y = signal.reshape(-1, 1)

            # Robustly fit linear model with RANSAC algorithm
            ransac = linear_model.RANSACRegressor()
            ransac.fit(X, y)
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)
            error = np.sum((ransac.predict(X[inlier_mask]) - y[inlier_mask]) ** 2)
            local_outliers = X.reshape(-1)[outlier_mask]
            bad_ids = []

            if error < self.tolerance:
                bad_electrodes.append(signal_id)
            else:
                local_outliers = []
                signal_dt = detrend(signal, bp=range(0, len(signal), self.win_len))
                threshold = max(signal_dt.std() * self.dynamic_threshold, self.static_threshold)

                for i in range(self.win_len, len(signal), self.win_len):

                    index = X[(i - self.win_len) : i].reshape(-1)

                    ransac = linear_model.RANSACRegressor(residual_threshold=threshold)
                    ransac.fit(X[index], y[index])
                    inlier_mask = ransac.inlier_mask_
                    outlier_mask = np.logical_not(inlier_mask)
                    local_outliers.extend(index[outlier_mask])

                    prop_outliers = sum(outlier_mask) / self.win_len

                    if prop_outliers > self.outliers_proportion:
                        bad_ids.extend(list(range((i - self.win_len), i)))

            bad_segments.append(bad_ids)
            outliers.append(local_outliers)

            if len(bad_ids) / len(X) > self.segments_proportion:
                bad_electrodes.append([signal_id])

        outliers_dict = dict()
        for signal_name, outlier in zip(self.column_names, outliers):
            outliers_dict[signal_name] = outlier
        return outliers_dict, bad_electrodes

    def butter_highpass(self, cutoff, order=5):
        """
        Design a highpass Butterworth filter.
        Parameters:
        - cutoff (float): Cutoff frequency in Hz
        - order (int): Order of the filter
        Returns:
        - b, a: Numerator and denominator polynomials of the filter
        """
        nyquist = 0.5 * self.fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="high", analog=False)
        return b, a

    def highpass_filter(self, data, cutoff, order=5):
        """
        Apply a highpass Butterworth filter to the data.
        Parameters:
        - data (array-like): The original signal
        - cutoff (float): Cutoff frequency in Hz
        - order (int): Order of the filter
        Returns:
        - array-like: Filtered signal
        """
        b, a = self.butter_highpass(cutoff, order=order)
        y = filtfilt(b, a, data)
        return y

    def remove_baseline_wander(self):
        """
        Remove baseline wander from ECG signals.
        Parameters:
        - centered_signals (pd.DataFrame): Centered ECG signals.
        Returns:
        - pd.DataFrame: Corrected ECG signals.
        """
        # Substract the mean to center the signals in 0
        centered_signals = self.signals[self.column_names].apply(lambda x: x - np.mean(x), axis=0)
        # Cutoff frequency in Hz (commonly 0.5-1 Hz for ECG)
        cutoff = 0.5
        # Apply highpass filter to correct baseline wander
        corrected_signals = centered_signals.apply(lambda x: self.highpass_filter(x, cutoff), axis=0)
        self.signals = corrected_signals
