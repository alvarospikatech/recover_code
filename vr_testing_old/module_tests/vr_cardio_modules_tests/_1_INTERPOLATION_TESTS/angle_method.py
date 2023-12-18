"""
Author: SpikaTech
Date: 24/10/2023
Description:
"""

import warnings

import numpy as np
import pandas as pd
from ecg_feature_extractor import ECGFeatureExtractor

# from vrcardio_vest.app.filters import apply_filters_per_lead, clean_ecg_from_df

warnings.filterwarnings("ignore")


def das_signal(path_leads_vest, filtering="wavelet"):
    """Calculate the cardiac axis and the 12-lead ECG from the vest 30-lead ECG signals

    Args:
        path_leads_vest (str): Path to the folder with the vest 30-lead ECG signals

    Returns:
        int: Cardiac axis
        pd.DataFrame: DataFrame with the cardiac axis and the 12-lead ECG signals
    """

    with open(path_leads_vest, "r") as f:
        data_from_vest = pd.read_csv(
            f.name,
            names=[
                "A1-FRV",
                "A2-FRV",
                "A3-FRV",
                "A4-FRA",
                "A5-RRV",
                "A6-RRV",
                "B1-FLV",
                "B2-FLV",
                "B3-FRA",
                "B4-FRA",
                "B5-FRA",
                "B6-FRA",
                "B7-FRV",
                "B8-FRV",
                "C1-LLV",
                "C2-LLV",
                "C3-LLV",
                "C4-LLA",
                "C5-BLA",
                "C6-BLA",
                "C7-BLA",
                "C8-BLV",
                "D1-BRV",
                "D2-BRV",
                "D3-BRA",
                "D4-BRA",
                "D5-BDK",
                "D6-BDK",
                "D7-BDK",
                "D8-BDK",
            ],
        )

    equiphasic_method = EquiphasicMethod()
    cardiac_axis, ecg_signal = equiphasic_method.calculate_angle_cardiac_axis(data_from_vest)
    names_header = ecg_signal.columns.tolist()

    return cardiac_axis, ecg_signal


class EquiphasicMethod:
    def __init__(self):
        self.sample_frequency = 100
        self.ecg_feature_extractor = ECGFeatureExtractor()

    def _build_ecg_signal_lead(
        self, data_from_vest: pd.DataFrame, lower: float = -1.5, upper: float = 1.5
    ) -> pd.DataFrame:
        """Create a dataframe of 12-lead ECG from a dataframe of VRCardio vest 30 ECG signals

        Some formulas from https://ecgwaves.com/topic/ekg-ecg-leads-electrodes-systems-limb-chest-precordial/

        I = LA - RA
        II = LL - RA
        III = LL - LA

        aVR = (RA + LA)/2
        aVL = (LL + LA)/2
        aVF = (LL + RA)/2

        Args:
            data_from_vest (pd.DataFrame): DataFrame with 30 ECG signals
            lower (float, optional): lower bound of the interval. Defaults to -1.5.
            upper (float, optional): upper bound of the interval. Defaults to 1.5.

        Returns:
            pd.DataFrame: DataFrame with 12-lead ECG signals
        """
        leads = self._get_electrode_twelve_leads()

        data = data_from_vest.copy()
        data["I"] = data_from_vest[leads["LA"]] - data_from_vest[leads["RA"]]
        data["II"] = data_from_vest[leads["LL"]] - data_from_vest[leads["RA"]]
        data["III"] = data_from_vest[leads["LL"]] - data_from_vest[leads["LA"]]
        data["aVR"] = (data_from_vest[leads["RA"]] - data_from_vest[leads["LA"]]) / 2
        data["aVL"] = (data_from_vest[leads["LL"]] - data_from_vest[leads["LA"]]) / 2
        data["aVF"] = (data_from_vest[leads["LL"]] - data_from_vest[leads["RA"]]) / 2
        data["V1"] = data_from_vest[leads["V1"]]
        data["V2"] = data_from_vest[leads["V2"]]
        data["V3"] = data_from_vest[leads["V3"]]
        data["V4"] = data_from_vest[leads["V4"]]
        data["V5"] = data_from_vest[leads["V5"]]
        data["V6"] = data_from_vest[leads["V6"]]

        data = data[["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]]
        if lower is not None and upper is not None:
            try:
                data = data.clip(lower=lower, upper=upper)
            except Exception:
                ValueError("Lower and upper must be numbers")
        else:
            pass
        return data

    @staticmethod
    def _get_electrode_twelve_leads():
        """Get the electrode dictionary for the twelve leads.

        Returns:
            Electrode dictionary.
        """
        return {
            "LA": "B4-FRA",
            "RA": "A4-FRA",
            "LL": "B8-FRV",
            "V1": "A3-FRV",
            "V2": "B6-FRA",
            "V3": "B7-FRV",
            "V4": "B2-FLV",
            "V5": "C4-LLA",
            "V6": "C7-BLA",
        }

    def _get_r_s_peaks(self, signal):
        qrs_complexes = self.ecg_feature_extractor.get_features(ecg_signal=signal)
        r_peaks_index = [qrs_complex[1] for qrs_complex in qrs_complexes]
        s_peaks_index = [qrs_complex[2] for qrs_complex in qrs_complexes]
        r_peaks = [signal[index] for index in r_peaks_index]
        s_peaks = [signal[index] for index in s_peaks_index]
        return r_peaks, s_peaks

    @staticmethod
    def _calculate_mean_qrs(r, s):
        if len(r) != 0 and len(s) != 0:
            amplitude_qrs_complexes = [abs(abs(r[i]) - abs(s[i])) for i in range(len(r))]
            mean_amplitude = np.mean(amplitude_qrs_complexes)
        else:
            mean_amplitude = 999
        return mean_amplitude

    @staticmethod
    def _calculate_polarity(r, s):
        if len(r) != 0 and len(s) != 0:
            mean_r_amplitude = abs(np.mean(r))
            mean_s_amplitude = abs(np.mean(s))
            if mean_r_amplitude >= mean_s_amplitude:
                polarity = 1
            else:
                polarity = -1
        else:
            polarity = 999
        return polarity

    @staticmethod
    def _get_equiphasic_lead(leads_info):
        amplitudes = [value[0] for value in leads_info.values()]
        min_mean_amplitude = np.min(amplitudes)
        equiphasic_lead = [key for key, value in leads_info.items() if value[0] == min_mean_amplitude][0]
        return equiphasic_lead

    @staticmethod
    def _from_lead_to_angle(lead):
        hexaxial_system_angles = {
            "+I": 0,
            "-I": -180,
            "+II": 60,
            "-II": -120,
            "+III": 120,
            "-III": -60,
            "+AVF": 90,
            "-AVF": -90,
            "+AVL": 150,
            "-AVL": -30,
            "+AVR": 30,
            "-AVR": -150,
        }
        cardiac_axis_angle = hexaxial_system_angles[lead]
        return cardiac_axis_angle

    @staticmethod
    def _create_hexaxial_system(leads):
        leads = list(leads)
        perpendicular_leads = leads[3:] + leads[:3]
        hexaxial_system_leads = dict(zip(leads, perpendicular_leads))
        return hexaxial_system_leads

    def _get_cardiac_axis(self, equiphasic_lead, leads_info):
        """
        Calculate the cardiac axis lead based on the hexaxial reference system.
        The cardiac axis lead in perpendicular to the most equiphasic lead.
        Args:
            leads (list): list with the names of each lead.
            equiphasic_lead (str): the name of the mos equiphasic lead.
            rs_mean_per_lead (dict): a dictionary with leads as keys and the mean of RS amplitudes as values
        """
        hexaxial_system_leads = self._create_hexaxial_system(leads=leads_info.keys())
        cardiac_axis_lead = hexaxial_system_leads[equiphasic_lead]
        polarity = leads_info[cardiac_axis_lead][1]
        if polarity > 0:
            cardiac_axis_lead = "+" + cardiac_axis_lead
        else:
            cardiac_axis_lead = "-" + cardiac_axis_lead
        cardiac_axis_angle = self._from_lead_to_angle(cardiac_axis_lead)
        return cardiac_axis_angle

    def calculate_angle_cardiac_axis(self, data_from_vest):
        """Calculate the cardiac axis angle and 12-lead ECG.

        Args:
            data_from_vest (pd.DataFrame): Dataframe with the data from the vest.

        Returns:
            float: cardiac axis angle.
            pd.DataFrame: 12-lead ECG.
        """
        ecg_signal = self._build_ecg_signal_lead(data_from_vest, lower=None, upper=None)

        return 0, ecg_signal
