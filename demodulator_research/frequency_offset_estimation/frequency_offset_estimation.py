from typing import Union

import numpy as np
import scipy.signal as sp

EPS = 1e-12


class FrequencyOffsetEstimator:
    def __init__(self, method_type: str, reference_signal: Union[np.ndarray, None]):
        self.method_type = method_type
        self.reference_signal = reference_signal


    @staticmethod
    def normalized_correlation(signal: np.ndarray, uw: np.ndarray) -> np.ndarray:
        non_normalized_correlation = sp.convolve(signal, np.conj(uw[::-1]))

        sliding_signal_energy = np.sqrt(sp.convolve(np.abs(signal) ** 2, np.ones_like(uw)))
        uw_energy = np.linalg.norm(uw, ord=2)

        normalize_correlation = non_normalized_correlation / (sliding_signal_energy * uw_energy)

        return normalize_correlation


    @staticmethod
    def phase_diff_coarse_estimate(signal: np.ndarray, sample_rate: float) -> float:
        """
        This function estimates the frequency offset of cpm signals. The algorithm:
        cpm signal: e^(j * 2 * pi * h * integral(s(t)) + j * 2 * pi * f_d * t)
        taking phase giving 2 * pi * h * integral(s(t)) + 2 * pi * f_d * t
        diff giving 2 * pi * h * s(t) + 2 * pi * f_d
        dividing in 2*pi: h * s(t) + f_d
        and the method assume that the given part of the signal is zero mean (which is under some cases correct)
        so taking the mean giving f_d
        :param signal: the cpm signal
        :param sample_rate: the sample rate of the signal
        :return: the frequency offset
        """
        phase_diff = np.diff(np.unwrap(np.angle(signal)))

        mean_rad_per_sample = np.mean(phase_diff)

        coarse_hz = mean_rad_per_sample * sample_rate / (2 * np.pi)

        return float(coarse_hz)


    @staticmethod
    def differential_circular_coarse_estimation(signal: np.ndarray, sample_rate: float) -> float:
        """
        This function estimates the frequency offset of cpm signals. The algorithm:
        cpm signal: e^(j phi[n] + j * 2 * pi * f_d * t)
        multiplying rx[n] * conj(rx[n-1]) gives e^(j phi[n]) * e^(j phi[n-1]) * e^(j * 2 * pi * f_d)
        then taking the mean and the angle gives the frequency offset
        :param signal: the cpm signal
        :param sample_rate: the sample rate of the signal
        :return: the frequency offset
        """
        signal = np.asarray(signal, dtype=np.complex64).reshape(-1)
        signal = signal / np.maximum(np.abs(signal), EPS)

        differential_rx = signal[1:] * np.conj(signal[:-1])

        mean_differential_phasor = np.mean(differential_rx)

        phase_increment_rad = np.angle(mean_differential_phasor)

        estimated_offset_hz = phase_increment_rad * sample_rate / (2.0 * np.pi)

        return float(estimated_offset_hz)



    def differential_second_method_for_coarse_estimation(self, signal: np.ndarray, reference_signal: np.ndarray, sample_rate: float) -> float:
        signal = np.asarray(signal, dtype=np.complex64).reshape(-1)
        reference_signal = np.asarray(reference_signal, dtype=np.complex64).reshape(-1)

        n_samples = min(len(signal), len(reference_signal))
        if n_samples < 2:
            return 0.0

        signal = signal[:n_samples]
        reference_signal = reference_signal[:n_samples]

        signal = signal / np.maximum(np.abs(signal), EPS)
        reference_signal = reference_signal / np.maximum(np.abs(reference_signal), EPS)

        differential_rx = signal[1:] * np.conj(signal[:-1])
        differential_reference = reference_signal[1:] * np.conj(reference_signal[:-1])

        correlation = self.normalized_correlation(differential_rx, differential_reference)

        peak_index = int(np.argmax(np.abs(correlation)))
        phase_increment_rad = np.angle(correlation[peak_index])

        estimated_offset_hz = phase_increment_rad * sample_rate / (2.0 * np.pi)

        return float(estimated_offset_hz)


    @staticmethod
    def differential_data_aided_estimation(signal: np.ndarray, reference_signal: np.ndarray, sample_rate: float) -> float:
        """
        Data-aided differential frequency-offset estimator.

        The reference_signal should be the known transmitted UW waveform without
        frequency offset, generated with the same modulation, pulse shape, sample
        rate, symbol time, and modulation index as the received signal.

        The method compares differential phasors:

            differential_rx[n]  = rx[n] * conj(rx[n - 1])
            differential_ref[n] = ref[n] * conj(ref[n - 1])

        Their product removes the known CPM/UW phase evolution:

            differential_rx[n] * conj(differential_ref[n])
                ~= exp(j * 2*pi*frequency_offset / sample_rate)

        The angle of the circular mean is the estimated phase increment per sample.
        """
        signal = np.asarray(signal, dtype=np.complex64).reshape(-1)
        reference_signal = np.asarray(reference_signal, dtype=np.complex64).reshape(-1)

        n_samples = min(len(signal), len(reference_signal))
        if n_samples < 2:
            return 0.0

        signal = signal[:n_samples]
        reference_signal = reference_signal[:n_samples]

        signal = signal / np.maximum(np.abs(signal), EPS)
        reference_signal = reference_signal / np.maximum(np.abs(reference_signal), EPS)

        differential_rx = signal[1:] * np.conj(signal[:-1])
        differential_reference = reference_signal[1:] * np.conj(reference_signal[:-1])

        data_aided_differential = differential_rx * np.conj(differential_reference)

        mean_data_aided_phasor = np.mean(data_aided_differential)

        phase_increment_rad = np.angle(mean_data_aided_phasor)

        estimated_offset_hz = phase_increment_rad * sample_rate / (2.0 * np.pi)

        return float(estimated_offset_hz)


    @staticmethod
    def correct_frequency_offset(signal: np.ndarray, frequency_offset_hz: float, sample_rate: float) -> np.ndarray:
        n = np.arange(len(signal))
        correction = np.exp(-1j * 2 * np.pi * frequency_offset_hz * n / sample_rate)
        return signal * correction


    def estimate(self, signal: np.ndarray, sample_rate: float, reference_signal: Union[np.ndarray, None] = None) -> float:
        if self.method_type == "phase_diff_coarse":
            frequency_offset_estimation = self.phase_diff_coarse_estimate(signal, sample_rate)
        elif self.method_type == "differential_circular_coarse":
            frequency_offset_estimation = self.differential_circular_coarse_estimation(signal, sample_rate)
        elif self.method_type == "differential_data_aided":
            if reference_signal is None:
                reference_signal = self.reference_signal
            if reference_signal is None:
                raise ValueError("reference_signal is required for differential_data_aided estimation")
            frequency_offset_estimation = self.differential_data_aided_estimation(signal, reference_signal, sample_rate)
        elif self.method_type == "differential_second_method":
            if reference_signal is None:
                reference_signal = self.reference_signal
            if reference_signal is None:
                raise ValueError("reference_signal is required for differential_second_method estimation")
            frequency_offset_estimation = self.differential_second_method_for_coarse_estimation(
                signal=signal,
                reference_signal=reference_signal,
                sample_rate=sample_rate,
            )
        else:
            raise NotImplementedError
        return frequency_offset_estimation
