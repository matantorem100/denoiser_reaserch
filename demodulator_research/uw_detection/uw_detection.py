import numpy as np
import scipy.signal as sp

EPS = 1e-12


class UwDetector:
    def __init__(self, method_type: str):
        self.method_type = method_type

    @staticmethod
    def normalized_correlation(signal: np.ndarray, uw: np.ndarray) -> np.ndarray:
        non_normalized_correlation = sp.convolve(signal, np.conj(uw[::-1]))

        sliding_signal_energy = np.sqrt(sp.convolve(np.abs(signal) ** 2, np.ones_like(uw)))
        uw_energy = np.linalg.norm(uw, ord=2)

        normalize_correlation = non_normalized_correlation / (sliding_signal_energy * uw_energy)

        return normalize_correlation


    def complex_correlation(self, signal: np.ndarray, cpfsk_uw: np.ndarray) -> np.ndarray:
        """
        This function computes the complex correlation between the modulated cpfsk uw and modulated cpfsk signal
        :param signal: the modulated cpfsk signal
        :param cpfsk_uw: the modulated cpfsk uw
        :return: the complex correlation
        """
        correlation = np.abs(self.normalized_correlation(signal, cpfsk_uw))

        return correlation


    def regular_correlation(self, signal: np.ndarray, cpfsk_uw: np.ndarray) -> np.ndarray:
        """
        This function computes the correlation between the diff(unwrap(angle(sig))) to the  uw
        :param signal: the cpfsk signal
        :param cpfsk_uw: the modulated cpfsk uw
        :return: the correlation
        """
        uw = np.diff(np.unwrap(np.angle(cpfsk_uw)))
        signal = np.diff(np.unwrap(np.angle(signal)))

        correlation = np.abs(self.normalized_correlation(signal, uw))

        return correlation


    def differential_correlation(self, signal: np.ndarray, cpfsk_uw: np.ndarray) -> np.ndarray:
        """

        :param signal:
        :param cpfsk_uw:
        :return:
        """
        signal = signal / np.maximum(np.abs(signal), EPS)
        differential_signal = signal[10:] * np.conj(signal[:-10])

        cpfsk_uw = cpfsk_uw / np.maximum(np.abs(cpfsk_uw), EPS)
        differential_uw = cpfsk_uw[10:] * np.conj(cpfsk_uw[:-10])

        correlation = np.abs(self.normalized_correlation(differential_signal, differential_uw))

        return correlation


    def estimate(self, signal: np.ndarray, cpfsk_uw: np.ndarray) -> np.ndarray:

        if self.method_type == "complex_correlation":
            correlation = self.complex_correlation(signal=signal, cpfsk_uw=cpfsk_uw)

        elif self.method_type == "regular_correlation":
            correlation = self.regular_correlation(signal=signal, cpfsk_uw=cpfsk_uw)

        elif self.method_type == "differential_correlation":
            correlation = self.differential_correlation(signal=signal, cpfsk_uw=cpfsk_uw)

        else:
            raise NotImplementedError

        return correlation
