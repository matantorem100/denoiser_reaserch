import numpy as np
import scipy.signal as sp

EPS = 1e-12


class UwDetector:
    def __init__(self, method_type: str, differential_lag_samples: int = 1):
        self.method_type = method_type
        self.differential_lag_samples = int(differential_lag_samples)

    @staticmethod
    def normalized_correlation(signal: np.ndarray, uw: np.ndarray) -> np.ndarray:
        non_normalized_correlation = sp.convolve(signal, np.conj(uw[::-1]))

        sliding_signal_energy = np.sqrt(sp.convolve(np.abs(signal) ** 2, np.ones_like(uw)))
        uw_energy = np.linalg.norm(uw, ord=2)

        normalize_correlation = non_normalized_correlation / (sliding_signal_energy * uw_energy)

        return normalize_correlation


    def complex_correlation(self, signal: np.ndarray, cpfsk_uw: np.ndarray) -> np.ndarray:
        correlation = np.abs(self.normalized_correlation(signal, cpfsk_uw))

        return correlation


    def regular_correlation(self, signal: np.ndarray, cpfsk_uw: np.ndarray) -> np.ndarray:
        uw = np.diff(np.unwrap(np.angle(cpfsk_uw)))
        signal = np.diff(np.unwrap(np.angle(signal)))

        correlation = np.abs(self.normalized_correlation(signal, uw))

        return correlation


    def differential_correlation(self, signal: np.ndarray, cpfsk_uw: np.ndarray) -> np.ndarray:
        lag = self.differential_lag_samples

        signal = signal / np.maximum(np.abs(signal), EPS)
        differential_signal = signal[lag:] * np.conj(signal[:-lag])

        cpfsk_uw = cpfsk_uw / np.maximum(np.abs(cpfsk_uw), EPS)
        differential_uw = (cpfsk_uw[lag:]) * np.conj(cpfsk_uw[:-lag])

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
