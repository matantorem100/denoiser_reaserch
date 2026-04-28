import numpy as np
import scipy.signal as sp


class FrequencyOffsetEstimator:
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate


    @staticmethod
    def calculate_normalized_correlation(signal: np.ndarray, uw: np.ndarray, is_bits = False) -> np.ndarray:
        if is_bits:
            signal = 2 * signal - 1
            uw = 2 * uw - 1

        corr = sp.convolve(signal, np.conj(uw[::-1]), mode="full")

        uw_energy = np.linalg.norm(uw, ord=2)
        sliding_energy = np.sqrt(sp.convolve(np.abs(signal) ** 2, np.ones(len(uw)), mode="full"))

        denom = uw_energy * sliding_energy
        denom = np.maximum(denom, 1e-12)

        norm_corr = corr / denom

        return norm_corr

    def coarse_estimate_hz(self, signal: np.ndarray, uw: np.ndarray) -> float:
        phase_diff = np.diff(np.unwrap(np.angle(signal)))

        mean_rad_per_sample = np.mean(phase_diff)

        coarse_hz = mean_rad_per_sample * self.sample_rate / (2 * np.pi)

        # grid_to_search = np.linspace(-50 + coarse_hz, 50 + coarse_hz, 100)
        #
        # best_correlation = -1
        # best_grid = grid_to_search[0]
        # for fine_offset in grid_to_search:
        #     corrected_signal = self.correct_frequency_offset(signal, fine_offset)
        #     normalize_correlation = self.calculate_normalized_correlation(signal, uw)
        #     if np.max(normalize_correlation) > best_correlation:
        #         best_correlation = np.max(normalize_correlation)
        #         best_grid = fine_offset

        return float(coarse_hz)

    def correct_frequency_offset(self, signal: np.ndarray, frequency_offset_hz: float) -> np.ndarray:
        n = np.arange(len(signal))
        correction = np.exp(-1j * 2 * np.pi * frequency_offset_hz * n / self.sample_rate)
        return signal * correction

    def estimate(self, signal: np.ndarray, uw: np.ndarray) -> float:
        coarse_hz = self.coarse_estimate_hz(signal, uw)
        return float(coarse_hz)