import itertools

import numpy as np
import scipy.signal as sp

EPS = 1e-12


class UwDetector:
    def __init__(self, method_type: str, differential_lag_samples: int = 1,
                 context_references: np.ndarray | None = None,
                 uw_start_in_context_samples: int | None = None,
                 gaussian_sigma_samples: float | None = None):
        self.method_type = method_type
        self.differential_lag_samples = int(differential_lag_samples)
        self.context_references = context_references
        self.uw_start_in_context_samples = uw_start_in_context_samples
        self.gaussian_sigma_samples = gaussian_sigma_samples

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


    @staticmethod
    def gaussian_uw_window(reference_len: int, uw_start: int, uw_len: int,
                           gaussian_sigma_samples: float | None) -> np.ndarray:
        if gaussian_sigma_samples is None:
            gaussian_sigma_samples = max(1.0, uw_len / 2.0)

        sample_index = np.arange(reference_len)
        uw_center = uw_start + (uw_len - 1) / 2.0
        window = np.exp(-0.5 * ((sample_index - uw_center) / gaussian_sigma_samples) ** 2)
        window = window / np.max(window)

        return window


    @staticmethod
    def build_exhaustive_context_references(modulator, uw_bits: np.ndarray, symbol_time: float, sample_rate: float,
                                            frequency_sensitivity: float, n_bits_before_uw: int,
                                            n_bits_after_uw: int = 0, max_references: int = 65536) -> tuple[np.ndarray, int]:
        """
        Build every possible CPM UW waveform for all bit patterns around the UW.

        This is useful when the CPM pulse has memory, because the received UW
        waveform can depend on nearby unknown symbols.

        n_bits_before_uw:
            Number of unknown bits immediately before the UW to enumerate.

        n_bits_after_uw:
            Number of unknown bits immediately after the UW to enumerate.
            Use this when the pulse is longer than one symbol or centered/noncausal,
            because future symbols can affect the end/tail of the UW waveform.

        Returns:
            references, uw_start_in_context_samples
        """
        uw_bits = np.asarray(uw_bits, dtype=np.uint8).reshape(-1)
        bits_per_symbol = int(np.log2(modulator.config.constellation_config.constellation_order))

        n_bits_before_uw = int(n_bits_before_uw)
        n_bits_after_uw = int(n_bits_after_uw)
        if n_bits_before_uw < 0 or n_bits_after_uw < 0:
            raise ValueError("n_bits_before_uw and n_bits_after_uw must be non-negative")
        if n_bits_before_uw % bits_per_symbol != 0:
            raise ValueError("n_bits_before_uw must be divisible by bits_per_symbol")
        if n_bits_after_uw % bits_per_symbol != 0:
            raise ValueError("n_bits_after_uw must be divisible by bits_per_symbol")
        if len(uw_bits) % bits_per_symbol != 0:
            raise ValueError("uw_bits length must be divisible by bits_per_symbol")

        n_context_bits = n_bits_before_uw + n_bits_after_uw
        n_references = 2 ** n_context_bits
        if n_references > max_references:
            raise ValueError(
                f"Exhaustive context would build {n_references} references. "
                f"Increase max_references or reduce context bits."
            )

        references = []
        before_patterns = itertools.product((0, 1), repeat=n_bits_before_uw)
        after_patterns = list(itertools.product((0, 1), repeat=n_bits_after_uw))

        for before_bits_tuple in before_patterns:
            before_bits = np.asarray(before_bits_tuple, dtype=np.uint8)
            for after_bits_tuple in after_patterns:
                after_bits = np.asarray(after_bits_tuple, dtype=np.uint8)
                reference_bits = np.concatenate((before_bits, uw_bits, after_bits))
                reference_signal, _ = modulator.modulate(
                    bits=reference_bits,
                    symbol_time=symbol_time,
                    sample_rate=sample_rate,
                    uw=None,
                    frequency_offset=0.0,
                    frequency_sensitivity=frequency_sensitivity,
                )
                references.append(reference_signal)

        sps = int(round(sample_rate * symbol_time))
        uw_start_in_context_samples = int((n_bits_before_uw // bits_per_symbol) * sps)

        return np.asarray(references, dtype=np.complex128), uw_start_in_context_samples


    def context_exhaustive_correlation_matrix(self, signal: np.ndarray, cpfsk_uw: np.ndarray) -> np.ndarray:
        """
        Correlate against every context-padded UW reference.

        Returns:
            Matrix with shape (n_context_references, n_lags). Each row is the
            normalized correlation magnitude for one possible context around
            the UW. The caller can later reduce it with max/mean or inspect the
            best context per lag.
        """
        signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
        cpfsk_uw = np.asarray(cpfsk_uw, dtype=np.complex128).reshape(-1)

        if self.context_references is None:
            raise ValueError(
                "context_references are required. Build them with "
                "UwDetector.build_exhaustive_context_references(...)."
            )

        references = np.asarray(self.context_references, dtype=np.complex128)
        if references.ndim == 1:
            references = references.reshape(1, -1)
        if self.uw_start_in_context_samples is None:
            raise ValueError("uw_start_in_context_samples is required with context_references")
        uw_start = int(self.uw_start_in_context_samples)

        if references.size == 0:
            return np.empty((0, 0), dtype=np.float64)

        correlations = []
        for reference in references:
            reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
            if len(reference) == 0:
                continue

            window = self.gaussian_uw_window(
                reference_len=len(reference),
                uw_start=uw_start,
                uw_len=len(cpfsk_uw),
                gaussian_sigma_samples=self.gaussian_sigma_samples,
            )
            weighted_reference = reference * window

            correlation = np.abs(self.normalized_correlation(signal, weighted_reference))

            # Convert padded-reference peak location back to UW-start peak convention.
            if uw_start > 0:
                correlation = correlation[uw_start:]

            correlations.append(correlation.astype(np.float64))

        if not correlations:
            return np.empty((0, 0), dtype=np.float64)

        min_len = min(len(correlation) for correlation in correlations)
        return np.asarray([correlation[:min_len] for correlation in correlations], dtype=np.float64)


    def estimate(self, signal: np.ndarray, cpfsk_uw: np.ndarray) -> np.ndarray:

        if self.method_type == "complex_correlation":
            correlation = self.complex_correlation(signal=signal, cpfsk_uw=cpfsk_uw)

        elif self.method_type == "regular_correlation":
            correlation = self.regular_correlation(signal=signal, cpfsk_uw=cpfsk_uw)

        elif self.method_type == "differential_correlation":
            correlation = self.differential_correlation(signal=signal, cpfsk_uw=cpfsk_uw)

        elif self.method_type == "context_exhaustive_correlation_matrix":
            correlation = self.context_exhaustive_correlation_matrix(signal=signal, cpfsk_uw=cpfsk_uw)

        else:
            raise NotImplementedError

        return correlation
