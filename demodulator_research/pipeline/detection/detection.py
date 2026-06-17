import numpy as np

from demodulator_research.find_grid.find_grid import find_best_uw_grid
from demodulator_research.frequency_offset_estimation.frequency_offset_estimation import FrequencyOffsetEstimator
from demodulator_research.uw_detection.uw_detection import UwDetector


def estimate_frequency_offset_from_detected_uws(signal: np.ndarray, uw_start_samples: list[int], sample_rate: float) -> dict:
    signal = np.asarray(signal, dtype=np.complex64).reshape(-1)
    estimator = FrequencyOffsetEstimator(method_type="differential_circular_coarse", reference_signal=signal[uw_start_samples[0]: ])
    estimated_frequency_offset = estimator.estimate(signal=signal[uw_start_samples[0]: ], sample_rate=sample_rate)
    return {"frequency_offset_hz": estimated_frequency_offset}


def fine_frequency_estimation(signal: np.ndarray, sample_rate: float, coarse_estimation: float, middle_cpfsk_uw: np.ndarray) -> dict:
    grid_of_frequency_estimations = np.arange(coarse_estimation - 100, coarse_estimation + 100, 10)
    complex_correlation_instance = UwDetector(method_type="complex_correlation")
    n = np.arange(len(signal))
    best_peak_list = []
    for frequency in grid_of_frequency_estimations:
        current_corrected_signal = signal * np.exp(-1j * 2 * np.pi * frequency * n / sample_rate)
        complex_correlation = complex_correlation_instance.estimate(current_corrected_signal, middle_cpfsk_uw)
        best_peak_list.append(np.max(complex_correlation))
    best_peak_array = np.array(best_peak_list)
    best_frequency_estimation = grid_of_frequency_estimations[np.argmax(best_peak_array)]
    return {"frequency_offset_hz": best_frequency_estimation}



def detect_middle_uw_grid(signal: np.ndarray, middle_cpfsk_uw: np.ndarray, threshold: float, sps: int,
                          uw_spacing_samples: int, n_uw: int | None, grid_tolerance_samples: int,
                          uw_n_symbols: int, n_symbols_each_side: int,
                          min_peak_distance_samples: int, min_n_uw: int,
                          first_peak_weight: float) -> dict:
    detector = UwDetector(method_type="differential_correlation", differential_lag_samples=int(sps))
    correlation = detector.estimate(signal=np.asarray(signal, dtype=np.complex64),
                                    cpfsk_uw=np.asarray(middle_cpfsk_uw, dtype=np.complex64))

    grid = find_best_uw_grid(correlation=np.asarray(correlation, dtype=float),
                             uw_spacing=int(uw_spacing_samples),
                             threshold=float(threshold),
                             n_uw=n_uw,
                             spacing_tolerance=int(grid_tolerance_samples),
                             min_peak_distance=int(min_peak_distance_samples),
                             min_n_uw=int(min_n_uw),
                             first_peak_weight=float(first_peak_weight))

    offset = (int(uw_n_symbols) - 2 * int(n_symbols_each_side)) * int(sps) # cause the middle uw and not the whole uw
    uw_start_samples = [int(peak - offset) for peak in grid.uw_positions]
    detected = len(uw_start_samples) >= int(min_n_uw)
    if n_uw is not None:
        detected = detected and len(uw_start_samples) == int(n_uw)

    return {
        "detected": bool(detected),
        "uw_start_samples": uw_start_samples,
        "peak_indices": list(grid.uw_positions),
        "peak_values": list(grid.peak_values),
        "grid_score": float(grid.score),
        "correlation": correlation,
    }
