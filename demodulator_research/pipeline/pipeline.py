
from dataclasses import dataclass
from functools import lru_cache
import math
import time

from numba import njit
import numpy as np

from demodulator_research.find_grid.find_grid import find_best_uw_grid
from demodulator_research.frequency_offset_estimation.frequency_offset_estimation import FrequencyOffsetEstimator
from demodulator_research.pipeline.dataset.dataset_for_pipeline import build_pipeline_test_dataset
from demodulator_research.uw_detection.uw_detection import UwDetector
from synthetic_dataset_creation.constellation.constellation import Constellation, ConstellationConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShape, PulseShapeConfig


TWO_PI = 2.0 * np.pi


@lru_cache(maxsize=32)
def _cached_pulse(pulse_shape_type: str, pulse_normalization: str, span_in_symbols: int,
                  pulse_causal: bool, sample_rate: float, symbol_time: float) -> np.ndarray:
    pulse_config = PulseShapeConfig(
        pulse_shape_type=pulse_shape_type,
        normalization_type=pulse_normalization,
        span_in_symbols=int(span_in_symbols),
        pulse_causal=bool(pulse_causal),
    )
    pulse = np.asarray(PulseShape(pulse_config).generate_pulse_shape(float(sample_rate), float(symbol_time)),
                       dtype=np.float32).reshape(-1)
    pulse.setflags(write=False)
    return pulse


@lru_cache(maxsize=16)
def _cached_constellation_points(constellation_type: str, constellation_order: int) -> np.ndarray:
    constellation_config = ConstellationConfig(
        constellation_type=constellation_type,
        constellation_order=int(constellation_order),
    )
    constellation_points = np.asarray(Constellation(constellation_config).generate_constellation_points(),
                                      dtype=np.float32).reshape(-1)
    constellation_points.setflags(write=False)
    return constellation_points


@lru_cache(maxsize=16)
def _cached_constellation_bits(constellation_order: int) -> np.ndarray:
    bits_per_symbol = int(math.log2(int(constellation_order)))
    bit_rows = []
    for symbol_index in range(int(constellation_order)):
        gray_index = symbol_index ^ (symbol_index >> 1)
        bit_rows.append([(gray_index >> bit_index) & 1 for bit_index in reversed(range(bits_per_symbol))])
    bits = np.asarray(bit_rows, dtype=np.uint8)
    bits.setflags(write=False)
    return bits


def _config_pulse(config: "CpfskViterbiConfig") -> np.ndarray:
    return _cached_pulse(
        config.pulse_shape_type,
        config.pulse_normalization,
        int(config.span_in_symbols),
        bool(config.pulse_causal),
        float(config.sample_rate),
        float(config.symbol_time),
    )


def _config_constellation_points(config: "CpfskViterbiConfig") -> np.ndarray:
    return _cached_constellation_points(config.constellation_type, int(config.constellation_order))


##########################################################################################################
# Dataset creation
##########################################################################################################

def create_dataset(params: dict) -> dict[float, dict[int, dict]]:
    return build_pipeline_test_dataset(params)



##########################################################################################################
# First step - detection - find the places of the uw using differential correlation and find grid function
##########################################################################################################

def detection(signal: np.ndarray, middle_cpfsk_uw: np.ndarray, threshold: float, sps: int, sample_rate: float,
                          uw_spacing_samples: int, n_uw: int | None, grid_tolerance_samples: int,
                          uw_n_symbols: int, n_symbols_each_side: int,
                          min_peak_distance_samples: int, min_n_uw: int,
                          first_peak_weight: float,
                          complex_phase_correlation_shift_samples: int | None = None,
                          frequency_correction_origin_sample: int = 0,
                          full_uw: np.ndarray = None,
                          initial_phase_future_guard_symbols: int = 0) -> dict:
    differential_lag_samples = int(sps)
    detector = UwDetector(method_type="differential_correlation", differential_lag_samples=differential_lag_samples)
    differential_correlation = detector.estimate(signal=np.asarray(signal, dtype=np.complex64),
                                    cpfsk_uw=np.asarray(middle_cpfsk_uw, dtype=np.complex64))
    differential_correlation_abs = np.abs(differential_correlation)

    grid = find_best_uw_grid(correlation=np.asarray(differential_correlation_abs, dtype=float),
                             uw_spacing=int(uw_spacing_samples),
                             threshold=float(threshold),
                             n_uw=n_uw,
                             spacing_tolerance=int(grid_tolerance_samples),
                             min_peak_distance=int(min_peak_distance_samples),
                             min_n_uw=int(min_n_uw),
                             first_peak_weight=float(first_peak_weight))

    offset_differential = len(middle_cpfsk_uw) // 2 + (len(full_uw) - len(middle_cpfsk_uw))//2 - differential_lag_samples / 2
    uw_start_samples = [int(peak - offset_differential) for peak in grid.uw_positions]
    detected = len(uw_start_samples) >= int(min_n_uw)
    if n_uw is not None:
        detected = detected and len(uw_start_samples) == int(n_uw)

    frequency_offset_phase_rad = float("nan")
    frequency_offset_estimation_hz = float("nan")
    if len(grid.uw_positions) > 0:
        peak_phases = np.asarray([np.angle(differential_correlation[int(peak_index)])
                                  for peak_index in grid.uw_positions], dtype=np.float64)
        frequency_offset_phase_rad = float(np.angle(np.mean(np.exp(1j * peak_phases))))
        frequency_offset_estimation_hz = float(frequency_offset_phase_rad * float(sample_rate) / (TWO_PI * float(sps)))

    initial_phase_estimation_rad = float("nan")
    initial_phase_correlation_score = float("nan")
    initial_phase_reference_len = 0
    complex_phase_indices = []

    if len(grid.uw_positions) > 0 and np.isfinite(frequency_offset_estimation_hz):
        complex_correlation_instance = UwDetector(method_type="complex_correlation")
        n = np.arange(len(signal), dtype=np.float32) - np.float32(frequency_correction_origin_sample)
        phase_scale = np.float32(-2.0 * np.pi / float(sample_rate))
        corrected_signal = signal * np.exp(1j * phase_scale * np.float32(frequency_offset_estimation_hz) * n).astype(np.complex64)
        if full_uw is not None and uw_start_samples:
            phase_reference = np.asarray(full_uw, dtype=np.complex64).reshape(-1)
            phase_reference_symbols = max(1, int(uw_n_symbols) - int(initial_phase_future_guard_symbols))
            phase_reference_len = min(len(phase_reference), max(int(sps), phase_reference_symbols * int(sps)))
            phase_reference = phase_reference[:phase_reference_len]
            predicted_complex_index = int(uw_start_samples[0]) + int(len(phase_reference) // 2)
        else:
            phase_reference = np.asarray(middle_cpfsk_uw, dtype=np.complex64).reshape(-1)
            predicted_complex_index = int(grid.uw_positions[0]) + int(complex_phase_correlation_shift_samples or 0)

        complex_correlation = complex_correlation_instance.estimate(corrected_signal, cpfsk_uw=phase_reference)
        search_radius = max(int(grid_tolerance_samples), int(sps))
        search_start = max(0, predicted_complex_index - search_radius)
        search_stop = min(len(complex_correlation), predicted_complex_index + search_radius + 1)
        if search_start < search_stop:
            local_scores = np.abs(complex_correlation[search_start:search_stop])
            complex_index = int(search_start + int(np.argmax(local_scores)))
        else:
            complex_index = int(predicted_complex_index)
        complex_index = int(min(max(complex_index, 0), len(complex_correlation) - 1))

        complex_phase_indices = [complex_index]
        initial_phase_estimation_rad = float(np.angle(complex_correlation[complex_index]))
        initial_phase_correlation_score = float(np.abs(complex_correlation[complex_index]))
        initial_phase_reference_len = int(len(phase_reference))

    return {
        "detected": bool(detected),
        "uw_start_samples": uw_start_samples,
        "peak_indices": list(grid.uw_positions),
        "peak_values": list(grid.peak_values),
        "grid_score": float(grid.score),
        "frequency_offset_estimation_phase_rad": frequency_offset_phase_rad,
        "frequency_offset_estimation_hz": frequency_offset_estimation_hz,
        "initial_phase_estimation_rad": initial_phase_estimation_rad,
        "_initial_phase_for_viterbi_rad": initial_phase_estimation_rad,
        "initial_phase_correlation_score": initial_phase_correlation_score,
        "initial_phase_reference_len": initial_phase_reference_len,
        "initial_phase_correlation_indices": complex_phase_indices,
    }


##########################################################################################################
# second step - coarse frequency estimation using differential circular coarse estimation
##########################################################################################################
def estimate_frequency_offset_from_detected_uws(signal: np.ndarray, uw_start_samples: list[int], sample_rate: float) -> dict:
    signal = np.asarray(signal, dtype=np.complex64).reshape(-1)
    if len(signal) < 2 or not uw_start_samples:
        return {"frequency_offset_hz": float("nan")}

    start = max(int(uw_start_samples[0]), 0)
    estimated_frequency_offset = FrequencyOffsetEstimator.differential_circular_coarse_estimation(
        signal=signal[start:],
        sample_rate=float(sample_rate),
    )

    return {"frequency_offset_hz": estimated_frequency_offset}


##########################################################################################################
# third step - fine frequency estimation using hypothesis
##########################################################################################################
def fine_frequency_estimation(signal: np.ndarray, sample_rate: float, coarse_estimation: float, middle_cpfsk_uw: np.ndarray) -> dict:
    if not np.isfinite(coarse_estimation):
        return {"frequency_offset_hz": float("nan")}
    signal = np.asarray(signal, dtype=np.complex64).reshape(-1)
    middle_cpfsk_uw = np.asarray(middle_cpfsk_uw, dtype=np.complex64).reshape(-1)
    grid_of_frequency_estimations = np.arange(coarse_estimation - 100, coarse_estimation + 100, 10, dtype=np.float32)
    complex_correlation_instance = UwDetector(method_type="complex_correlation")
    n = np.arange(len(signal), dtype=np.float32)
    phase_scale = np.float32(-2.0 * np.pi / float(sample_rate))
    best_peak_list = []
    for frequency in grid_of_frequency_estimations:
        current_corrected_signal = signal * np.exp(1j * phase_scale * np.float32(frequency) * n).astype(np.complex64)
        complex_correlation = complex_correlation_instance.estimate(current_corrected_signal, middle_cpfsk_uw)
        best_peak_list.append(float(np.max(complex_correlation)))
    best_peak_array = np.asarray(best_peak_list, dtype=np.float32)
    best_frequency_estimation = grid_of_frequency_estimations[np.argmax(best_peak_array)]
    return {"frequency_offset_hz": float(best_frequency_estimation)}


def calculate_remaining_phase_at_uws(signal: np.ndarray, cpfsk_uw: np.ndarray, uw_start_samples: list[int],
                                     sample_rate: float, search_radius_samples: int = 0) -> list[dict]:
    signal = np.asarray(signal, dtype=np.complex64).reshape(-1)
    cpfsk_uw = np.asarray(cpfsk_uw, dtype=np.complex64).reshape(-1)
    if len(signal) == 0 or len(cpfsk_uw) == 0:
        return []

    reference_norm = float(np.linalg.norm(cpfsk_uw))
    if reference_norm <= 0.0:
        return []

    diagnostics = []
    radius = int(search_radius_samples)
    uw_len = len(cpfsk_uw)

    for uw_index, nominal_start in enumerate(uw_start_samples):
        best_start = int(nominal_start)
        best_correlation = 0.0 + 0.0j
        best_score = -1.0

        for shift in range(-radius, radius + 1):
            start = int(nominal_start) + shift
            stop = start + uw_len
            if start < 0 or stop > len(signal):
                continue

            segment = signal[start:stop]
            segment_norm = float(np.linalg.norm(segment))
            if segment_norm <= 0.0:
                continue

            correlation = np.vdot(cpfsk_uw, segment) / (reference_norm * segment_norm)
            score = float(np.abs(correlation))
            if score > best_score:
                best_score = score
                best_correlation = complex(correlation)
                best_start = start

        if best_score < 0.0:
            continue

        segment = signal[best_start:best_start + uw_len]
        residual = segment * np.conj(cpfsk_uw)
        time_sec = np.arange(best_start, best_start + uw_len, dtype=np.float64) / float(sample_rate)
        relative_time_sec = time_sec - (float(best_start) / float(sample_rate))
        phase_rad = np.unwrap(np.angle(residual))

        diagnostics.append({
            "uw_index": int(uw_index),
            "nominal_start_sample": int(nominal_start),
            "best_start_sample": int(best_start),
            "start_error_samples": int(best_start) - int(nominal_start),
            "correlation": best_correlation,
            "correlation_score": float(best_score),
            "correlation_phase_rad": float(np.angle(best_correlation)),
            "relative_time_sec": relative_time_sec,
            "time_sec": time_sec,
            "remaining_phase_rad": phase_rad,
        })

    return diagnostics


##########################################################################################################
# forth step - coherent detection using viterbi
##########################################################################################################
@dataclass
class CpfskViterbiConfig:
    sample_rate: float
    symbol_time: float
    h: float = 0.5
    pulse_shape_type: str = "rrc"
    pulse_normalization: str = "cpfsk"
    span_in_symbols: int = 10
    pulse_causal: bool = False
    constellation_type: str = "PAM"
    constellation_order: int = 4
    depth_symbols: int = 6
    convolution_mode: str = "same"
    phase_mode: str = "pll"  # Kept for old callers; Viterbi always uses PLL tracking.
    initial_phase: float = 0.0
    initial_frequency_hz: float = 0.0
    pll_alpha: float = 0.05
    pll_beta: float = 0.006
    max_survivors: int | None = None
    n_symbols: int | None = None
    metric_memory_symbols: int | None = None
    metric_type: str = "real"
    debug: bool = False


def differentiate(rx_signal: np.ndarray, sample_rate: float, frequency_sensitivity: float) -> np.ndarray:
    rx_signal = np.asarray(rx_signal, dtype=np.complex128).reshape(-1)
    if len(rx_signal) < 2:
        return np.array([], dtype=np.float64)

    phase = np.unwrap(np.angle(rx_signal))
    return np.diff(phase) * float(sample_rate) / (TWO_PI * float(frequency_sensitivity))


def differentiate_demodulate(rx_signal: np.ndarray, config: CpfskViterbiConfig,
                             frequency_offset_hz: float | None = None) -> np.ndarray:
    sps = int(round(float(config.sample_rate) * float(config.symbol_time)))
    if sps <= 0:
        raise ValueError("sample_rate * symbol_time must be positive")

    pulse = _config_pulse(config)
    constellation_points = _config_constellation_points(config)
    constellation_bits = _cached_constellation_bits(int(config.constellation_order))

    discriminator = differentiate(rx_signal=rx_signal,
                                  sample_rate=config.sample_rate,
                                  frequency_sensitivity=config.h)
    if len(discriminator) == 0:
        return np.array([], dtype=np.uint8)

    if frequency_offset_hz is None:
        frequency_offset_hz = config.initial_frequency_hz
    discriminator = discriminator - float(frequency_offset_hz) / float(config.h)

    matched_filter = pulse[::-1]
    matched_energy = float(np.sum(np.abs(matched_filter) ** 2))
    if matched_energy <= 0.0:
        return np.array([], dtype=np.uint8)

    filtered = np.convolve(discriminator, matched_filter / matched_energy, mode="full")
    # TX uses np.convolve(..., mode="same"), so the transmit pulse is centered
    # on the symbol grid. The RX matched filter adds half the pulse delay, and
    # diff(...) advances the discriminator by one sample.
    sampling_offset = max((len(pulse) - 1) // 2 - 1, 0)
    sample_positions = np.arange(sampling_offset, len(filtered), sps, dtype=int)
    if config.n_symbols is not None:
        sample_positions = sample_positions[:int(config.n_symbols)]

    decided_indices = []
    for sample_position in sample_positions:
        value = filtered[sample_position]
        decided_indices.append(int(np.argmin(np.abs(value - constellation_points))))

    if not decided_indices:
        return np.array([], dtype=np.uint8)

    return constellation_bits[np.asarray(decided_indices, dtype=np.int64)].reshape(-1).astype(np.uint8)


@njit(cache=True)
def _cpfsk_branch_metric_numba(rx_signal: np.ndarray, pulse: np.ndarray, constellation_points: np.ndarray,
                               path_indices: np.ndarray, path_len: int, score_symbol_index: int,
                               sps: int, sample_rate: float, symbol_time: float, h: float,
                               initial_phase: float, pll_phase_rad: float, pll_frequency_hz: float,
                               use_abs_metric: bool) -> tuple[float, float]:
    pulse_center = (len(pulse) - 1) // 2
    chunk_start = score_symbol_index * sps
    chunk_stop = chunk_start + sps
    if chunk_stop > len(rx_signal):
        chunk_stop = len(rx_signal)
    if chunk_start >= chunk_stop:
        return 0.0, 0.0

    message = np.zeros(chunk_stop, dtype=np.float32)
    for symbol_index in range(path_len):
        symbol_value = constellation_points[path_indices[symbol_index]]
        symbol_sample = symbol_index * sps
        for pulse_index in range(len(pulse)):
            sample_index = pulse_index - pulse_center + symbol_sample
            if 0 <= sample_index < chunk_stop:
                message[sample_index] += symbol_value * pulse[pulse_index]

    phase = initial_phase + pll_phase_rad
    corr_real = 0.0
    corr_imag = 0.0

    for sample_index in range(chunk_stop):
        phase += TWO_PI * h * message[sample_index] / sample_rate
        phase_with_frequency = phase + TWO_PI * pll_frequency_hz * (sample_index / sample_rate)

        if sample_index >= chunk_start:
            c = np.cos(phase_with_frequency)
            s = np.sin(phase_with_frequency)
            rx = rx_signal[sample_index]
            # conj(exp(j*phase))*rx
            corr_real += c * rx.real + s * rx.imag
            corr_imag += c * rx.imag - s * rx.real

    phase_error = np.arctan2(corr_imag, corr_real)
    if use_abs_metric:
        branch_metric = -np.sqrt(corr_real * corr_real + corr_imag * corr_imag)
    else:
        branch_metric = -corr_real
    return branch_metric, phase_error


@njit(cache=True)
def cpfsk_viterbi_core_numba(rx_signal: np.ndarray, pulse: np.ndarray, constellation_points: np.ndarray,
                              n_symbols: int, sps: int, sample_rate: float, symbol_time: float, h: float,
                              initial_phase: float, initial_frequency_hz: float, pll_alpha: float,
                              pll_beta: float, max_survivors: int,
                              use_abs_metric: bool) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, int, np.ndarray]:
    modulation_order = len(constellation_points)
    pulse_center = (len(pulse) - 1) // 2
    future_symbols = int(np.ceil(max(0, len(pulse) - 1 - pulse_center) / sps))
    score_delay_symbols = future_symbols
    max_candidates = max_survivors * modulation_order

    survivor_paths = np.zeros((max_survivors, n_symbols), dtype=np.int16)
    survivor_lengths = np.zeros(max_survivors, dtype=np.int32)
    survivor_metrics = np.empty(max_survivors, dtype=np.float64)
    survivor_metrics[0] = 0.0
    survivor_pll_phase = np.zeros(max_survivors, dtype=np.float64)
    survivor_pll_frequency = np.empty(max_survivors, dtype=np.float64)
    survivor_pll_frequency[0] = initial_frequency_hz
    survivor_phase_errors = np.zeros((max_survivors, n_symbols), dtype=np.float64)
    survivor_frequencies = np.zeros((max_survivors, n_symbols), dtype=np.float64)
    survivor_trace_lengths = np.zeros(max_survivors, dtype=np.int32)
    n_survivors = 1

    candidate_paths = np.zeros((max_candidates, n_symbols), dtype=np.int16)
    candidate_lengths = np.zeros(max_candidates, dtype=np.int32)
    candidate_metrics = np.zeros(max_candidates, dtype=np.float64)
    candidate_pll_phase = np.zeros(max_candidates, dtype=np.float64)
    candidate_pll_frequency = np.zeros(max_candidates, dtype=np.float64)
    candidate_phase_errors = np.zeros((max_candidates, n_symbols), dtype=np.float64)
    candidate_frequencies = np.zeros((max_candidates, n_symbols), dtype=np.float64)
    candidate_trace_lengths = np.zeros(max_candidates, dtype=np.int32)

    for step in range(n_symbols + score_delay_symbols):
        candidate_count = 0
        n_candidate_symbols = modulation_order if step < n_symbols else 1
        score_symbol_index = step - score_delay_symbols

        for survivor_index in range(n_survivors):
            for candidate_symbol_position in range(n_candidate_symbols):
                candidate_symbol = candidate_symbol_position if step < n_symbols else -1
                path_len = survivor_lengths[survivor_index]
                new_path_len = path_len

                for path_index in range(path_len):
                    candidate_paths[candidate_count, path_index] = survivor_paths[survivor_index, path_index]

                if candidate_symbol >= 0 and path_len < n_symbols:
                    candidate_paths[candidate_count, path_len] = candidate_symbol
                    new_path_len = path_len + 1

                candidate_lengths[candidate_count] = new_path_len
                metric = survivor_metrics[survivor_index]
                pll_phase = survivor_pll_phase[survivor_index]
                pll_frequency = survivor_pll_frequency[survivor_index]
                trace_len = survivor_trace_lengths[survivor_index]

                for trace_index in range(trace_len):
                    candidate_phase_errors[candidate_count, trace_index] = survivor_phase_errors[survivor_index, trace_index]
                    candidate_frequencies[candidate_count, trace_index] = survivor_frequencies[survivor_index, trace_index]

                if 0 <= score_symbol_index < n_symbols and score_symbol_index < new_path_len:
                    branch_metric, phase_error = _cpfsk_branch_metric_numba(
                        rx_signal=rx_signal,
                        pulse=pulse,
                        constellation_points=constellation_points,
                        path_indices=candidate_paths[candidate_count],
                        path_len=new_path_len,
                        score_symbol_index=score_symbol_index,
                        sps=sps,
                        sample_rate=sample_rate,
                        symbol_time=symbol_time,
                        h=h,
                        initial_phase=initial_phase,
                        pll_phase_rad=pll_phase,
                        pll_frequency_hz=pll_frequency,
                        use_abs_metric=use_abs_metric,
                    )
                    metric += branch_metric
                    pll_frequency += pll_beta * phase_error / (TWO_PI * symbol_time)
                    pll_phase += pll_alpha * phase_error
                    if trace_len < n_symbols:
                        candidate_phase_errors[candidate_count, trace_len] = phase_error
                        candidate_frequencies[candidate_count, trace_len] = pll_frequency
                        trace_len += 1

                candidate_metrics[candidate_count] = metric
                candidate_pll_phase[candidate_count] = pll_phase
                candidate_pll_frequency[candidate_count] = pll_frequency
                candidate_trace_lengths[candidate_count] = trace_len
                candidate_count += 1

        order = np.argsort(candidate_metrics[:candidate_count])
        n_survivors = min(max_survivors, candidate_count)
        for output_index in range(n_survivors):
            source_index = order[output_index]
            survivor_lengths[output_index] = candidate_lengths[source_index]
            survivor_metrics[output_index] = candidate_metrics[source_index]
            survivor_pll_phase[output_index] = candidate_pll_phase[source_index]
            survivor_pll_frequency[output_index] = candidate_pll_frequency[source_index]
            survivor_trace_lengths[output_index] = candidate_trace_lengths[source_index]
            for path_index in range(n_symbols):
                survivor_paths[output_index, path_index] = candidate_paths[source_index, path_index]
            for trace_index in range(n_symbols):
                survivor_phase_errors[output_index, trace_index] = candidate_phase_errors[source_index, trace_index]
                survivor_frequencies[output_index, trace_index] = candidate_frequencies[source_index, trace_index]

    best_index = 0
    best_metric = survivor_metrics[0]
    for survivor_index in range(1, n_survivors):
        if survivor_metrics[survivor_index] < best_metric:
            best_metric = survivor_metrics[survivor_index]
            best_index = survivor_index

    best_path = survivor_paths[best_index].copy()
    trace_len = survivor_trace_lengths[best_index]
    best_phase_errors = survivor_phase_errors[best_index].copy()
    best_frequencies = survivor_frequencies[best_index].copy()
    final_metrics = survivor_metrics[:n_survivors].copy()
    return best_path, best_metric, best_phase_errors, best_frequencies, trace_len, final_metrics


def cpfsk_viterbi_demodulate(rx_signal: np.ndarray, config: CpfskViterbiConfig) -> dict:
    """
    General, readable CPFSK Viterbi demodulator.

    The important idea:
    - CPFSK is not memoryless when the frequency pulse is longer than one symbol.
    - A received chunk around symbol k can depend on symbols before k and, for a
      centered RRC pulse, also symbols after k.
    - Therefore the Viterbi waits until enough future symbols are hypothesized
      before it scores a symbol chunk.

    This implementation intentionally keeps the whole branch metric in this one
    function. It is not the fastest version, but it is the easiest one to inspect
    and it works with:
    - rect pulse, span 1
    - RRC pulse with any span
    - centered/noncausal pulse shapes generated by the current modulator
    """
    rx_signal = np.asarray(rx_signal, dtype=np.complex64).reshape(-1)
    sample_rate = float(config.sample_rate)
    symbol_time = float(config.symbol_time)
    sps = int(round(sample_rate * symbol_time))

    # Build the exact same pulse and constellation objects used by the modulator.
    # These arrays are cached across calls because they are identical for every
    # signal in a sweep.
    full_pulse = _config_pulse(config)
    pulse = full_pulse
    if config.metric_memory_symbols is not None:
        memory_samples_each_side = int(config.metric_memory_symbols) * sps
        full_pulse_center = (len(full_pulse) - 1) // 2
        truncate_start = max(0, full_pulse_center - memory_samples_each_side)
        truncate_stop = min(len(full_pulse), full_pulse_center + memory_samples_each_side + 1)
        pulse = full_pulse[truncate_start:truncate_stop]
    constellation_points = _config_constellation_points(config)
    constellation_bits = _cached_constellation_bits(int(config.constellation_order))

    n_symbols = int(config.n_symbols) if config.n_symbols is not None else int(np.ceil(len(rx_signal) / sps))
    modulation_order = len(constellation_points)
    bits_per_symbol = int(math.log2(modulation_order))
    upsampled_len = max((n_symbols - 1) * sps + 1, 1)
    symbol_positions_all = np.arange(n_symbols, dtype=np.int32) * int(sps)
    if config.convolution_mode == "same":
        message_len_for_time_axis = max(upsampled_len, len(pulse))
    elif config.convolution_mode == "full":
        message_len_for_time_axis = upsampled_len + len(pulse) - 1
    elif config.convolution_mode == "valid":
        message_len_for_time_axis = max(0, max(upsampled_len, len(pulse)) - min(upsampled_len, len(pulse)) + 1)
    else:
        message_len_for_time_axis = max(upsampled_len, len(pulse))
    time_axis = np.arange(message_len_for_time_axis, dtype=np.float32) / np.float32(sample_rate)

    # The pulse center tells us how many future symbols can affect the current
    # samples. For a centered RRC span 20, this is about 10 symbols. For rect span
    # 1 it is small, so the same code still works.
    pulse_center = (len(pulse) - 1) // 2
    future_symbols = int(math.ceil(max(0, len(pulse) - 1 - pulse_center) / sps))
    score_delay_symbols = future_symbols

    # depth_symbols controls the default number of survivors. If max_survivors is
    # provided, it overrides this. Beam pruning is the only approximation here.
    if config.max_survivors is None:
        max_survivors = int(modulation_order ** max(1, int(config.depth_symbols)))
    else:
        max_survivors = int(config.max_survivors)
    max_survivors = max(1, max_survivors)
    metric_type = str(config.metric_type).lower()
    if metric_type not in {"real", "abs"}:
        raise ValueError("metric_type must be 'real' or 'abs'")
    use_abs_metric = metric_type == "abs"

    if not config.debug and config.convolution_mode == "same":
        best_path, best_metric, phase_errors, frequencies_hz, trace_len, survivor_metrics = cpfsk_viterbi_core_numba(
            rx_signal=np.ascontiguousarray(rx_signal),
            pulse=np.ascontiguousarray(pulse),
            constellation_points=np.ascontiguousarray(constellation_points),
            n_symbols=int(n_symbols),
            sps=int(sps),
            sample_rate=float(sample_rate),
            symbol_time=float(symbol_time),
            h=float(config.h),
            initial_phase=float(config.initial_phase),
            initial_frequency_hz=float(config.initial_frequency_hz),
            pll_alpha=float(config.pll_alpha),
            pll_beta=float(config.pll_beta),
            max_survivors=int(max_survivors),
            use_abs_metric=bool(use_abs_metric),
        )
        decoded_indices = best_path[:n_symbols].astype(np.int64)
        decoded_bits = constellation_bits[decoded_indices].reshape(-1).astype(np.uint8)
        decoded_bits = decoded_bits[:n_symbols * bits_per_symbol]
        phase_errors = phase_errors[:trace_len]
        frequencies_hz = frequencies_hz[:trace_len]
        result = {
            "bits": decoded_bits,
            "symbol_indices": decoded_indices,
            "metric": float(best_metric),
            "metrics": survivor_metrics.astype(np.float64),
            "depth_symbols": int(config.depth_symbols),
            "full_future_symbols": int(future_symbols),
            "score_delay_symbols": int(score_delay_symbols),
            "max_survivors": int(max_survivors),
            "metric_memory_symbols": config.metric_memory_symbols,
            "metric_type": metric_type,
            "pll_phase_errors_rad": phase_errors.astype(np.float64),
            "pll_frequencies_hz": frequencies_hz.astype(np.float64),
        }
        result["tracking_phase_error_rad"] = result["pll_phase_errors_rad"]
        result["tracked_frequency_hz"] = result["pll_frequencies_hz"]
        return result

    # A survivor is one candidate symbol path plus its accumulated metric. PLL
    # state is stored per survivor because different symbol decisions can produce
    # different phase-error histories.
    survivors = [{
        "path": tuple(),
        "metric": 0.0,
        "pll_phase_rad": 0.0,
        "pll_frequency_hz": float(config.initial_frequency_hz),
        "phase_errors": [],
        "frequencies_hz": [],
    }]

    # Run through all real symbols, then run a tail of score_delay_symbols steps.
    # During the tail no new symbols are added; it only scores the last delayed
    # chunks after enough future context is known.
    for step in range(n_symbols + score_delay_symbols):
        candidate_indices = range(modulation_order) if step < n_symbols else (None,)
        expanded_survivors = []

        for survivor in survivors:
            for candidate_index in candidate_indices:
                if candidate_index is None:
                    candidate_path = survivor["path"]
                else:
                    candidate_path = survivor["path"] + (int(candidate_index),)

                score_symbol_index = step - score_delay_symbols

                # If score_symbol_index is negative, this candidate does not yet
                # have enough future symbols to score any received chunk.
                if score_symbol_index < 0:
                    expanded_survivors.append({
                        "path": candidate_path,
                        "metric": float(survivor["metric"]),
                        "pll_phase_rad": float(survivor["pll_phase_rad"]),
                        "pll_frequency_hz": float(survivor["pll_frequency_hz"]),
                        "phase_errors": list(survivor["phase_errors"]),
                        "frequencies_hz": list(survivor["frequencies_hz"]),
                    })
                    continue

                # During the tail, the candidate path does not grow. Stop scoring
                # when all real symbols have been scored.
                if score_symbol_index >= len(candidate_path) or score_symbol_index >= n_symbols:
                    expanded_survivors.append({
                        "path": candidate_path,
                        "metric": float(survivor["metric"]),
                        "pll_phase_rad": float(survivor["pll_phase_rad"]),
                        "pll_frequency_hz": float(survivor["pll_frequency_hz"]),
                        "phase_errors": list(survivor["phase_errors"]),
                        "frequencies_hz": list(survivor["frequencies_hz"]),
                    })
                    continue

                # Convert the candidate path into the real-valued CPM message
                # exactly like the transmitter:
                #   bits -> PAM symbols -> upsample -> pulse filter
                #
                # Important detail:
                # np.convolve(..., mode="same") returns length max(len(a), len(v)).
                # If we convolve only a short partial path while the pulse is
                # longer than that path, the alignment is not the same as the
                # transmitter. Therefore every branch is placed into a full
                # zero-padded transmit-length sequence before filtering.
                path_symbols = constellation_points[np.asarray(candidate_path, dtype=np.int32)]
                upsampled = np.zeros(upsampled_len, dtype=np.float32)
                symbol_positions = symbol_positions_all[:len(path_symbols)]
                valid_positions = symbol_positions < len(upsampled)
                upsampled[symbol_positions[valid_positions]] = path_symbols[valid_positions]

                message = np.convolve(upsampled, pulse, mode=config.convolution_mode)

                # Convert the CPM message into complex baseband CPFSK exactly like
                # the transmitter:
                #   phase[n] = 2*pi*h*sum(message)/Fs + 2*pi*f0*t + phase0
                n_message = len(message)
                t = time_axis[:n_message]
                phase = (
                    TWO_PI * float(config.h) * np.cumsum(message) / sample_rate
                    + TWO_PI * float(survivor["pll_frequency_hz"]) * t
                    + float(config.initial_phase)
                    + float(survivor["pll_phase_rad"])
                )
                expected_signal = np.exp(1j * phase)

                chunk_start = int(score_symbol_index * sps)
                chunk_stop = min(int(chunk_start + sps), len(rx_signal), len(expected_signal))
                if chunk_start >= chunk_stop:
                    branch_metric = 0.0
                    phase_error = 0.0
                else:
                    rx_chunk = rx_signal[chunk_start:chunk_stop]
                    expected_chunk = expected_signal[chunk_start:chunk_stop]

                    # Branch metric options:
                    #   real: coherent metric, -Re{sum conj(expected)*rx}
                    #   abs: phase-insensitive metric, -abs{sum conj(expected)*rx}
                    #
                    # The complex correlation still gives the phase detector for
                    # the second-order decision-directed PLL below.
                    correlation = np.vdot(expected_chunk, rx_chunk)
                    phase_error = float(np.angle(correlation))
                    if use_abs_metric:
                        branch_metric = -float(np.abs(correlation))
                    else:
                        branch_metric = -float(np.real(correlation))

                next_pll_phase = float(survivor["pll_phase_rad"])
                next_pll_frequency = float(survivor["pll_frequency_hz"])
                phase_errors = list(survivor["phase_errors"])
                frequencies_hz = list(survivor["frequencies_hz"])

                next_pll_frequency += (float(config.pll_beta) * phase_error / (TWO_PI * symbol_time))
                next_pll_phase += float(config.pll_alpha) * phase_error
                phase_errors.append(phase_error)
                frequencies_hz.append(next_pll_frequency)

                expanded_survivors.append({
                    "path": candidate_path,
                    "metric": float(survivor["metric"]) + branch_metric,
                    "pll_phase_rad": next_pll_phase,
                    "pll_frequency_hz": next_pll_frequency,
                    "phase_errors": phase_errors,
                    "frequencies_hz": frequencies_hz,
                })

        # Beam pruning: keep only the best survivors. Without this, the number of
        # paths is M^N. With this, the number of paths is bounded.
        expanded_survivors.sort(key=lambda item: item["metric"])
        survivors = expanded_survivors[:max_survivors]

    best_survivor = min(survivors, key=lambda item: item["metric"])
    decoded_indices = np.asarray(best_survivor["path"][:n_symbols], dtype=np.int64)
    decoded_bits = constellation_bits[decoded_indices].reshape(-1) if len(decoded_indices) else np.array([], dtype=np.uint8)

    expected_n_bits = n_symbols * bits_per_symbol
    decoded_bits = decoded_bits[:expected_n_bits]

    phase_errors_array = np.asarray(best_survivor["phase_errors"], dtype=np.float64)

    result = {
        "bits": decoded_bits.astype(np.uint8),
        "symbol_indices": decoded_indices,
        "metric": float(best_survivor["metric"]),
        "metrics": [float(survivor["metric"]) for survivor in survivors],
        "depth_symbols": int(config.depth_symbols),
        "full_future_symbols": int(future_symbols),
        "score_delay_symbols": int(score_delay_symbols),
        "max_survivors": int(max_survivors),
        "metric_memory_symbols": config.metric_memory_symbols,
        "metric_type": metric_type,
        "pll_phase_errors_rad": phase_errors_array,
        "pll_frequencies_hz": np.asarray(best_survivor["frequencies_hz"], dtype=np.float64),
    }
    result["tracking_phase_error_rad"] = result["pll_phase_errors_rad"]
    result["tracked_frequency_hz"] = result["pll_frequencies_hz"]

    if config.debug:
        best_symbols = constellation_points[decoded_indices] if len(decoded_indices) else np.array([], dtype=np.float32)
        best_upsampled = np.zeros(upsampled_len, dtype=np.float32)
        if len(best_symbols):
            best_upsampled[np.arange(len(best_symbols), dtype=int) * sps] = best_symbols
        best_message = np.convolve(best_upsampled, pulse, mode=config.convolution_mode)
        t = np.arange(len(best_message), dtype=np.float32) / sample_rate
        best_phase = (
            TWO_PI * float(config.h) * np.cumsum(best_message) / sample_rate
            + TWO_PI * float(config.initial_frequency_hz) * t
            + float(config.initial_phase)
        )
        best_expected_signal = np.exp(1j * best_phase)

        result["debug"] = {
            "rx_signal": rx_signal.copy(),
            "full_pulse": full_pulse.copy(),
            "pulse": pulse.copy(),
            "constellation_points": constellation_points.copy(),
            "sps": int(sps),
            "n_symbols": int(n_symbols),
            "score_delay_symbols": int(score_delay_symbols),
            "decoded_symbols": best_symbols.copy(),
            "decoded_upsampled_symbols": best_upsampled.copy(),
            "decoded_message": best_message.copy(),
            "decoded_expected_signal": best_expected_signal.copy(),
        }

    return result


##########################################################################################################
# Last step - full receiver pipeline for one signal
##########################################################################################################

def build_viterbi_config(params: dict, n_symbols: int, frequency_offset_hz: float,
                         initial_phase: float = 0.0) -> CpfskViterbiConfig:
    return CpfskViterbiConfig(
        sample_rate=params["sample_rate"],
        symbol_time=params["symbol_time"],
        h=params.get("h", 0.5),
        pulse_shape_type=params["pulse_shape_type"],
        pulse_normalization=params["pulse_normalization"],
        span_in_symbols=params["span_in_symbols"],
        pulse_causal=params.get("pulse_causal", False),
        constellation_type=params["constellation_type"],
        constellation_order=params["constellation_order"],
        depth_symbols=params.get("viterbi_depth_symbols", 6),
        max_survivors=params.get("viterbi_max_survivors", None),
        initial_frequency_hz=float(frequency_offset_hz),
        initial_phase=initial_phase,
        pll_alpha=params.get("pll_alpha", 0.05),
        pll_beta=params.get("pll_beta", 0.006),
        n_symbols=int(n_symbols),
        metric_memory_symbols=params.get("viterbi_metric_memory_symbols", None),
        metric_type=params.get("viterbi_metric_type", "real"),
        debug=bool(params.get("viterbi_debug", False)),
    )


def differential_demodulation(signal: np.ndarray, params: dict, n_symbols: int,
                              frequency_offset_hz: float) -> np.ndarray:
    config = build_viterbi_config(
        params=params,
        n_symbols=int(n_symbols),
        frequency_offset_hz=float(frequency_offset_hz),
    )
    return differentiate_demodulate(
        rx_signal=np.asarray(signal, dtype=np.complex128),
        config=config,
        frequency_offset_hz=float(frequency_offset_hz),
    )


def pipeline(signal: np.ndarray, middle_cpfsk_uw: np.ndarray, params: dict,
             n_symbols: int | None = None, signal_start_sample: int = 0, full_uw: np.ndarray = None) -> dict:
    total_start_time = time.perf_counter()
    timing_sec = {
        "detection": 0.0,
        "coarse_frequency_estimation": 0.0,
        "fine_frequency_estimation": 0.0,
        "demodulation": 0.0,
        "differential_demodulation": 0.0,
        "total": 0.0,
    }

    signal = np.asarray(signal, dtype=np.complex64).reshape(-1)
    sps = int(round(float(params["sample_rate"]) * float(params["symbol_time"])))
    bits_per_symbol = int(np.log2(int(params["constellation_order"])))
    uw_n_symbols = len(params["uw_bits"]) // bits_per_symbol
    n_symbols_each_side = int(params["middle_crop_symbols_each_side"])
    uw_spacing_samples = int(params.get("uw_spacing_samples", round(float(params["uw_spacing_sec"]) * float(params["sample_rate"]))))
    initial_phase_future_guard_symbols = params.get("initial_phase_future_guard_symbols", None)
    if initial_phase_future_guard_symbols is None:
        initial_phase_future_guard_symbols = int(math.ceil(float(params["span_in_symbols"]) / 2.0))
    stage_start_time = time.perf_counter()
    detection_result = detection(signal=signal, middle_cpfsk_uw=middle_cpfsk_uw, threshold=params["detection_threshold"],
                                 sps=sps, sample_rate=params["sample_rate"], uw_spacing_samples=uw_spacing_samples,
                                 n_uw=params.get("detection_n_uw", None),
                                 grid_tolerance_samples=params["grid_tolerance_samples"], uw_n_symbols=uw_n_symbols,
                                 n_symbols_each_side=n_symbols_each_side,
                                 min_peak_distance_samples=params["min_peak_distance_samples"],
                                 min_n_uw=params["min_n_uw_for_detection"],
                                 first_peak_weight=params["grid_first_peak_weight"],
                                 complex_phase_correlation_shift_samples=params.get(
                                     "complex_phase_correlation_shift_samples", sps // 2),
                                 frequency_correction_origin_sample=int(signal_start_sample),
                                 full_uw=full_uw,
                                 initial_phase_future_guard_symbols=int(initial_phase_future_guard_symbols))
    timing_sec["detection"] = time.perf_counter() - stage_start_time

    coarse_frequency_estimation = {"frequency_offset_hz": float("nan")}
    fine_frequency_estimation_result = {"frequency_offset_hz": float("nan")}
    frequency_estimation = {"frequency_offset_hz": float("nan"), "source": "detection"}
    demodulation = None
    differential_demodulation_bits = None
    run_demodulation = bool(params.get("run_demodulation", True))

    if detection_result["detected"] and run_demodulation:
        detection_frequency_offset_hz = float(detection_result["frequency_offset_estimation_hz"])
        frequency_estimation = {
            "frequency_offset_hz": detection_frequency_offset_hz,
            "source": "detection",
        }
        phase_estimation = float(detection_result.get("_initial_phase_for_viterbi_rad", 0.0))
        if not np.isfinite(phase_estimation):
            phase_estimation = 0.0

        rx_for_demod = signal[int(signal_start_sample):]
        if n_symbols is None:
            n_symbols = max(0, int(np.ceil(len(rx_for_demod) / sps)))
        demod_config = build_viterbi_config(params=params, n_symbols=int(n_symbols),
                                            frequency_offset_hz=detection_frequency_offset_hz,
                                            initial_phase=phase_estimation)
        stage_start_time = time.perf_counter()
        demodulation = cpfsk_viterbi_demodulate(rx_signal=rx_for_demod, config=demod_config)
        timing_sec["demodulation"] = time.perf_counter() - stage_start_time

        stage_start_time = time.perf_counter()
        differential_demodulation_bits = differential_demodulation(
            signal=rx_for_demod,
            params=params,
            n_symbols=int(n_symbols),
            frequency_offset_hz=detection_frequency_offset_hz,
        )
        timing_sec["differential_demodulation"] = time.perf_counter() - stage_start_time

    timing_sec["total"] = time.perf_counter() - total_start_time

    return {
        "detected": bool(detection_result["detected"]),
        "detection": detection_result,
        "frequency_estimation": frequency_estimation,
        "coarse_frequency_estimation": coarse_frequency_estimation,
        "fine_frequency_estimation": fine_frequency_estimation_result,
        "demodulation": demodulation,
        "differential_demodulation": differential_demodulation_bits,
        "timing_sec": timing_sec,
    }
