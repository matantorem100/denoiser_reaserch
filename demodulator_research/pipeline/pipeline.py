from typing import Any
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demodulator_research.find_grid.find_grid import find_best_uw_grid
from demodulator_research.frequency_offset_estimation.frequency_offset_estimation import FrequencyOffsetEstimator
from demodulator_research.demodulator.demodulator import DemodulatorConfig, Demodulator
from demodulator_research.uw_detection.uw_detection import UwDetector
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.dataset_creation import Dataset, DatasetConfig
from synthetic_dataset_creation.modulator.modulator import ModulatorConfig, Modulator
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig


UW_BITS_20 = [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0]

PULSE_SHAPE_TYPE = "rrc"
SPAN_IN_SYMBOLS = 5
PULSE_NORMALIZATION = "cpfsk"

CONSTELLATION_TYPE = "PAM"
CONSTELLATION_ORDER = 4


def build_dataset_config() -> DatasetConfig:

    return DatasetConfig(
        sample_rate=10000,
        symbol_time=1e-3,
        n_bits_to_transmit=2000,

        min_frequency_offset=-500.0,
        max_frequency_offset=500.0,
        const_frequency_offset=None,

        min_phase_offset=0.0,
        max_phase_offset=2.0 * np.pi,
        const_phase_offset=None,

        pulse_shape_type=PULSE_SHAPE_TYPE,
        span_in_symbols=SPAN_IN_SYMBOLS,
        pulse_normalization=PULSE_NORMALIZATION,

        constellation_type=CONSTELLATION_TYPE,
        constellation_order=CONSTELLATION_ORDER,

        noise_type="snr",
        noise_db_values=list(np.arange(0.0, 22.0, 1.0)),
        n_signals_per_snr=100,

        uw_bits=UW_BITS_20,
        use_uw=True,
        uw_probability=1.0,
        n_uw_values=[0, 3],
        uw_spacing_bits=500,
        random_uw_start=True,
        uw_mode="overwrite",

        random_seed=0,
    )

def build_modulator_config() -> ModulatorConfig:

    pulse_shape_config = PulseShapeConfig(pulse_shape_type=PULSE_SHAPE_TYPE,
                                          normalization_type=PULSE_NORMALIZATION,
                                          span_in_symbols=SPAN_IN_SYMBOLS)

    constellation_config = ConstellationConfig(constellation_type=CONSTELLATION_TYPE,
                                               constellation_order=CONSTELLATION_ORDER)

    modulator_config = ModulatorConfig(pulse_shape_config=pulse_shape_config, constellation_config=constellation_config)


    return modulator_config


def build_demodulator_config() -> DemodulatorConfig:

    pulse_shape_config = PulseShapeConfig(pulse_shape_type=PULSE_SHAPE_TYPE,
                                          normalization_type=PULSE_NORMALIZATION,
                                          span_in_symbols=SPAN_IN_SYMBOLS)

    constellation_config = ConstellationConfig(constellation_type=CONSTELLATION_TYPE,
                                               constellation_order=CONSTELLATION_ORDER)

    demodulator_config = DemodulatorConfig(pulse_shape_config=pulse_shape_config,
                                           constellation_config=constellation_config)

    return demodulator_config


def extract_segment(signal: np.ndarray, start_sample: int, segment_len: int) -> np.ndarray:
    start_sample = max(int(start_sample), 0)
    stop_sample = start_sample + int(segment_len)
    segment = np.asarray(signal, dtype=np.complex128).reshape(-1)[start_sample:stop_sample]

    if len(segment) < segment_len:
        segment = np.pad(segment, (0, segment_len - len(segment)))

    return segment


def estimate_phase_offset(rx_uw_segment: np.ndarray, cpfsk_uw: np.ndarray) -> float:
    n_samples = min(len(rx_uw_segment), len(cpfsk_uw))
    if n_samples == 0:
        return float("nan")

    rx_uw_segment = np.asarray(rx_uw_segment[:n_samples], dtype=np.complex128)
    cpfsk_uw = np.asarray(cpfsk_uw[:n_samples], dtype=np.complex128)

    return float(np.angle(np.vdot(cpfsk_uw, rx_uw_segment)))


def circular_mean(phases: list[float]) -> float:
    if not phases:
        return float("nan")

    phasors = np.exp(1j * np.asarray(phases, dtype=float))
    return float(np.angle(np.mean(phasors)))


def build_cpfsk_uw(dataset_config: DatasetConfig) -> np.ndarray:
    modulator_config = build_modulator_config()
    return Modulator(modulator_config).modulate(
        bits=np.array(UW_BITS_20),
        symbol_time=dataset_config.symbol_time,
        sample_rate=dataset_config.sample_rate,
    )[0]


def compute_uw_correlations(dataset: dict[float, dict[int, dict[str, Any]]], dataset_config: DatasetConfig) -> list[dict[str, Any]]:

    cpfsk_uw = build_cpfsk_uw(dataset_config)

    sps = int(round(dataset_config.sample_rate * dataset_config.symbol_time))
    bits_per_symbol = int(np.log2(dataset_config.constellation_order))
    if dataset_config.uw_spacing_bits is None:
        raise ValueError("dataset_config.uw_spacing_bits is required for grid search")
    uw_spacing_samples = int((dataset_config.uw_spacing_bits // bits_per_symbol) * sps)
    differential_peak_to_start_offset = len(cpfsk_uw) - sps - 1

    uw_detector_instance = UwDetector(method_type="differential_correlation", differential_lag_samples=sps)

    correlation_results = []
    for snr_db, signals_by_index in dataset.items():
        for signal_index, sample in signals_by_index.items():
            correlation = uw_detector_instance.estimate(signal=sample["rx_signal"], cpfsk_uw=cpfsk_uw)

            sample["uw_correlation"] = correlation
            if sample["n_uw"] == 0:
                grid_result = None
                detected_peak_indices = []
                detected_uw_start_samples = []
                detected_uw_start_bits = []

            else:
                grid_result = find_best_uw_grid(correlation=correlation, uw_spacing=uw_spacing_samples, threshold=0.45,
                                                n_uw=sample["n_uw"], spacing_tolerance=2)
                detected_peak_indices = grid_result.uw_positions
                detected_uw_start_samples = [int(peak_index - differential_peak_to_start_offset)
                                             for peak_index in detected_peak_indices]
                detected_uw_start_bits = [int(round(start_sample / sps) * bits_per_symbol)
                                          for start_sample in detected_uw_start_samples]

            sample["detected_uw_peak_indices"] = detected_peak_indices
            sample["detected_uw_start_samples"] = detected_uw_start_samples
            sample["detected_uw_start_bits"] = detected_uw_start_bits
            correlation_results.append(
                {
                    "snr_db": snr_db,
                    "signal_index": signal_index,
                    "n_uw": sample["n_uw"],
                    "uw_start_bits": sample["uw_start_bits"],
                    "detected_uw_peak_indices": detected_peak_indices,
                    "detected_uw_start_samples": detected_uw_start_samples,
                    "detected_uw_start_bits": detected_uw_start_bits,
                    "grid_score": 0.0 if grid_result is None else grid_result.score,
                    "correlation": correlation,
                }
            )

    return correlation_results


def estimate_frequency_phase_offsets_and_correct(dataset: dict[float, dict[int, dict[str, Any]]],
                                                 dataset_config: DatasetConfig) -> list[dict[str, Any]]:
    cpfsk_uw = build_cpfsk_uw(dataset_config)
    frequency_estimator = FrequencyOffsetEstimator(method_type="differential_data_aided", reference_signal=cpfsk_uw)

    estimation_results = []
    for snr_db, signals_by_index in dataset.items():
        for signal_index, sample in signals_by_index.items():
            detected_uw_start_samples = sample.get("detected_uw_start_samples", [])

            if detected_uw_start_samples:
                per_uw_frequency_offsets_hz = []
                for start_sample in detected_uw_start_samples:
                    uw_segment = extract_segment(
                        signal=sample["rx_signal"],
                        start_sample=start_sample,
                        segment_len=len(cpfsk_uw),
                    )
                    per_uw_frequency_offsets_hz.append(
                        frequency_estimator.estimate(
                            signal=uw_segment,
                            sample_rate=dataset_config.sample_rate,
                            reference_signal=cpfsk_uw,
                        )
                    )

                estimated_frequency_offset_hz = float(np.mean(per_uw_frequency_offsets_hz))
                corrected_signal = frequency_estimator.correct_frequency_offset(
                    signal=sample["rx_signal"],
                    frequency_offset_hz=estimated_frequency_offset_hz,
                    sample_rate=dataset_config.sample_rate,
                )

                per_uw_phase_offsets_rad = []
                for start_sample in detected_uw_start_samples:
                    corrected_uw_segment = extract_segment(
                        signal=corrected_signal,
                        start_sample=start_sample,
                        segment_len=len(cpfsk_uw),
                    )
                    per_uw_phase_offsets_rad.append(
                        estimate_phase_offset(
                            rx_uw_segment=corrected_uw_segment,
                            cpfsk_uw=cpfsk_uw,
                        )
                    )

                estimated_phase_offset_rad = circular_mean(per_uw_phase_offsets_rad)
            else:
                per_uw_frequency_offsets_hz = []
                estimated_frequency_offset_hz = float("nan")
                corrected_signal = None
                per_uw_phase_offsets_rad = []
                estimated_phase_offset_rad = float("nan")

            sample["per_uw_frequency_offsets_hz"] = per_uw_frequency_offsets_hz
            sample["estimated_frequency_offset_hz"] = estimated_frequency_offset_hz
            sample["corrected_signal"] = corrected_signal
            sample["per_uw_phase_offsets_rad"] = per_uw_phase_offsets_rad
            sample["estimated_phase_offset_rad"] = estimated_phase_offset_rad

            estimation_results.append(
                {
                    "snr_db": snr_db,
                    "signal_index": signal_index,
                    "n_uw": sample["n_uw"],
                    "uw_start_bits": sample["uw_start_bits"],
                    "detected_uw_start_bits": sample.get("detected_uw_start_bits", []),
                    "true_frequency_offset_hz": sample["frequency_offset"],
                    "estimated_frequency_offset_hz": estimated_frequency_offset_hz,
                    "per_uw_frequency_offsets_hz": per_uw_frequency_offsets_hz,
                    "true_phase_offset_rad": sample["phase_offset"],
                    "estimated_phase_offset_rad": estimated_phase_offset_rad,
                    "per_uw_phase_offsets_rad": per_uw_phase_offsets_rad,
                    "corrected_signal": corrected_signal,
                }
            )

    return estimation_results


def demodulate_corrected_signals_semi_coherent(dataset: dict[float, dict[int, dict[str, Any]]],
                                               dataset_config: DatasetConfig) -> list[dict[str, Any]]:

    demodulator_config = build_demodulator_config()
    demodulator = Demodulator(demodulator_config)

    sps = int(round(dataset_config.sample_rate * dataset_config.symbol_time))
    bits_per_symbol = int(np.log2(dataset_config.constellation_order))

    demodulation_results = []
    for snr_db, signals_by_index in dataset.items():
        for signal_index, sample in signals_by_index.items():
            corrected_signal = sample.get("corrected_signal")
            detected_uw_start_samples = sample.get("detected_uw_start_samples", [])
            detected_uw_start_bits = sample.get("detected_uw_start_bits", [])
            per_uw_phase_offsets_rad = sample.get("per_uw_phase_offsets_rad", [])

            if corrected_signal is None or not detected_uw_start_samples:
                aligned_signal = None
                rx_bits = np.array([], dtype=np.uint8)
                tx_bits_for_ber = np.array([], dtype=np.uint8)
                pll_trace = {}
                ber = float("nan")
                demod_start_sample = None
                demod_start_bit = None
                initial_phase_rad = float("nan")

            else:
                demod_start_sample = int(detected_uw_start_samples[0])
                demod_start_bit = int(detected_uw_start_bits[0])
                initial_phase_rad = float(per_uw_phase_offsets_rad[0]) if per_uw_phase_offsets_rad else 0.0

                demod_start_sample = max(demod_start_sample, 0)
                demod_start_bit = max(demod_start_bit, 0)
                aligned_signal = corrected_signal[demod_start_sample:]

                rx_bits, pll_trace = demodulator.semi_coherent_demodulate(
                    rx_signal=aligned_signal,
                    symbol_time=dataset_config.symbol_time,
                    sample_rate=dataset_config.sample_rate,
                    frequency_sensitivity=dataset_config.h,
                    initial_phase=initil_phase_rad,
                    initial_frequency_hz=0.0,
                    return_trace=True,
                )

                tx_bits_for_ber = sample["tx_bits"][demod_start_bit:]
                n_compare_bits = min(len(rx_bits), len(tx_bits_for_ber))
                n_compare_bits = (n_compare_bits // bits_per_symbol) * bits_per_symbol

                if n_compare_bits == 0:
                    ber = float("nan")
                else:
                    ber = float(np.mean(rx_bits[:n_compare_bits] != tx_bits_for_ber[:n_compare_bits]))

                rx_bits = rx_bits[:n_compare_bits]
                tx_bits_for_ber = tx_bits_for_ber[:n_compare_bits]

            sample["semi_coherent_aligned_signal"] = aligned_signal
            sample["semi_coherent_rx_bits"] = rx_bits
            sample["semi_coherent_tx_bits_for_ber"] = tx_bits_for_ber
            sample["semi_coherent_pll_trace"] = pll_trace
            sample["semi_coherent_ber"] = ber
            sample["semi_coherent_demod_start_sample"] = demod_start_sample
            sample["semi_coherent_demod_start_bit"] = demod_start_bit
            sample["semi_coherent_initial_phase_rad"] = initial_phase_rad

            demodulation_results.append(
                {
                    "snr_db": snr_db,
                    "signal_index": signal_index,
                    "n_uw": sample["n_uw"],
                    "uw_start_bits": sample["uw_start_bits"],
                    "detected_uw_start_bits": detected_uw_start_bits,
                    "demod_start_sample": demod_start_sample,
                    "demod_start_bit": demod_start_bit,
                    "initial_phase_rad": initial_phase_rad,
                    "ber": ber,
                    "n_bits": int(len(rx_bits)),
                    "pll_trace": pll_trace,
                    "rx_bits": rx_bits,
                    "tx_bits_for_ber": tx_bits_for_ber,
                }
            )

    return demodulation_results


def main():
    # build dataset
    dataset_config = build_dataset_config()
    dataset = Dataset(dataset_config).generate_dataset()

    # detection
    correlation_results = compute_uw_correlations(dataset, dataset_config)

    # frequency and phase offsets estimation and then corrections
    estimation_results = estimate_frequency_phase_offsets_and_correct(dataset, dataset_config)

    # semi coherent demodulation
    demodulation_results = demodulate_corrected_signals_semi_coherent(dataset, dataset_config)


if __name__ == "__main__":
    main()
