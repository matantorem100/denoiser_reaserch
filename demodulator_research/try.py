from pathlib import Path
import os
import sys
from typing import Any

import numpy as np

EPS = 1e-12

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-cache"))

import matplotlib.pyplot as plt

from demodulator_research.find_grid.find_grid import find_best_uw_grid
from demodulator_research.uw_detection.uw_detection import UwDetector
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.dataset_creation import Dataset, DatasetConfig
from synthetic_dataset_creation.modulator.modulator import Modulator, ModulatorConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig


def force_length(signal: np.ndarray, n_samples: int) -> np.ndarray:
    signal = np.asarray(signal)
    if len(signal) >= n_samples:
        return signal[:n_samples]
    return np.pad(signal, (0, n_samples - len(signal)))


def force_sample_lengths(sample: dict[str, Any], total_signal_samples: int, tx_signal_samples: int) -> None:
    sample["rx_signal"] = force_length(sample["rx_signal"], total_signal_samples)
    sample["channel_noise"] = force_length(sample["channel_noise"], total_signal_samples)
    sample["tx_signal"] = force_length(sample["tx_signal"], tx_signal_samples)
    sample["total_signal_samples"] = int(total_signal_samples)


def build_noise_or_signal_dataset(
        snr_db_values: list[float] | tuple[float, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20),
        n_samples_per_snr: int = 10,
        signal_probability: float = 0.5,
        min_frequency_offset_hz: float = -500.0,
        max_frequency_offset_hz: float = 500.0,
        min_phase_offset_rad: float = 0.0,
        max_phase_offset_rad: float = 2.0 * np.pi,
        random_seed: int | None = None,
) -> dict[float, dict[int, dict[str, Any]]]:
    sample_rate = 10000
    symbol_time = 1e-3
    constellation_order = 4
    bits_per_symbol = int(np.log2(constellation_order))
    sps = int(round(sample_rate * symbol_time))

    total_signal_duration_sec = 0.600
    uw_spacing_sec = 0.100
    leading_noise_duration_sec = 0.100
    n_uw = 5

    total_signal_samples = int(round(total_signal_duration_sec * sample_rate))
    leading_noise_samples = int(round(leading_noise_duration_sec * sample_rate))
    uw_spacing_samples = int(round(uw_spacing_sec * sample_rate))
    uw_spacing_bits = int(round(uw_spacing_samples / sps)) * bits_per_symbol

    n_symbols_after_noise = int(round((total_signal_samples - leading_noise_samples) / sps))
    n_bits_to_transmit = n_symbols_after_noise * bits_per_symbol

    uw_bits = [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0]

    dataset = {}
    rng = np.random.default_rng(random_seed)

    for snr_db in snr_db_values:
        snr_db = float(snr_db)
        dataset[snr_db] = {}

        for sample_idx in range(int(n_samples_per_snr)):
            has_signal = bool(rng.random() < signal_probability)
            n_uw_value = n_uw if has_signal else 0
            frequency_offset_hz = float(rng.uniform(min_frequency_offset_hz, max_frequency_offset_hz))
            phase_offset_rad = float(rng.uniform(min_phase_offset_rad, max_phase_offset_rad))

            config = DatasetConfig(
                sample_rate=sample_rate,
                symbol_time=symbol_time,
                n_bits_to_transmit=n_bits_to_transmit,
                const_frequency_offset=frequency_offset_hz,
                const_phase_offset=phase_offset_rad,
                h=0.5,
                pulse_shape_type="rrc",
                span_in_symbols=5,
                pulse_normalization="cpfsk",
                constellation_type="PAM",
                constellation_order=constellation_order,
                noise_type="snr",
                noise_db_values=[snr_db],
                n_signals_per_snr=1,
                uw_bits=uw_bits,
                use_uw=True,
                uw_probability=1.0,
                n_uw_values=[n_uw_value],
                uw_spacing_bits=uw_spacing_bits,
                first_uw_after_noise=True,
                leading_noise_samples=leading_noise_samples,
                noise_only_when_no_uw=True,
                random_uw_start=False,
                uw_mode="overwrite",
                random_seed=int(rng.integers(0, 2**31 - 1)),
            )

            sample = Dataset(config).generate_dataset()[snr_db][0]
            force_sample_lengths(
                sample=sample,
                total_signal_samples=total_signal_samples,
                tx_signal_samples=total_signal_samples - leading_noise_samples,
            )
            sample["uw_spacing_samples"] = int(uw_spacing_samples)
            sample["uw_spacing_bits"] = int(uw_spacing_bits)
            sample["total_signal_duration_sec"] = float(total_signal_duration_sec)
            dataset[snr_db][sample_idx] = sample

    return dataset


def build_cpfsk_uw_reference(
        uw_bits: list[int] | np.ndarray,
        sample_rate: float = 10000,
        symbol_time: float = 1e-3,
        h: float = 0.5,
        pulse_shape_type: str = "rrc",
        span_in_symbols: int = 5,
        pulse_normalization: str = "cpfsk",
        constellation_type: str = "PAM",
        constellation_order: int = 4,
) -> np.ndarray:
    pulse_shape_config = PulseShapeConfig(
        pulse_shape_type=pulse_shape_type,
        normalization_type=pulse_normalization,
        span_in_symbols=span_in_symbols,
    )
    constellation_config = ConstellationConfig(
        constellation_type=constellation_type,
        constellation_order=constellation_order,
    )
    modulator = Modulator(
        ModulatorConfig(
            pulse_shape_config=pulse_shape_config,
            constellation_config=constellation_config,
        )
    )
    signal, _ = modulator.modulate(
        bits=np.asarray(uw_bits, dtype=np.uint8),
        symbol_time=symbol_time,
        sample_rate=sample_rate,
        frequency_offset=0.0,
        frequency_sensitivity=h,
        uw=None,
    )
    return signal


def crop_middle_uw_reference(cpfsk_uw: np.ndarray, sps: int, n_symbols_each_side: int = 3) -> tuple[np.ndarray, int, int]:
    uw_length = len(cpfsk_uw)
    uw_crop_start_sample = int(uw_length // 2 - sps * n_symbols_each_side)
    uw_crop_stop_sample = int(uw_length // 2 + sps * n_symbols_each_side)

    if uw_crop_start_sample < 0 or uw_crop_stop_sample > uw_length:
        raise ValueError("Middle UW crop is outside the clean UW reference")

    return cpfsk_uw[uw_crop_start_sample:uw_crop_stop_sample], uw_crop_start_sample, uw_crop_stop_sample


def detect_middle_uw_differential(
        rx_signal: np.ndarray,
        uw_spacing_samples: int,
        n_uw: int | None,
        threshold: float = 0.45,
        sample_rate: float = 10000,
        symbol_time: float = 1e-3,
        h: float = 0.5,
        uw_bits: list[int] | np.ndarray | None = None,
        pulse_shape_type: str = "rrc",
        span_in_symbols: int = 5,
        pulse_normalization: str = "cpfsk",
        constellation_type: str = "PAM",
        constellation_order: int = 4,
        grid_spacing_tolerance: int = 2,
) -> dict[str, Any]:
    if uw_bits is None:
        uw_bits = [
            1, 0, 0, 1, 0, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
        ]

    sps = int(round(sample_rate * symbol_time))
    cpfsk_uw = build_cpfsk_uw_reference(
        uw_bits=uw_bits,
        sample_rate=sample_rate,
        symbol_time=symbol_time,
        h=h,
        pulse_shape_type=pulse_shape_type,
        span_in_symbols=span_in_symbols,
        pulse_normalization=pulse_normalization,
        constellation_type=constellation_type,
        constellation_order=constellation_order,
    )
    middle_uw, uw_crop_start_sample, uw_crop_stop_sample = crop_middle_uw_reference(
        cpfsk_uw=cpfsk_uw,
        sps=sps,
        n_symbols_each_side=3,
    )

    correlation = UwDetector(
        method_type="differential_correlation",
        differential_lag_samples=sps,
    ).estimate(signal=np.asarray(rx_signal, dtype=np.complex128), cpfsk_uw=middle_uw)

    grid = find_best_uw_grid(
        correlation=correlation,
        uw_spacing=int(uw_spacing_samples),
        threshold=float(threshold),
        n_uw=n_uw if n_uw and n_uw > 0 else None,
        spacing_tolerance=int(grid_spacing_tolerance),
        min_n_uw=1,
    )

    peak_to_start_offset = uw_crop_start_sample + len(middle_uw) - sps - 1
    detected_uw_start_samples = [int(peak - peak_to_start_offset) for peak in grid.uw_positions]

    return {
        "correlation": correlation,
        "uw_reference": middle_uw,
        "uw_crop_start_sample": int(uw_crop_start_sample),
        "uw_crop_stop_sample": int(uw_crop_stop_sample),
        "peak_to_start_offset": int(peak_to_start_offset),
        "uw_peak_indices": grid.uw_positions,
        "uw_peak_scores": grid.peak_values,
        "uw_start_samples": detected_uw_start_samples,
        "grid_score": grid.score,
    }



def differential_data_aided_estimation(signal: np.ndarray, reference_signal: np.ndarray, sample_rate: float) -> float:

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


def extract_fixed_length(signal: np.ndarray, start_sample: int, n_samples: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    start_sample = int(start_sample)

    if start_sample < 0:
        segment = signal[:max(0, start_sample + n_samples)]
        segment = np.pad(segment, (abs(start_sample), 0))
    else:
        segment = signal[start_sample:start_sample + n_samples]

    if len(segment) < n_samples:
        segment = np.pad(segment, (0, n_samples - len(segment)))

    return segment[:n_samples]


def estimate_frequency_offset_from_middle_uw(
        rx_signal: np.ndarray,
        uw_start_samples: list[int],
        sample_rate: float = 10000,
        symbol_time: float = 1e-3,
        h: float = 0.5,
        uw_bits: list[int] | np.ndarray | None = None,
        pulse_shape_type: str = "rrc",
        span_in_symbols: int = 5,
        pulse_normalization: str = "cpfsk",
        constellation_type: str = "PAM",
        constellation_order: int = 4,
) -> dict[str, Any]:
    if uw_bits is None:
        uw_bits = [
            1, 0, 0, 1, 0, 1, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
        ]

    sps = int(round(sample_rate * symbol_time))
    cpfsk_uw = build_cpfsk_uw_reference(
        uw_bits=uw_bits,
        sample_rate=sample_rate,
        symbol_time=symbol_time,
        h=h,
        pulse_shape_type=pulse_shape_type,
        span_in_symbols=span_in_symbols,
        pulse_normalization=pulse_normalization,
        constellation_type=constellation_type,
        constellation_order=constellation_order,
    )
    middle_uw, uw_crop_start_sample, uw_crop_stop_sample = crop_middle_uw_reference(
        cpfsk_uw=cpfsk_uw,
        sps=sps,
        n_symbols_each_side=3,
    )

    per_uw_cfo_hz = []
    for uw_start_sample in uw_start_samples:
        rx_middle_uw = extract_fixed_length(
            signal=rx_signal,
            start_sample=int(uw_start_sample) + uw_crop_start_sample,
            n_samples=len(middle_uw),
        )
        per_uw_cfo_hz.append(
            differential_data_aided_estimation(
                signal=rx_middle_uw,
                reference_signal=middle_uw,
                sample_rate=sample_rate,
            )
        )

    finite_estimates = [estimate for estimate in per_uw_cfo_hz if np.isfinite(estimate)]
    mean_cfo_hz = float(np.mean(finite_estimates)) if finite_estimates else float("nan")

    return {
        "mean_cfo_hz": mean_cfo_hz,
        "per_uw_cfo_hz": per_uw_cfo_hz,
        "uw_reference": middle_uw,
        "uw_crop_start_sample": int(uw_crop_start_sample),
        "uw_crop_stop_sample": int(uw_crop_stop_sample),
    }


def exact_grid_match(detected_starts: list[int], true_starts: list[int], tolerance_samples: int = 2) -> bool:
    if len(detected_starts) != len(true_starts):
        return False
    return all(abs(int(detected) - int(true)) <= tolerance_samples
               for detected, true in zip(detected_starts, true_starts))


def evaluate_dataset(dataset: dict[float, dict[int, dict[str, Any]]],
                     detection_threshold: float = 0.53,
                     grid_tolerance_samples: int = 2) -> dict[float, dict[str, float]]:
    stats = {}

    for snr_db, samples in dataset.items():
        n_signal = 0
        n_noise = 0
        n_exact_grid = 0
        n_missed_detection = 0
        n_false_alarm = 0
        cfo_errors = []
        cfo_abs_errors = []

        for sample in samples.values():
            has_signal = sample["n_uw"] > 0
            if has_signal:
                n_signal += 1
            else:
                n_noise += 1

            detection = detect_middle_uw_differential(
                rx_signal=sample["rx_signal"],
                uw_spacing_samples=sample["uw_spacing_samples"],
                n_uw=sample["n_uw"] if has_signal else None,
                threshold=detection_threshold,
                grid_spacing_tolerance=grid_tolerance_samples,
            )
            detected_any = len(detection["uw_start_samples"]) > 0

            if has_signal:
                exact = exact_grid_match(
                    detected_starts=detection["uw_start_samples"],
                    true_starts=sample["uw_start_samples"],
                    tolerance_samples=grid_tolerance_samples,
                )
                n_exact_grid += int(exact)
                n_missed_detection += int(not exact)

                if exact:
                    cfo_estimation = estimate_frequency_offset_from_middle_uw(
                        rx_signal=sample["rx_signal"],
                        uw_start_samples=detection["uw_start_samples"],
                    )
                    cfo_error = cfo_estimation["mean_cfo_hz"] - float(sample["frequency_offset"])
                    cfo_errors.append(float(cfo_error))
                    cfo_abs_errors.append(abs(float(cfo_error)))
            else:
                n_false_alarm += int(detected_any)

        stats[float(snr_db)] = {
            "n_signal": float(n_signal),
            "n_noise": float(n_noise),
            "exact_grid_rate": float(n_exact_grid / n_signal) if n_signal else float("nan"),
            "missed_detection_rate": float(n_missed_detection / n_signal) if n_signal else float("nan"),
            "false_alarm_rate": float(n_false_alarm / n_noise) if n_noise else float("nan"),
            "mean_cfo_error_hz": float(np.mean(cfo_errors)) if cfo_errors else float("nan"),
            "mean_abs_cfo_error_hz": float(np.mean(cfo_abs_errors)) if cfo_abs_errors else float("nan"),
            "std_cfo_error_hz": float(np.std(cfo_errors)) if cfo_errors else float("nan"),
        }

    return stats


def plot_dataset_stats(stats: dict[float, dict[str, float]]) -> None:
    snr_values = np.asarray(sorted(stats.keys()), dtype=float)
    mean_cfo_error = np.asarray([stats[snr]["mean_cfo_error_hz"] for snr in snr_values], dtype=float)
    mean_abs_cfo_error = np.asarray([stats[snr]["mean_abs_cfo_error_hz"] for snr in snr_values], dtype=float)
    exact_grid_rate = np.asarray([stats[snr]["exact_grid_rate"] for snr in snr_values], dtype=float)
    false_alarm_rate = np.asarray([stats[snr]["false_alarm_rate"] for snr in snr_values], dtype=float)
    missed_detection_rate = np.asarray([stats[snr]["missed_detection_rate"] for snr in snr_values], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    axes[0].plot(snr_values, mean_cfo_error, marker="o", label="mean signed CFO error")
    axes[0].plot(snr_values, mean_abs_cfo_error, marker="o", label="mean abs CFO error")
    axes[0].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_title("Frequency Offset Estimation vs SNR")
    axes[0].set_ylabel("error [Hz]")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(snr_values, 100.0 * exact_grid_rate, marker="o")
    axes[1].set_title("Exact Grid Detection on Signal Samples")
    axes[1].set_xlabel("SNR [dB]")
    axes[1].set_ylabel("exact grid [%]")
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True)

    axes[2].plot(snr_values, 100.0 * false_alarm_rate, marker="o", label="FA on noise")
    axes[2].plot(snr_values, 100.0 * missed_detection_rate, marker="o", label="MD on signal")
    axes[2].set_title("False Alarm and Missed Detection vs SNR")
    axes[2].set_xlabel("SNR [dB]")
    axes[2].set_ylabel("rate [%]")
    axes[2].set_ylim(-5, 105)
    axes[2].grid(True)
    axes[2].legend()

    plt.show()


if __name__ == "__main__":
    dataset = build_noise_or_signal_dataset(n_samples_per_snr=100)
    stats = evaluate_dataset(dataset)

    for snr_db in sorted(stats):
        row = stats[snr_db]
        print(
            f"SNR={snr_db:g} dB | "
            f"signal={int(row['n_signal'])}, noise={int(row['n_noise'])} | "
            f"exact={100 * row['exact_grid_rate']:.1f}% | "
            f"FA={100 * row['false_alarm_rate']:.1f}% | "
            f"MD={100 * row['missed_detection_rate']:.1f}% | "
            f"mean_abs_cfo_error={row['mean_abs_cfo_error_hz']:.2f} Hz"
        )

    plot_dataset_stats(stats)
