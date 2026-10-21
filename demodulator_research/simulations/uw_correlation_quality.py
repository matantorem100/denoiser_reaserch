from dataclasses import dataclass
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import os
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal as sp


from demodulator_research.find_grid.find_grid import find_best_uw_grid
from demodulator_research.uw_detection.uw_detection import UwDetector
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.dataset_creation import Dataset, DatasetConfig
from synthetic_dataset_creation.modulator.modulator import Modulator, ModulatorConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig


@dataclass
class Config:
    sample_rate: float = 10_000
    symbol_time: float = 1e-3
    h: float = 0.5

    pulse_shape_type: str = "rrc"
    span_in_symbols: int = 10
    pulse_normalization: str = "cpfsk"
    constellation_type: str = "PAM"
    constellation_order: int = 4

    n_bits: int = 1200
    n_signals_per_condition: int = 20
    snr_db_values: tuple[float, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    frequency_offsets_hz: tuple[float, ...] = (0, 0)

    uw_bits: tuple[int, ...] = (
        1, 0, 0, 1, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    )
    uw_spacing_bits: int = 500
    leading_noise_samples: int = 1000

    detection_threshold: float = 0.45
    grid_tolerance_samples: int = 20
    detection_tolerance_samples: int = 10
    min_peak_distance_samples: int = 10
    min_n_uw_for_detection: int = 3
    grid_first_peak_weight: float = 1.0
    random_seed: int = 0

    n_context_symbols_before_crop: int = 6
    n_context_symbols_after_crop: int = 6
    n_random_context_trials: int = 1
    show_plots: bool = False
    save_plots: bool = True
    plot_output_dir: str = "demodulator_research/simulations/uw_correlation_quality_plots"

    simulate_noninteger_sps: bool = False
    physical_sample_rate: float = 10_000
    symbol_rate: float = 2400
    integer_reference_sps: int = 4
    resample_noninteger_back_to_integer: bool = False


def build_modulator(config: Config) -> Modulator:
    pulse_shape_config = PulseShapeConfig(
        pulse_shape_type=config.pulse_shape_type,
        normalization_type=config.pulse_normalization,
        span_in_symbols=config.span_in_symbols,
    )
    constellation_config = ConstellationConfig(
        constellation_type=config.constellation_type,
        constellation_order=config.constellation_order,
    )
    return Modulator(
        ModulatorConfig(
            pulse_shape_config=pulse_shape_config,
            constellation_config=constellation_config,
        )
    )


def rational_resampling_factors(input_sample_rate: float, output_sample_rate: float,
                                max_denominator: int = 100000) -> tuple[int, int]:
    ratio = Fraction(float(output_sample_rate) / float(input_sample_rate)).limit_denominator(int(max_denominator))
    return int(ratio.numerator), int(ratio.denominator)


def resample_complex_signal(signal: np.ndarray, input_sample_rate: float, output_sample_rate: float) -> np.ndarray:
    if np.isclose(float(input_sample_rate), float(output_sample_rate)):
        return np.asarray(signal, dtype=np.complex64).copy()

    up, down = rational_resampling_factors(input_sample_rate, output_sample_rate)
    return sp.resample_poly(np.asarray(signal, dtype=np.complex64), up=up, down=down).astype(np.complex64)


def integer_reference_config(config: Config) -> Config:
    if not bool(config.simulate_noninteger_sps):
        return config

    symbol_time = 1.0 / float(config.symbol_rate)
    integer_sample_rate = float(config.integer_reference_sps) * float(config.symbol_rate)
    leading_noise_seconds = float(config.leading_noise_samples) / float(config.physical_sample_rate)
    return replace(
        config,
        sample_rate=integer_sample_rate,
        symbol_time=symbol_time,
        leading_noise_samples=int(round(leading_noise_seconds * integer_sample_rate)),
    )


def detection_config(config: Config) -> Config:
    if not bool(config.simulate_noninteger_sps):
        return config

    # The detector still uses the integer-SPS UW reference. If the samples were
    # not resampled back to the integer grid, this intentionally creates the
    # mismatch we want to test.
    return integer_reference_config(config)


def apply_noninteger_sps_simulation_to_sample(sample: dict[str, Any], config: Config,
                                              generation_config: Config) -> dict[str, Any]:
    if not bool(config.simulate_noninteger_sps):
        return sample

    generated_sample_rate = float(generation_config.sample_rate)
    physical_sample_rate = float(config.physical_sample_rate)
    integer_sample_rate = float(config.integer_reference_sps) * float(config.symbol_rate)

    sample = dict(sample)
    physical_signal = resample_complex_signal(
        signal=sample["rx_signal"],
        input_sample_rate=generated_sample_rate,
        output_sample_rate=physical_sample_rate,
    )
    physical_uw_starts = [
        int(round(int(start) * physical_sample_rate / generated_sample_rate))
        for start in sample["uw_start_samples"]
    ]
    physical_leading_noise = int(round(int(sample["leading_noise_samples"]) * physical_sample_rate / generated_sample_rate))

    if bool(config.resample_noninteger_back_to_integer):
        receiver_signal = resample_complex_signal(
            signal=physical_signal,
            input_sample_rate=physical_sample_rate,
            output_sample_rate=integer_sample_rate,
        )
        receiver_uw_starts = [
            int(round(int(start) * integer_sample_rate / physical_sample_rate))
            for start in physical_uw_starts
        ]
        receiver_leading_noise = int(round(physical_leading_noise * integer_sample_rate / physical_sample_rate))
        receiver_sample_rate = integer_sample_rate
    else:
        receiver_signal = physical_signal
        receiver_uw_starts = physical_uw_starts
        receiver_leading_noise = physical_leading_noise
        receiver_sample_rate = physical_sample_rate

    sample["rx_signal"] = np.asarray(receiver_signal, dtype=np.complex64)
    sample["uw_start_samples"] = receiver_uw_starts
    sample["leading_noise_samples"] = receiver_leading_noise
    sample["noninteger_sps"] = {
        "generated_sample_rate": generated_sample_rate,
        "physical_sample_rate": physical_sample_rate,
        "receiver_sample_rate": receiver_sample_rate,
        "symbol_rate": float(config.symbol_rate),
        "physical_sps": physical_sample_rate / float(config.symbol_rate),
        "integer_reference_sps": int(config.integer_reference_sps),
        "resample_back_to_integer": bool(config.resample_noninteger_back_to_integer),
    }
    return sample


def build_cpfsk_uw(config: Config) -> np.ndarray:
    return modulate_bits(detection_config(config), np.asarray(config.uw_bits, dtype=np.uint8))


def modulate_bits(config: Config, bits: np.ndarray) -> np.ndarray:
    signal, _ = build_modulator(config).modulate(
        bits=bits,
        symbol_time=config.symbol_time,
        sample_rate=config.sample_rate,
        frequency_offset=0.0,
        frequency_sensitivity=config.h,
        uw=None,
    )
    return signal


def create_dataset(config: Config) -> dict[tuple[float, float], dict[int, dict[str, Any]]]:
    """
    Create samples for every (SNR, CFO) condition.

    Each condition contains signals with either:
        n_uw = 0 -> noise only
        n_uw = 3 -> noise prefix, UW, payload, UW, payload, UW

    UWs are spaced by config.uw_spacing_bits.
    """
    rng = np.random.default_rng(config.random_seed)
    dataset = {}
    generation_config = integer_reference_config(config)

    for snr_db in config.snr_db_values:
        for frequency_offset_hz in config.frequency_offsets_hz:
            condition_samples: dict[int, dict[str, Any]] = {}
            next_index = 0

            for n_uw in (0, 3):
                dataset_config = DatasetConfig(
                    sample_rate=generation_config.sample_rate,
                    symbol_time=generation_config.symbol_time,
                    n_bits_to_transmit=config.n_bits,
                    const_frequency_offset=float(frequency_offset_hz),
                    min_phase_offset=0.0,
                    max_phase_offset=2.0 * np.pi,
                    const_phase_offset=None,
                    h=config.h,
                    pulse_shape_type=config.pulse_shape_type,
                    span_in_symbols=config.span_in_symbols,
                    pulse_normalization=config.pulse_normalization,
                    constellation_type=config.constellation_type,
                    constellation_order=config.constellation_order,
                    noise_type="eb_to_n0",
                    noise_db_values=[float(snr_db)],
                    n_signals_per_snr=config.n_signals_per_condition,
                    uw_bits=list(config.uw_bits),
                    use_uw=True,
                    uw_probability=1.0,
                    n_uw_values=[int(n_uw)],
                    uw_spacing_bits=config.uw_spacing_bits,
                    first_uw_after_noise=True,
                    leading_noise_samples=generation_config.leading_noise_samples,
                    noise_only_when_no_uw=True,
                    random_uw_start=False,
                    uw_mode="overwrite",
                    random_seed=int(rng.integers(0, 2**31 - 1)),
                )
                generated = Dataset(dataset_config).generate_dataset()[float(snr_db)]
                for sample in generated.values():
                    condition_samples[next_index] = apply_noninteger_sps_simulation_to_sample(
                        sample=sample,
                        config=config,
                        generation_config=generation_config,
                    )
                    next_index += 1

            dataset[(float(snr_db), float(frequency_offset_hz))] = condition_samples

    return dataset


def crop_stable_uw_middle(config: Config, cpfsk_uw: np.ndarray) -> tuple[np.ndarray, int, int]:
    sps = int(round(config.sample_rate * config.symbol_time))
    uw_length = len(cpfsk_uw)
    uw_crop_start_sample = int(uw_length // 2 - sps * 3)
    uw_crop_stop_sample = int(uw_length // 2 + sps * 3)

    if uw_crop_start_sample < 0 or uw_crop_stop_sample > uw_length:
        raise ValueError("UW middle crop is outside the clean UW signal")

    return cpfsk_uw[uw_crop_start_sample:uw_crop_stop_sample], uw_crop_start_sample, uw_crop_stop_sample


def crop_symbol_range(config: Config, cpfsk_uw: np.ndarray) -> tuple[int, int]:
    sps = int(round(config.sample_rate * config.symbol_time))
    bits_per_symbol = int(np.log2(config.constellation_order))
    n_uw_symbols = len(config.uw_bits) // bits_per_symbol
    _, uw_crop_start_sample, uw_crop_stop_sample = crop_stable_uw_middle(config, cpfsk_uw)

    n_crop_symbols = (uw_crop_stop_sample - uw_crop_start_sample) // sps
    crop_start_symbol = n_uw_symbols // 2 - n_crop_symbols // 2
    crop_stop_symbol = crop_start_symbol + n_crop_symbols

    if crop_start_symbol < 0 or crop_stop_symbol > n_uw_symbols:
        raise ValueError("UW crop does not fit inside the UW symbols")

    return crop_start_symbol, crop_stop_symbol


def random_context_middle_correlation(
        rx_signal: np.ndarray,
        config: Config,
        cpfsk_uw: np.ndarray,
        rng: np.random.Generator,
) -> np.ndarray:
    sps = int(round(config.sample_rate * config.symbol_time))
    bits_per_symbol = int(np.log2(config.constellation_order))
    uw_bits = np.asarray(config.uw_bits, dtype=np.uint8)
    _, uw_crop_start_sample, uw_crop_stop_sample = crop_stable_uw_middle(config, cpfsk_uw)

    crop_start_symbol, crop_stop_symbol = crop_symbol_range(config, cpfsk_uw)
    crop_start_bit = crop_start_symbol * bits_per_symbol
    crop_stop_bit = crop_stop_symbol * bits_per_symbol
    fixed_crop_bits = uw_bits[crop_start_bit:crop_stop_bit]

    crop_start_offset_samples = uw_crop_start_sample - crop_start_symbol * sps
    reference_crop_start = config.n_context_symbols_before_crop * sps + crop_start_offset_samples
    reference_crop_len = uw_crop_stop_sample - uw_crop_start_sample

    best_score = -np.inf
    best_correlation = None

    for _ in range(config.n_random_context_trials):
        before_bits = rng.integers(
            0,
            2,
            size=config.n_context_symbols_before_crop * bits_per_symbol,
            dtype=np.uint8,
        )
        after_bits = rng.integers(
            0,
            2,
            size=config.n_context_symbols_after_crop * bits_per_symbol,
            dtype=np.uint8,
        )
        reference_bits = np.concatenate((before_bits, fixed_crop_bits, after_bits))
        reference_signal = modulate_bits(config, reference_bits)
        reference_middle = reference_signal[reference_crop_start:reference_crop_start + reference_crop_len]

        correlation = UwDetector(
            "differential_correlation",
            differential_lag_samples=sps,
        ).estimate(rx_signal, reference_middle)
        score = float(np.max(np.abs(correlation)))

        if score > best_score:
            best_score = score
            best_correlation = correlation

    return best_correlation


def run_all_correlation_methods(
        rx_signal: np.ndarray,
        config: Config,
        cpfsk_uw: np.ndarray,
        rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Run every UW detector correlation method.
    """
    sps = int(round(config.sample_rate * config.symbol_time))
    cpfsk_uw_middle, _, _ = crop_stable_uw_middle(config, cpfsk_uw)

    return {
        "complex_correlation": UwDetector("complex_correlation").estimate(rx_signal, cpfsk_uw),
        "regular_correlation": UwDetector("regular_correlation").estimate(rx_signal, cpfsk_uw),
        "differential_correlation": UwDetector(
            "differential_correlation",
            differential_lag_samples=sps,
        ).estimate(rx_signal, cpfsk_uw),
        "middle_differential_correlation": UwDetector(
            "differential_correlation",
            differential_lag_samples=sps,
        ).estimate(rx_signal, cpfsk_uw_middle),
        "random_context_middle_differential_correlation": random_context_middle_correlation(
            rx_signal=rx_signal,
            config=config,
            cpfsk_uw=cpfsk_uw,
            rng=rng,
        ),
    }


def correlation_detection_score(correlation: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.abs(np.asarray(correlation, dtype=np.complex128)), nan=0.0, posinf=0.0, neginf=0.0)


def true_peak_offset(method: str, config: Config, cpfsk_uw: np.ndarray) -> int:
    sps = int(round(config.sample_rate * config.symbol_time))

    if method == "complex_correlation":
        return len(cpfsk_uw) - 1
    if method == "regular_correlation":
        return len(cpfsk_uw) - 2
    if method == "differential_correlation":
        return len(cpfsk_uw) - sps - 1
    if method == "middle_differential_correlation":
        cpfsk_uw_middle, uw_crop_start_sample, _ = crop_stable_uw_middle(config, cpfsk_uw)
        return uw_crop_start_sample + len(cpfsk_uw_middle) - sps - 1
    if method == "random_context_middle_differential_correlation":
        cpfsk_uw_middle, uw_crop_start_sample, _ = crop_stable_uw_middle(config, cpfsk_uw)
        return uw_crop_start_sample + len(cpfsk_uw_middle) - sps - 1

    raise ValueError(f"Unknown method: {method}")


def calibrate_true_peak_offsets(config: Config, cpfsk_uw: np.ndarray) -> dict[str, int]:
    """
    Calibrate each method's peak offset in same-mode correlation coordinates.

    The older hand-written formulas are useful intuition, but the safest way to
    evaluate FA/MD is to measure where the current implementation actually puts
    the peak for a clean UW placed at a known sample.
    """
    sps = int(round(config.sample_rate * config.symbol_time))
    rng = np.random.default_rng(config.random_seed + 1)
    start_sample = max(len(cpfsk_uw), 8 * sps)
    clean_signal = np.zeros(start_sample + len(cpfsk_uw) + max(len(cpfsk_uw), 8 * sps), dtype=np.complex64)
    clean_signal[start_sample:start_sample + len(cpfsk_uw)] = np.asarray(cpfsk_uw, dtype=np.complex64)

    offsets = {}
    correlations = run_all_correlation_methods(
        rx_signal=clean_signal,
        config=config,
        cpfsk_uw=cpfsk_uw,
        rng=rng,
    )
    for method, correlation in correlations.items():
        score = correlation_detection_score(correlation)
        offsets[method] = int(np.argmax(score)) - int(start_sample)
    offsets["random_context_middle_differential_correlation"] = offsets["middle_differential_correlation"]

    return offsets


def expected_uw_spacing_samples(config: Config) -> int:
    sps = int(round(config.sample_rate * config.symbol_time))
    bits_per_symbol = int(np.log2(config.constellation_order))
    return int(round(config.uw_spacing_bits / bits_per_symbol * sps))


def detect_grid_from_correlation(correlation: np.ndarray, config: Config) -> tuple[bool, list[int], list[float]]:
    score = correlation_detection_score(correlation)
    grid = find_best_uw_grid(
        correlation=score,
        uw_spacing=expected_uw_spacing_samples(config),
        threshold=float(config.detection_threshold),
        n_uw=None,
        spacing_tolerance=int(config.grid_tolerance_samples),
        min_peak_distance=int(config.min_peak_distance_samples),
        min_n_uw=int(config.min_n_uw_for_detection),
        first_peak_weight=float(config.grid_first_peak_weight),
    )
    detected = len(grid.uw_positions) >= int(config.min_n_uw_for_detection)
    return bool(detected), [int(position) for position in grid.uw_positions], [float(value) for value in grid.peak_values]


def correct_detection_by_peak_grid(
        detected_peaks: list[int],
        true_uw_starts: list[int],
        peak_offset: int,
        config: Config,
) -> bool:
    expected_peaks = [int(start) + int(peak_offset) for start in true_uw_starts]
    unused_expected_peaks = expected_peaks.copy()
    n_matches = 0

    for detected_peak in detected_peaks:
        if not unused_expected_peaks:
            break
        errors = [abs(int(detected_peak) - int(expected_peak)) for expected_peak in unused_expected_peaks]
        best_index = int(np.argmin(errors))
        if errors[best_index] <= int(config.detection_tolerance_samples):
            unused_expected_peaks.pop(best_index)
            n_matches += 1

    return n_matches >= int(config.min_n_uw_for_detection)


def evaluate_fa_md_vs_snr(
        config: Config,
        dataset: dict[tuple[float, float], dict[int, dict[str, Any]]],
        cpfsk_uw: np.ndarray,
) -> dict[str, dict[float, dict[str, float]]]:
    rng = np.random.default_rng(config.random_seed)
    peak_offsets = calibrate_true_peak_offsets(config, cpfsk_uw)
    method_names = list(peak_offsets)
    stats = {
        method: {
            float(snr_db): {
                "n_noise": 0.0,
                "n_signal": 0.0,
                "false_alarm": 0.0,
                "missed_detection": 0.0,
            }
            for snr_db in config.snr_db_values
        }
        for method in method_names
    }

    print("calibrated peak offsets:", peak_offsets, flush=True)

    for (snr_db, frequency_offset_hz), samples in dataset.items():
        print(f"evaluating SNR={snr_db:g} dB CFO={frequency_offset_hz:g} Hz", flush=True)
        for sample in samples.values():
            has_signal = bool(sample["n_uw"] > 0)
            correlations = run_all_correlation_methods(
                rx_signal=sample["rx_signal"],
                config=config,
                cpfsk_uw=cpfsk_uw,
                rng=rng,
            )

            for method, correlation in correlations.items():
                method_stats = stats[method][float(snr_db)]
                detected, peaks, _peak_values = detect_grid_from_correlation(correlation, config)

                if not has_signal:
                    method_stats["n_noise"] += 1.0
                    method_stats["false_alarm"] += float(detected)
                    continue

                method_stats["n_signal"] += 1.0
                correct_detection = detected and correct_detection_by_peak_grid(
                    detected_peaks=peaks,
                    true_uw_starts=sample["uw_start_samples"],
                    peak_offset=peak_offsets[method],
                    config=config,
                )
                method_stats["missed_detection"] += float(not correct_detection)

    for method in method_names:
        for snr_db in config.snr_db_values:
            row = stats[method][float(snr_db)]
            row["false_alarm_rate"] = row["false_alarm"] / row["n_noise"] if row["n_noise"] else float("nan")
            row["missed_detection_rate"] = row["missed_detection"] / row["n_signal"] if row["n_signal"] else float("nan")

    return stats


def plot_fa_md_vs_snr_by_method(config: Config, stats: dict[str, dict[float, dict[str, float]]]) -> None:
    snrs = np.asarray(config.snr_db_values, dtype=float)

    for method, method_stats in stats.items():
        fa = np.asarray([method_stats[float(snr)]["false_alarm_rate"] for snr in snrs], dtype=float)
        md = np.asarray([method_stats[float(snr)]["missed_detection_rate"] for snr in snrs], dtype=float)

        plt.figure(figsize=(9, 5))
        plt.plot(snrs, 100.0 * fa, marker="o", linewidth=2.0, label="FA")
        plt.plot(snrs, 100.0 * md, marker="s", linewidth=2.0, label="MD")
        plt.title(f"FA / MD vs SNR | {method} | threshold={config.detection_threshold:.3f}")
        plt.xlabel("SNR [dB]")
        plt.ylabel("rate [%]")
        plt.ylim(-5, 105)
        plt.grid(True)
        plt.legend()
        finish_plot(config, f"fa_md_{method}.png")


def plot_all_methods_summary(config: Config, stats: dict[str, dict[float, dict[str, float]]]) -> None:
    snrs = np.asarray(config.snr_db_values, dtype=float)
    fig, (ax_fa, ax_md) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for method, method_stats in stats.items():
        fa = np.asarray([method_stats[float(snr)]["false_alarm_rate"] for snr in snrs], dtype=float)
        md = np.asarray([method_stats[float(snr)]["missed_detection_rate"] for snr in snrs], dtype=float)
        ax_fa.plot(snrs, 100.0 * fa, marker="o", linewidth=1.8, label=method)
        ax_md.plot(snrs, 100.0 * md, marker="s", linewidth=1.8, label=method)

    ax_fa.set_ylabel("FA [%]")
    ax_fa.set_title(f"False Alarm vs SNR | threshold={config.detection_threshold:.3f}")
    ax_fa.grid(True)
    ax_fa.legend(fontsize=8)

    ax_md.set_xlabel("SNR [dB]")
    ax_md.set_ylabel("MD [%]")
    ax_md.set_title("Missed Detection vs SNR")
    ax_md.grid(True)
    ax_md.legend(fontsize=8)
    finish_plot(config, "fa_md_all_methods_summary.png")


def print_fa_md_summary(config: Config, stats: dict[str, dict[float, dict[str, float]]]) -> None:
    print("\nFA / MD summary")
    for method, method_stats in stats.items():
        print(f"\n{method}")
        print("SNR | FA[%] | MD[%] | n_noise | n_signal")
        print("-" * 46)
        for snr_db in config.snr_db_values:
            row = method_stats[float(snr_db)]
            print(
                f"{float(snr_db):>3.0f} | "
                f"{100.0 * row['false_alarm_rate']:>5.1f} | "
                f"{100.0 * row['missed_detection_rate']:>5.1f} | "
                f"{int(row['n_noise']):>7d} | "
                f"{int(row['n_signal']):>8d}"
            )


def finish_plot(config: Config, filename: str) -> None:
    try:
        plt.tight_layout()
    except Exception:
        pass
    if config.save_plots:
        output_dir = Path(config.plot_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"saved plot: {output_path}", flush=True)
    if config.show_plots:
        plt.show()
    else:
        plt.close()


def plot_correlations_and_uw_positions(
        config: Config,
        sample: dict[str, Any],
        correlations: dict[str, np.ndarray],
        cpfsk_uw: np.ndarray,
        title: str = "",
) -> None:
    """
    Plot all correlations and draw dashed vertical lines at the true UW peaks.
    """
    fig, axes = plt.subplots(len(correlations), 1, figsize=(12, 2.8 * len(correlations)))
    if len(correlations) == 1:
        axes = [axes]

    for axis, (method, correlation) in zip(axes, correlations.items()):
        true_peaks = [
            start + true_peak_offset(method, config, cpfsk_uw)
            for start in sample["uw_start_samples"]
        ]

        axis.plot(correlation, label=method)

        for peak in true_peaks:
            axis.axvline(peak, color="tab:green", linestyle="--", linewidth=1.5, label="true UW")
        axis.axhline(config.detection_threshold, color="black", linestyle=":", linewidth=1.0, label="threshold")

        axis.set_title(method)
        axis.set_ylabel("corr")
        axis.grid(True)
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc="upper right")

    axes[-1].set_xlabel("correlation index")
    fig.suptitle(title)
    finish_plot(config, "correlation_example.png")


if __name__ == "__main__":
    scenarios = [
        Config(
            simulate_noninteger_sps=True,
            resample_noninteger_back_to_integer=False,
            sample_rate=10_000,
            symbol_time=1.0 / 2400.0,
            physical_sample_rate=10_000,
            symbol_rate=2400,
            integer_reference_sps=4,
            snr_db_values=(0, 2, 4, 6, 8, 10, 12),
            frequency_offsets_hz=(0,),
            n_signals_per_condition=5,
            plot_output_dir="demodulator_research/simulations/uw_correlation_quality_plots/noninteger_raw_10khz",
        ),
        Config(
            simulate_noninteger_sps=True,
            resample_noninteger_back_to_integer=True,
            sample_rate=10_000,
            symbol_time=1.0 / 2400.0,
            physical_sample_rate=10_000,
            symbol_rate=2400,
            integer_reference_sps=4,
            snr_db_values=(0, 2, 4, 6, 8, 10, 12),
            frequency_offsets_hz=(0,),
            n_signals_per_condition=5,
            plot_output_dir="demodulator_research/simulations/uw_correlation_quality_plots/noninteger_resampled_to_integer",
        ),
    ]

    for config in scenarios:
        reference_config = detection_config(config)
        print("\n" + "=" * 100)
        print(
            f"scenario: {Path(config.plot_output_dir).name} | "
            f"physical_sps={config.physical_sample_rate / config.symbol_rate:.6f} | "
            f"detector_sps={int(round(reference_config.sample_rate * reference_config.symbol_time))} | "
            f"resample_back={config.resample_noninteger_back_to_integer}"
        )

        dataset = create_dataset(config)
        cpfsk_uw = build_cpfsk_uw(config)
        cpfsk_uw_middle, uw_crop_start_sample, uw_crop_stop_sample = crop_stable_uw_middle(reference_config, cpfsk_uw)
        print(
            f"middle UW crop: start={uw_crop_start_sample}, stop={uw_crop_stop_sample}, "
            f"length={len(cpfsk_uw_middle)} samples"
        )

        stats = evaluate_fa_md_vs_snr(config=reference_config, dataset=dataset, cpfsk_uw=cpfsk_uw)
        print_fa_md_summary(config=reference_config, stats=stats)
        plot_all_methods_summary(config=reference_config, stats=stats)
        plot_fa_md_vs_snr_by_method(config=reference_config, stats=stats)
