from pathlib import Path
import os
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-cache"))

import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from demodulator_research.pipeline.dataset.dataset_for_pipeline import build_pipeline_test_dataset
from demodulator_research.pipeline.detection.detection import (
    detect_middle_uw_grid,
    estimate_frequency_offset_from_detected_uws,
    fine_frequency_estimation,
)
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.modulator.modulator import Modulator, ModulatorConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig


def build_clean_full_uw(params: dict[str, Any]) -> np.ndarray:
    bits_per_symbol = int(np.log2(int(params["constellation_order"])))
    sps = int(round(float(params["sample_rate"]) * float(params["symbol_time"])))
    uw_n_symbols = len(params["uw_bits"]) // bits_per_symbol
    reference_len = max((uw_n_symbols - 1) * sps + 1, int(params["span_in_symbols"]) * sps + 1)
    padding_bits = max(int(params["span_in_symbols"]), 1) * bits_per_symbol

    pulse_shape_config = PulseShapeConfig(pulse_shape_type=params["pulse_shape_type"],
                                          normalization_type=params["pulse_normalization"],
                                          span_in_symbols=params["span_in_symbols"],
                                          pulse_causal=params["pulse_causal"])
    constellation_config = ConstellationConfig(constellation_type=params["constellation_type"],
                                               constellation_order=params["constellation_order"])
    modulator = Modulator(ModulatorConfig(pulse_shape_config=pulse_shape_config,
                                          constellation_config=constellation_config))
    cpfsk_uw, _ = modulator.modulate(bits=np.concatenate((
                                         np.asarray(params["uw_bits"], dtype=np.uint8),
                                         np.zeros(padding_bits, dtype=np.uint8),
                                     )),
                                     symbol_time=params["symbol_time"],
                                     sample_rate=params["sample_rate"],
                                     frequency_offset=0.0,
                                     frequency_sensitivity=params["h"],
                                     phase_offset=0.0,
                                     uw=None)
    return np.asarray(cpfsk_uw[:reference_len], dtype=np.complex128)


def crop_middle_uw(cpfsk_uw: np.ndarray, sps: int, n_symbols_each_side: int,
                   n_uw_bits: int, bits_per_symbol: int, span: int) -> tuple[np.ndarray, int]:
    uw_n_symbols = int(n_uw_bits) // int(bits_per_symbol)
    crop_center = np.max((int(uw_n_symbols * sps) // 2, int(span * sps) // 2))
    crop_start = crop_center - int(sps) * int(n_symbols_each_side)
    crop_stop = crop_center + int(sps) * int(n_symbols_each_side)
    if crop_start < 0 or crop_stop > len(cpfsk_uw) or crop_start >= crop_stop:
        raise ValueError("middle UW crop is outside the full UW reference")
    return cpfsk_uw[crop_start:crop_stop], crop_start


def exact_grid_match(detected_starts: list[int], true_starts: list[int], tolerance_samples: int) -> bool:
    if len(detected_starts) != len(true_starts):
        return False
    return all(abs(int(detected) - int(true)) <= int(tolerance_samples)
               for detected, true in zip(detected_starts, true_starts))


def count_matched_uw_starts(detected_starts: list[int], true_starts: list[int], tolerance_samples: int) -> int:
    unused_true_starts = list(map(int, true_starts))
    n_matches = 0

    for detected in map(int, detected_starts):
        if not unused_true_starts:
            break
        errors = [abs(detected - true_start) for true_start in unused_true_starts]
        best_match_index = int(np.argmin(errors))
        if errors[best_match_index] <= int(tolerance_samples):
            unused_true_starts.pop(best_match_index)
            n_matches += 1

    return n_matches


def partial_grid_match(detected_starts: list[int], true_starts: list[int], tolerance_samples: int,
                       min_n_uw: int) -> bool:
    return count_matched_uw_starts(
        detected_starts=detected_starts,
        true_starts=true_starts,
        tolerance_samples=tolerance_samples,
    ) >= int(min_n_uw)


def true_peak_presence(correlation: np.ndarray, true_starts: list[int], peak_offset: int,
                       tolerance_samples: int, threshold: float) -> tuple[bool, bool]:
    peak_found = []
    for true_start in true_starts:
        expected_peak = int(true_start) + int(peak_offset)
        start = max(0, expected_peak - int(tolerance_samples))
        stop = min(len(correlation), expected_peak + int(tolerance_samples) + 1)
        if start >= stop:
            peak_found.append(False)
            continue
        peak_found.append(float(np.max(correlation[start:stop])) >= float(threshold))

    return any(peak_found), all(peak_found) if peak_found else False


def evaluate_detection(dataset: dict[float, dict[int, dict[str, Any]]], params: dict[str, Any],
                       middle_cpfsk_uw: np.ndarray, full_cpfsk_uw: np.ndarray,
                       threshold: float) -> dict[float, dict[str, float]]:
    stats = {}
    sps = int(round(params["sample_rate"] * params["symbol_time"]))
    bits_per_symbol = int(np.log2(params["constellation_order"]))
    uw_n_symbols = len(params["uw_bits"]) // bits_per_symbol
    n_symbols_each_side = int(params["middle_crop_symbols_each_side"])
    true_peak_offset = (uw_n_symbols - 2 * n_symbols_each_side) * sps

    for snr_db in tqdm(sorted(dataset), desc="SNR", unit="snr"):
        n_noise = 0
        n_signal = 0
        n_false_alarm = 0
        n_missed_detection = 0
        n_bad_grid = 0
        n_grid_detected = 0
        n_any_true_peak_above_threshold = 0
        n_all_true_peaks_above_threshold = 0
        remaining_frequency_offset_hz = []
        remaining_coarse_frequency_offset_hz = []

        for sample in tqdm(dataset[snr_db].values(),
                           desc=f"SNR {float(snr_db):g} dB",
                           unit="sample",
                           leave=False):
            has_signal = bool(sample["n_uw"] > 0)
            if has_signal:
                n_signal += 1
            else:
                n_noise += 1

            detection = detect_middle_uw_grid(
                signal=sample["rx_signal"],
                middle_cpfsk_uw=middle_cpfsk_uw,
                threshold=float(threshold),
                sps=sps,
                uw_spacing_samples=sample["uw_spacing_samples"],
                n_uw=params["detection_n_uw"],
                grid_tolerance_samples=params["grid_tolerance_samples"],
                uw_n_symbols=uw_n_symbols,
                n_symbols_each_side=n_symbols_each_side,
                min_peak_distance_samples=params["min_peak_distance_samples"],
                min_n_uw=params["min_n_uw_for_detection"],
                first_peak_weight=params["grid_first_peak_weight"],
            )

            if not has_signal:
                n_false_alarm += int(detection["detected"])
                continue

            any_true_peak, all_true_peaks = true_peak_presence(
                correlation=detection["correlation"],
                true_starts=sample["uw_start_samples"],
                peak_offset=true_peak_offset,
                tolerance_samples=params["detection_tolerance_samples"],
                threshold=threshold,
            )
            n_any_true_peak_above_threshold += int(any_true_peak)
            n_all_true_peaks_above_threshold += int(all_true_peaks)
            n_grid_detected += int(detection["detected"])

            correct_detection = bool(detection["detected"]) and partial_grid_match(
                detected_starts=detection["uw_start_samples"],
                true_starts=sample["uw_start_samples"],
                tolerance_samples=params["detection_tolerance_samples"],
                min_n_uw=params["min_n_uw_for_detection"],
            )
            n_bad_grid += int(detection["detected"] and not correct_detection)
            n_missed_detection += int(not correct_detection)
            if correct_detection:
                coarse_frequency_estimation = estimate_frequency_offset_from_detected_uws(
                    signal=sample["rx_signal"],
                    uw_start_samples=detection["peak_indices"],
                    sample_rate=params["sample_rate"],
                )
                fine_frequency_estimation_result = fine_frequency_estimation(
                    signal=sample["rx_signal"],
                    sample_rate=params["sample_rate"],
                    coarse_estimation=coarse_frequency_estimation["frequency_offset_hz"],
                    middle_cpfsk_uw=middle_cpfsk_uw,
                )
                remaining_coarse_frequency_offset_hz.append(
                    float(sample["frequency_offset"]) - coarse_frequency_estimation["frequency_offset_hz"]
                )
                remaining_frequency_offset_hz.append(
                    float(sample["frequency_offset"]) - fine_frequency_estimation_result["frequency_offset_hz"]
                )


        stats[float(snr_db)] = {
            "n_noise": float(n_noise),
            "n_signal": float(n_signal),
            "n_bad_grid": float(n_bad_grid),
            "false_alarm_rate": float(n_false_alarm / n_noise) if n_noise else float("nan"),
            "missed_detection_rate": float(n_missed_detection / n_signal) if n_signal else float("nan"),
            "grid_detection_rate": float(n_grid_detected / n_signal) if n_signal else float("nan"),
            "any_true_peak_rate": float(n_any_true_peak_above_threshold / n_signal) if n_signal else float("nan"),
            "all_true_peaks_rate": float(n_all_true_peaks_above_threshold / n_signal) if n_signal else float("nan"),
            "remaining_frequency_offset_hz": remaining_frequency_offset_hz,
            "remaining_coarse_frequency_offset_hz": remaining_coarse_frequency_offset_hz,
            "remaining_frequency_offset_mean_hz": float(np.mean(remaining_frequency_offset_hz))
            if remaining_frequency_offset_hz else float("nan"),
            "remaining_frequency_offset_std_hz": float(np.std(remaining_frequency_offset_hz))
            if remaining_frequency_offset_hz else float("nan"),
            "remaining_coarse_frequency_offset_mean_hz": float(np.mean(remaining_coarse_frequency_offset_hz))
            if remaining_coarse_frequency_offset_hz else float("nan"),
            "remaining_coarse_frequency_offset_std_hz": float(np.std(remaining_coarse_frequency_offset_hz))
            if remaining_coarse_frequency_offset_hz else float("nan"),
        }

    return stats


def print_summary(stats: dict[float, dict[str, float]], threshold: float) -> None:
    print(f"Detection threshold = {threshold:.3f}")
    print("SNR | FA[%] | MD[%] | grid[%] | any true peak[%] | all true peaks[%] | bad grid | coarse rem mean [Hz] | fine rem mean [Hz] | fine rem std [Hz]")
    print("-" * 145)
    for snr_db in sorted(stats):
        row = stats[snr_db]
        print(f"{snr_db:>3.0f} | "
              f"{100.0 * row['false_alarm_rate']:>5.1f} | "
              f"{100.0 * row['missed_detection_rate']:>5.1f} | "
              f"{100.0 * row['grid_detection_rate']:>7.1f} | "
              f"{100.0 * row['any_true_peak_rate']:>16.1f} | "
              f"{100.0 * row['all_true_peaks_rate']:>17.1f} | "
              f"{int(row['n_bad_grid']):>8d} | "
              f"{row['remaining_coarse_frequency_offset_mean_hz']:>20.2f} | "
              f"{row['remaining_frequency_offset_mean_hz']:>18.2f} | "
              f"{row['remaining_frequency_offset_std_hz']:>17.2f}")


def plot_fa_md_vs_snr(stats: dict[float, dict[str, float]], threshold: float) -> None:
    snrs = np.asarray(sorted(stats), dtype=float)
    fa = np.asarray([stats[snr]["false_alarm_rate"] for snr in snrs], dtype=float)
    md = np.asarray([stats[snr]["missed_detection_rate"] for snr in snrs], dtype=float)

    plt.figure(figsize=(9, 5), constrained_layout=True)
    plt.plot(snrs, 100.0 * fa, marker="o", label="FA")
    plt.plot(snrs, 100.0 * md, marker="o", label="MD")
    plt.title(f"FA / MD vs SNR | threshold={threshold:.3f}")
    plt.xlabel("SNR [dB]")
    plt.ylabel("rate [%]")
    plt.ylim(-5, 105)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_remaining_frequency_offset_vs_snr(stats: dict[float, dict[str, Any]]) -> None:
    snr_points = []
    remaining_frequency_offset_points = []
    coarse_snr_points = []
    coarse_remaining_frequency_offset_points = []

    for snr_db in sorted(stats):
        offsets = stats[snr_db]["remaining_frequency_offset_hz"]
        snr_points.extend([float(snr_db)] * len(offsets))
        remaining_frequency_offset_points.extend(offsets)
        coarse_offsets = stats[snr_db]["remaining_coarse_frequency_offset_hz"]
        coarse_snr_points.extend([float(snr_db)] * len(coarse_offsets))
        coarse_remaining_frequency_offset_points.extend(coarse_offsets)

    if not remaining_frequency_offset_points:
        print("No correct detected signal samples for remaining frequency offset plot.")
        return

    plt.figure(figsize=(9, 5), constrained_layout=True)
    plt.scatter(coarse_snr_points, coarse_remaining_frequency_offset_points, s=20, alpha=0.35,
                label="coarse")
    plt.scatter(snr_points, remaining_frequency_offset_points, s=24, alpha=0.65,
                label="fine grid")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Remaining Frequency Offset vs SNR - Correct Detected Signals")
    plt.xlabel("SNR [dB]")
    plt.ylabel("true CFO - estimated CFO [Hz]")
    plt.grid(True)
    plt.legend()
    plt.show()


def main() -> None:
    params = {
        "sample_rate": 10000,
        "symbol_time": 1e-3,
        "h": 0.5,
        "pulse_shape_type": "rrc",
        "span_in_symbols": 10,
        "pulse_causal": False,
        "pulse_normalization": "cpfsk",
        "constellation_type": "PAM",
        "constellation_order": 4,
        "noise_type": "snr",
        "snr_db_values": tuple(range(0, 20, 1)),
        "n_samples_per_snr": 20,
        "signal_probability": 0.5,
        "min_frequency_offset_hz": -500.0,
        "max_frequency_offset_hz": 500.0,
        "min_phase_offset_rad": 0.0,
        "max_phase_offset_rad": 0 * np.pi,
        "simulate_timing_offset": False,
        "timing_offset_symbols": 0.0,
        "min_timing_offset_symbols": -0.25,
        "max_timing_offset_symbols": 0.25,
        "timing_offset_filter_taps": 81,
        "total_signal_duration_sec": 0.600,
        "leading_noise_duration_sec": 0.100,
        "uw_spacing_sec": 0.100,
        "n_uw": 5,
        "detection_n_uw": None,
        "uw_bits": [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        "middle_crop_symbols_each_side": 3,
        "detection_tolerance_samples": 8,
        "grid_tolerance_samples": 20,
        "grid_first_peak_weight": 1.0,
        "min_n_uw_for_detection": 3,
        "min_peak_distance_samples": 10,
        "fine_frequency_search_half_width_hz": 100.0,
        "fine_frequency_search_step_hz": 10.0,
        "random_seed": 1,
    }
    threshold = 0.375

    sps = int(round(params["sample_rate"] * params["symbol_time"]))
    bits_per_symbol = int(np.log2(params["constellation_order"]))
    full_uw = build_clean_full_uw(params)
    middle_uw, middle_start = crop_middle_uw(full_uw, sps, params["middle_crop_symbols_each_side"],
                                             n_uw_bits=len(params["uw_bits"]),
                                             bits_per_symbol=bits_per_symbol,
                                             span=params['span_in_symbols'])

    dataset = build_pipeline_test_dataset(params)
    stats = evaluate_detection(dataset=dataset,
                               params=params,
                               middle_cpfsk_uw=middle_uw,
                               full_cpfsk_uw=full_uw,
                               threshold=threshold)
    print_summary(stats, threshold)
    plot_fa_md_vs_snr(stats, threshold)
    plot_remaining_frequency_offset_vs_snr(stats)


if __name__ == "__main__":
    main()
