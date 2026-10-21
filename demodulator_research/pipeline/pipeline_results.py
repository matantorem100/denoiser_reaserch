from pathlib import Path
import os
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-cache"))

import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from demodulator_research.pipeline.pipeline import (
    calculate_remaining_phase_at_uws,
    create_dataset,
    pipeline,
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

    pulse_shape_config = PulseShapeConfig(
        pulse_shape_type=params["pulse_shape_type"],
        normalization_type=params["pulse_normalization"],
        span_in_symbols=params["span_in_symbols"],
        pulse_causal=params["pulse_causal"],
    )
    constellation_config = ConstellationConfig(
        constellation_type=params["constellation_type"],
        constellation_order=params["constellation_order"],
    )
    modulator = Modulator(ModulatorConfig(
        pulse_shape_config=pulse_shape_config,
        constellation_config=constellation_config,
    ))
    cpfsk_uw, _ = modulator.modulate(
        bits=np.concatenate((
            np.asarray(params["uw_bits"], dtype=np.uint8),
            np.zeros(padding_bits, dtype=np.uint8),
        )),
        symbol_time=params["symbol_time"],
        sample_rate=params["sample_rate"],
        frequency_offset=0.0,
        frequency_sensitivity=params["h"],
        phase_offset=0.0,
        uw=None,
    )
    return np.asarray(cpfsk_uw[:reference_len], dtype=np.complex128)


def crop_middle_uw(cpfsk_uw: np.ndarray, sps: int, n_symbols_each_side: int,
                   n_uw_bits: int, bits_per_symbol: int, span: int) -> tuple[np.ndarray, int]:
    uw_n_symbols = int(n_uw_bits) // int(bits_per_symbol)
    crop_center = max(int((uw_n_symbols-1) * sps) // 2, int(span * sps) // 2)
    crop_start = crop_center - int(sps) * int(n_symbols_each_side)
    crop_stop = crop_center + int(sps) * int(n_symbols_each_side)
    if crop_start < 0 or crop_stop > len(cpfsk_uw) or crop_start >= crop_stop:
        raise ValueError("middle UW crop is outside the full UW reference")
    return cpfsk_uw[crop_start:crop_stop], crop_start


def calculate_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    n_bits = min(len(tx_bits), len(rx_bits))
    if n_bits == 0:
        return float("nan")
    return float(np.mean(np.asarray(tx_bits[4:n_bits], dtype=np.uint8) != np.asarray(rx_bits[4:n_bits], dtype=np.uint8)))


def bit_error_indices(tx_bits: np.ndarray, rx_bits: np.ndarray, max_indices: int = 20) -> list[int]:
    n_bits = min(len(tx_bits), len(rx_bits))
    if n_bits == 0:
        return []
    tx = np.asarray(tx_bits[:n_bits], dtype=np.uint8)
    rx = np.asarray(rx_bits[:n_bits], dtype=np.uint8)
    return np.flatnonzero(tx != rx)[:int(max_indices)].astype(int).tolist()


def calculate_ber_for_offset(tx_bits: np.ndarray, rx_bits: np.ndarray,
                             bit_offset: int, edge_trim_bits: int = 0) -> tuple[float, int]:
    tx_bits = np.asarray(tx_bits, dtype=np.uint8).reshape(-1)
    rx_bits = np.asarray(rx_bits, dtype=np.uint8).reshape(-1)

    if bit_offset >= 0:
        tx_aligned = tx_bits[int(bit_offset):]
        rx_aligned = rx_bits
    else:
        tx_aligned = tx_bits
        rx_aligned = rx_bits[int(-bit_offset):]

    n_bits = min(len(tx_aligned), len(rx_aligned))
    if n_bits == 0:
        return float("nan"), 0

    tx_aligned = tx_aligned[:n_bits]
    rx_aligned = rx_aligned[:n_bits]

    edge_trim_bits = int(edge_trim_bits)
    if edge_trim_bits > 0:
        if n_bits <= 2 * edge_trim_bits:
            return float("nan"), 0
        tx_aligned = tx_aligned[edge_trim_bits:-edge_trim_bits]
        rx_aligned = rx_aligned[edge_trim_bits:-edge_trim_bits]
        n_bits = len(tx_aligned)

    return float(np.mean(tx_aligned != rx_aligned)), int(n_bits)


def calculate_aligned_differential_ber(tx_bits: np.ndarray, rx_bits: np.ndarray,
                                       bits_per_symbol: int, span_in_symbols: int,
                                       edge_trim_symbols: int | None = None) -> tuple[float, int, int]:
    if edge_trim_symbols is None:
        edge_trim_symbols = max(1, int(np.ceil(float(span_in_symbols) / 2.0)))
    edge_trim_bits = max(0, int(edge_trim_symbols)) * int(bits_per_symbol)
    max_symbol_shift = max(int(np.ceil(float(span_in_symbols))) + 2, 2)

    best_ber = float("nan")
    best_offset = 0
    best_n_bits = 0
    for symbol_shift in range(-max_symbol_shift, max_symbol_shift + 1):
        bit_offset = int(symbol_shift) * int(bits_per_symbol)
        ber, n_bits = calculate_ber_for_offset(
            tx_bits=tx_bits,
            rx_bits=rx_bits,
            bit_offset=bit_offset,
            edge_trim_bits=edge_trim_bits,
        )
        if n_bits == 0 or not np.isfinite(ber):
            continue
        if not np.isfinite(best_ber) or ber < best_ber or (ber == best_ber and abs(bit_offset) < abs(best_offset)):
            best_ber = float(ber)
            best_offset = int(bit_offset)
            best_n_bits = int(n_bits)

    return best_ber, best_offset, best_n_bits


def wrap_frequency_error_hz(error_hz: float, ambiguity_period_hz: float) -> float:
    if not np.isfinite(error_hz) or not np.isfinite(ambiguity_period_hz) or ambiguity_period_hz <= 0.0:
        return float(error_hz)
    return float((float(error_hz) + 0.5 * float(ambiguity_period_hz)) % float(ambiguity_period_hz)
                 - 0.5 * float(ambiguity_period_hz))


def wrap_phase_rad(phase_rad: float) -> float:
    return float(np.angle(np.exp(1j * float(phase_rad))))


def wrap_fractional_sample_error(error_samples: float) -> float:
    return float((float(error_samples) + 0.5) % 1.0 - 0.5)


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


def format_timing(timing_sec: dict[str, float]) -> str:
    return (
        f"det={timing_sec['detection']:.3f}s "
        f"timing_est={timing_sec['timing_estimation']:.3f}s "
        f"timing_corr={timing_sec['timing_correction']:.3f}s "
        f"freq={timing_sec['frequency_estimation']:.3f}s "
        f"phase={timing_sec['phase_estimation']:.3f}s "
        f"vit={timing_sec['demodulation']:.3f}s "
        f"diff={timing_sec['differential_demodulation']:.3f}s "
        f"total={timing_sec['total']:.3f}s"
    )


def print_detected_signal_summary(snr_db: float, sample_index: int, viterbi_ber: float,
                                  differential_ber: float,
                                  true_frequency_offset_hz: float,
                                  frequency_offset_hz: float,
                                  timing_sec: dict[str, float],
                                  viterbi_error_indices: list[int] | None = None,
                                  differential_error_indices: list[int] | None = None) -> None:
    frequency_error = true_frequency_offset_hz - frequency_offset_hz
    viterbi_error_indices = viterbi_error_indices or []
    differential_error_indices = differential_error_indices or []
    print(
        f"detected | snr={snr_db:g} dB | sample={sample_index} | "
        f"vitBER={viterbi_ber:.3e} | diffBER={differential_ber:.3e} | "
        f"vit_err_idx={viterbi_error_indices} | diff_err_idx={differential_error_indices} | "
        f"det_cfo_err={frequency_error:.2f} Hz | "
        f"{format_timing(timing_sec)}"
    )


def evaluate_pipeline(dataset: dict[float, dict[int, dict[str, Any]]], params: dict[str, Any],
                      middle_cpfsk_uw: np.ndarray, middle_cpfsk_uw_start: int, full_uw: np.ndarray) -> dict[float, dict[str, Any]]:
    sps = int(round(params["sample_rate"] * params["symbol_time"]))
    bits_per_symbol = int(np.log2(params["constellation_order"]))
    uw_n_symbols = len(params["uw_bits"]) // bits_per_symbol
    n_symbols_each_side = int(params["middle_crop_symbols_each_side"])
    stats = {}
    pll_lock_plots_shown = 0
    max_pll_lock_plots = int(params.get("max_pll_lock_plots", 5))

    for snr_db in tqdm(sorted(dataset), desc="SNR", unit="snr"):
        n_noise = 0
        n_signal = 0
        n_false_alarm = 0
        n_missed_detection = 0
        n_bad_grid = 0
        remaining_detection_complex_phase_rad = []
        remaining_correlation_peak_frequency_offset_hz = []
        remaining_timing_offset_samples = []
        remaining_timing_offset_symbols = []
        remaining_parabolic_timing_offset_samples = []
        max_differential_correlation_values = []
        viterbi_ber_values = []
        differential_ber_values = []
        viterbi_nonzero_ber_samples = []
        differential_nonzero_ber_samples = []
        viterbi_n_bit_values = []
        differential_n_bit_values = []
        demod_times_sec = []
        timing_values = {
            "detection": [],
            "timing_estimation": [],
            "timing_correction": [],
            "frequency_estimation": [],
            "phase_estimation": [],
            "demodulation": [],
            "differential_demodulation": [],
            "total": [],
        }

        for sample_index, sample in tqdm(dataset[snr_db].items(), desc=f"SNR {float(snr_db):g} dB",
                                         unit="sample", leave=False):

            has_signal = bool(sample["has_signal"])
            if has_signal:
                n_signal += 1
            else:
                n_noise += 1

            tx_bits = np.asarray(sample["tx_bits"], dtype=np.uint8) if has_signal else np.array([], dtype=np.uint8)
            n_symbols = len(tx_bits) // bits_per_symbol if has_signal else None
            signal_start = int(sample["leading_noise_samples"])
            sample_params = dict(params)
            sample_params["uw_spacing_samples"] = sample["uw_spacing_samples"]
            sample_params["run_demodulation"] = bool(has_signal)

            pipeline_result = pipeline(signal=sample["rx_signal"], middle_cpfsk_uw=middle_cpfsk_uw, params=sample_params,
                                       n_symbols=n_symbols, signal_start_sample=signal_start, full_uw=full_uw)
            timing_sec = pipeline_result["timing_sec"]
            for timing_name in timing_values:
                timing_values[timing_name].append(float(timing_sec[timing_name]))
            detection_result = pipeline_result["detection"]

            if not has_signal:
                n_false_alarm += int(detection_result["detected"])
                continue

            n_matched_uw = count_matched_uw_starts(detected_starts=detection_result["uw_start_samples"],
                                                   true_starts=sample["uw_start_samples"],
                                                   tolerance_samples=params["detection_tolerance_samples"])
            correct_detection = bool(detection_result["detected"]) and n_matched_uw >= int(params["min_n_uw_for_detection"])
            n_bad_grid += int(detection_result["detected"] and not correct_detection)
            n_missed_detection += int(not correct_detection)
            if not correct_detection:
                continue

            frequency_estimation = pipeline_result["frequency_estimation"]

            remaining_detection_complex_phase_rad.append(
                wrap_phase_rad(float(sample["phase_offset"]) - float(detection_result["initial_phase_estimation_rad"]))
            )
            remaining_correlation_peak_frequency_offset_hz.append(
                float(sample["frequency_offset"]) - float(frequency_estimation["frequency_offset_hz"])
            )
            timing_estimation = pipeline_result["timing_estimation"]
            estimated_timing_offset_samples = float(timing_estimation["timing_offset_samples"])
            true_timing_offset_samples = float(sample.get("timing_offset_samples", 0.0))
            if np.isfinite(estimated_timing_offset_samples):
                timing_residual_samples = true_timing_offset_samples - estimated_timing_offset_samples
                remaining_timing_offset_samples.append(float(timing_residual_samples))
                remaining_timing_offset_symbols.append(float(timing_residual_samples) / float(sps))
            estimated_fractional_timing_samples = float(timing_estimation.get("fractional_timing_offset_samples", float("nan")))
            true_fractional_timing_samples = float(sample.get(
                "fractional_timing_offset_samples",
                true_timing_offset_samples,
            ))
            if np.isfinite(estimated_fractional_timing_samples):
                true_fractional_timing_samples = wrap_fractional_sample_error(true_fractional_timing_samples)
                parabolic_residual_samples = wrap_fractional_sample_error(
                    true_fractional_timing_samples - estimated_fractional_timing_samples
                )
                remaining_parabolic_timing_offset_samples.append(float(parabolic_residual_samples))
            peak_values = np.asarray(detection_result.get("peak_values", []), dtype=float)
            if len(peak_values):
                max_differential_correlation_values.append(float(np.max(peak_values)))

            demodulation = pipeline_result["demodulation"]
            differential_bits = pipeline_result["differential_demodulation"]
            demod_times_sec.append(float(timing_sec["total"]))
            if demodulation is None or differential_bits is None:
                continue

            rx_bits = demodulation["bits"]
            n_bits = min(len(tx_bits), len(rx_bits))
            viterbi_ber = calculate_ber(tx_bits=tx_bits, rx_bits=rx_bits)
            viterbi_error_indices = bit_error_indices(tx_bits=tx_bits, rx_bits=rx_bits,
                                                      max_indices=params.get("max_print_error_indices", 20))
            viterbi_ber_values.append(viterbi_ber)
            viterbi_n_bit_values.append(n_bits)
            if np.isfinite(viterbi_ber) and viterbi_ber > 0.0:
                viterbi_nonzero_ber_samples.append({
                    "sample_index": int(sample_index),
                    "ber": float(viterbi_ber),
                    "error_indices": viterbi_error_indices,
                })

            if bool(params.get("plot_pll_lock", False)) and pll_lock_plots_shown < max_pll_lock_plots:
                plot_pll_lock_vs_symbol(
                    demodulation=demodulation,
                    snr_db=float(snr_db),
                    sample_index=int(sample_index),
                    true_frequency_offset_hz=float(sample["frequency_offset"]),
                )
                pll_lock_plots_shown += 1

            differential_ber, _differential_bit_offset, differential_n_bits = calculate_aligned_differential_ber(
                tx_bits=tx_bits,
                rx_bits=differential_bits,
                bits_per_symbol=bits_per_symbol,
                span_in_symbols=params["span_in_symbols"],
                edge_trim_symbols=params.get("differential_ber_edge_trim_symbols", None),
            )
            if differential_n_bits > 0 and np.isfinite(differential_ber):
                differential_ber_values.append(differential_ber)
                differential_n_bit_values.append(differential_n_bits)
            differential_error_indices = bit_error_indices(tx_bits=tx_bits, rx_bits=differential_bits,
                                                           max_indices=params.get("max_print_error_indices", 20))
            if np.isfinite(differential_ber) and differential_ber > 0.0:
                differential_nonzero_ber_samples.append({
                    "sample_index": int(sample_index),
                    "ber": float(differential_ber),
                    "direct_error_indices": differential_error_indices,
                })

            print_detected_signal_summary(
                snr_db=float(snr_db),
                sample_index=int(sample_index),
                viterbi_ber=viterbi_ber,
                differential_ber=differential_ber,
                true_frequency_offset_hz=float(sample["frequency_offset"]),
                frequency_offset_hz=float(frequency_estimation["frequency_offset_hz"]),
                timing_sec=timing_sec,
                viterbi_error_indices=viterbi_error_indices,
                differential_error_indices=differential_error_indices,
            )

        stats[float(snr_db)] = {
            "n_noise": float(n_noise),
            "n_signal": float(n_signal),
            "n_bad_grid": float(n_bad_grid),
            "false_alarm_rate": float(n_false_alarm / n_noise) if n_noise else float("nan"),
            "missed_detection_rate": float(n_missed_detection / n_signal) if n_signal else float("nan"),
            "remaining_detection_complex_phase_rad": remaining_detection_complex_phase_rad,
            "remaining_correlation_peak_frequency_offset_hz": remaining_correlation_peak_frequency_offset_hz,
            "remaining_timing_offset_samples": remaining_timing_offset_samples,
            "remaining_timing_offset_symbols": remaining_timing_offset_symbols,
            "remaining_parabolic_timing_offset_samples": remaining_parabolic_timing_offset_samples,
            "max_differential_correlation_values": max_differential_correlation_values,
            "remaining_detection_complex_phase_mean_rad": float(np.mean(remaining_detection_complex_phase_rad))
            if remaining_detection_complex_phase_rad else float("nan"),
            "remaining_detection_complex_phase_std_rad": float(np.std(remaining_detection_complex_phase_rad))
            if remaining_detection_complex_phase_rad else float("nan"),
            "remaining_correlation_peak_frequency_offset_mean_hz": float(np.mean(remaining_correlation_peak_frequency_offset_hz))
            if remaining_correlation_peak_frequency_offset_hz else float("nan"),
            "remaining_correlation_peak_frequency_offset_std_hz": float(np.std(remaining_correlation_peak_frequency_offset_hz))
            if remaining_correlation_peak_frequency_offset_hz else float("nan"),
            "remaining_timing_offset_mean_samples": float(np.mean(remaining_timing_offset_samples))
            if remaining_timing_offset_samples else float("nan"),
            "remaining_timing_offset_std_samples": float(np.std(remaining_timing_offset_samples))
            if remaining_timing_offset_samples else float("nan"),
            "remaining_parabolic_timing_offset_mean_samples": float(np.mean(remaining_parabolic_timing_offset_samples))
            if remaining_parabolic_timing_offset_samples else float("nan"),
            "remaining_parabolic_timing_offset_std_samples": float(np.std(remaining_parabolic_timing_offset_samples))
            if remaining_parabolic_timing_offset_samples else float("nan"),
            "viterbi_ber": float(np.mean(viterbi_ber_values))
            if viterbi_ber_values else float("nan"),
            "differential_ber": float(np.mean(differential_ber_values))
            if differential_ber_values else float("nan"),
            "ber": float(np.mean(viterbi_ber_values))
            if viterbi_ber_values else float("nan"),
            "n_ber_samples": float(len(viterbi_ber_values)),
            "viterbi_ber_values": [float(value) for value in viterbi_ber_values],
            "differential_ber_values": [float(value) for value in differential_ber_values],
            "viterbi_nonzero_ber_samples": viterbi_nonzero_ber_samples,
            "differential_nonzero_ber_samples": differential_nonzero_ber_samples,
            "mean_demod_time_sec": float(np.mean(demod_times_sec)) if demod_times_sec else float("nan"),
            "timing_sec": {
                timing_name: {
                    "mean": float(np.mean(values)) if values else float("nan"),
                    "sum": float(np.sum(values)) if values else float("nan"),
                    "count": float(len(values)),
                }
                for timing_name, values in timing_values.items()
            },
        }

    return stats


def plot_fa_md_vs_snr(stats: dict[float, dict[str, Any]]) -> None:
    snrs = np.asarray(sorted(stats), dtype=float)
    plt.figure(figsize=(9, 5), constrained_layout=True)
    plt.plot(snrs, [100.0 * stats[snr]["false_alarm_rate"] for snr in snrs], marker="o", label="FA")
    plt.plot(snrs, [100.0 * stats[snr]["missed_detection_rate"] for snr in snrs], marker="o", label="MD")
    plt.xlabel("SNR [dB]")
    plt.ylabel("rate [%]")
    plt.title("FA / MD vs SNR")
    plt.ylim(-5, 105)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_frequency_estimation_vs_snr(stats: dict[float, dict[str, Any]]) -> None:
    snr_points = []
    offset_points = []
    for snr_db in sorted(stats):
        offsets = stats[snr_db]["remaining_correlation_peak_frequency_offset_hz"]
        snr_points.extend([float(snr_db)] * len(offsets))
        offset_points.extend(offsets)

    if not offset_points:
        print("No correct detected signal samples for correlation-peak frequency offset plot.")
        return

    plt.figure(figsize=(9, 5), constrained_layout=True)
    plt.scatter(snr_points, offset_points, s=24, alpha=0.65, label="correlation peak phase CFO")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("SNR [dB]")
    plt.ylabel("true CFO - estimated CFO [Hz]")
    plt.title("Used Frequency Offset Residual vs SNR")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_detection_complex_phase_estimation_vs_snr(stats: dict[float, dict[str, Any]]) -> None:
    snr_points = []
    phase_points = []
    for snr_db in sorted(stats):
        phases = stats[snr_db]["remaining_detection_complex_phase_rad"]
        snr_points.extend([float(snr_db)] * len(phases))
        phase_points.extend(phases)

    if not phase_points:
        print("No correct detected signal samples for detection complex phase plot.")
        return

    plt.figure(figsize=(9, 5), constrained_layout=True)
    plt.scatter(snr_points, phase_points, s=24, alpha=0.65)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("SNR [dB]")
    plt.ylabel("wrapped true phase - estimated phase [rad]")
    plt.title("Detection Complex Correlation Initial Phase Residual vs SNR")
    plt.grid(True)
    plt.show()


def plot_timing_estimation_vs_snr(stats: dict[float, dict[str, Any]]) -> None:
    snr_points = []
    timing_points = []
    for snr_db in sorted(stats):
        timing_offsets = stats[snr_db]["remaining_timing_offset_samples"]
        snr_points.extend([float(snr_db)] * len(timing_offsets))
        timing_points.extend(timing_offsets)

    if not timing_points:
        print("No correct detected signal samples for timing correction quality plot.")
        return

    plt.figure(figsize=(9, 5), constrained_layout=True)
    plt.scatter(snr_points, timing_points, s=24, alpha=0.65)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("SNR [dB]")
    plt.ylabel("true timing offset - estimated timing offset [samples]")
    plt.title("Timing Correction Residual vs SNR")
    plt.grid(True)
    plt.show()


def plot_parabolic_timing_estimation_vs_snr(stats: dict[float, dict[str, Any]]) -> None:
    snr_points = []
    timing_points = []
    for snr_db in sorted(stats):
        timing_offsets = stats[snr_db]["remaining_parabolic_timing_offset_samples"]
        snr_points.extend([float(snr_db)] * len(timing_offsets))
        timing_points.extend(timing_offsets)

    if not timing_points:
        print("No correct detected signal samples for parabolic timing quality plot.")
        return

    plt.figure(figsize=(9, 5), constrained_layout=True)
    plt.scatter(snr_points, timing_points, s=24, alpha=0.65)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("SNR [dB]")
    plt.ylabel("true fractional timing - parabolic estimate [samples]")
    plt.title("Parabolic Fractional Timing Residual vs SNR")
    plt.grid(True)
    plt.show()


def plot_max_differential_correlation_vs_snr(stats: dict[float, dict[str, Any]]) -> None:
    snr_points = []
    peak_points = []

    for snr_db in sorted(stats):
        peak_values = stats[snr_db]["max_differential_correlation_values"]
        snr_points.extend([float(snr_db)] * len(peak_values))
        peak_points.extend(peak_values)

    if not peak_points:
        print("No correct detected signal samples for differential correlation peak plot.")
        return

    plt.figure(figsize=(9, 5), constrained_layout=True)
    plt.scatter(snr_points, peak_points, s=24, alpha=0.65)
    plt.xlabel("SNR [dB]")
    plt.ylabel("max differential correlation peak")
    plt.title("Max Differential Correlation Peak vs SNR")
    plt.grid(True)
    plt.show()


def plot_ber_and_time_vs_snr(stats: dict[float, dict[str, Any]], ber_floor: float) -> None:
    snrs = np.asarray(sorted(stats), dtype=float)
    viterbi_ber = np.asarray([stats[snr]["viterbi_ber"] for snr in snrs], dtype=float)
    differential_ber = np.asarray([stats[snr]["differential_ber"] for snr in snrs], dtype=float)
    pipeline_time = np.asarray([stats[snr]["timing_sec"]["total"]["mean"] for snr in snrs], dtype=float)

    fig, ax_ber = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax_ber.semilogy(snrs, np.maximum(viterbi_ber, ber_floor), marker="o", label="Viterbi BER")
    ax_ber.semilogy(snrs, np.maximum(differential_ber, ber_floor), marker="s", label="Differential BER")
    ax_ber.set_xlabel("SNR [dB]")
    ax_ber.set_ylabel(f"BER (zeros shown at {ber_floor:g})")
    ax_ber.grid(True, which="both")

    ax_time = ax_ber.twinx()
    ax_time.bar(snrs, pipeline_time, width=0.55, alpha=0.25, color="tab:orange", label="pipeline time")
    ax_time.set_ylabel("mean pipeline time [sec]")

    lines, labels = ax_ber.get_legend_handles_labels()
    bars, bar_labels = ax_time.get_legend_handles_labels()
    ax_ber.legend(lines + bars, labels + bar_labels, loc="best")
    ax_ber.set_title("BER and Pipeline Time vs SNR")
    plt.show()


def plot_pll_lock_vs_symbol(demodulation: dict[str, Any], snr_db: float, sample_index: int,
                            true_frequency_offset_hz: float | None = None) -> None:
    phase_error = np.asarray(demodulation.get("tracking_phase_error_rad", []), dtype=float)
    tracked_frequency = np.asarray(demodulation.get("tracked_frequency_hz", []), dtype=float)
    n_points = max(len(phase_error), len(tracked_frequency))
    if n_points == 0:
        print(f"No PLL tracking trace for SNR {snr_db:g} dB sample {sample_index}.")
        return

    symbols = np.arange(n_points, dtype=int)
    fig, (ax_phase, ax_frequency) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)

    if len(phase_error):
        ax_phase.plot(symbols[:len(phase_error)], phase_error, linewidth=1.2)
    ax_phase.axhline(0.0, color="black", linewidth=1)
    ax_phase.set_ylabel("phase error [rad]")
    ax_phase.set_title(f"PLL Lock Trace | SNR {snr_db:g} dB | sample {sample_index}")
    ax_phase.grid(True)

    if len(tracked_frequency):
        ax_frequency.plot(symbols[:len(tracked_frequency)], tracked_frequency, linewidth=1.2, label="tracked frequency")
    if true_frequency_offset_hz is not None and np.isfinite(true_frequency_offset_hz):
        ax_frequency.axhline(float(true_frequency_offset_hz), color="tab:red", linestyle="--", linewidth=1,
                             label="true CFO")
        ax_frequency.legend(loc="best")
    ax_frequency.set_xlabel("symbol index")
    ax_frequency.set_ylabel("frequency [Hz]")
    ax_frequency.grid(True)
    plt.show()


def print_summary(stats: dict[float, dict[str, Any]]) -> None:
    print("SNR | FA[%] | MD[%] | corr-peak CFO rem [Hz] | Vit BER | Diff BER | n BER | pipeline time [s]")
    print("-" * 92)
    for snr_db in sorted(stats):
        row = stats[snr_db]
        print(f"{snr_db:>3.0f} | "
              f"{100.0 * row['false_alarm_rate']:>5.1f} | "
              f"{100.0 * row['missed_detection_rate']:>5.1f} | "
              f"{row['remaining_correlation_peak_frequency_offset_mean_hz']:>22.2f} | "
              f"{row['viterbi_ber']:>8.3e} | "
              f"{row['differential_ber']:>8.3e} | "
              f"{int(row['n_ber_samples']):>5d} | "
              f"{row['timing_sec']['total']['mean']:>17.3f}")
        if row["viterbi_nonzero_ber_samples"]:
            print(f"    nonzero Viterbi BER samples: {row['viterbi_nonzero_ber_samples']}")
        else:
            print(f"    nonzero Viterbi BER samples: none; raw values={row['viterbi_ber_values']}")
        if row["differential_nonzero_ber_samples"]:
            print(f"    nonzero Differential BER samples: {row['differential_nonzero_ber_samples']}")


def print_timing_summary(stats: dict[float, dict[str, Any]]) -> None:
    timing_names = [
        "detection",
        "timing_estimation",
        "timing_correction",
        "frequency_estimation",
        "phase_estimation",
        "demodulation",
        "differential_demodulation",
        "total",
    ]
    print("\nTiming summary per SNR")
    print("SNR | stage | mean [s] | sum [s] | count")
    print("-" * 52)
    for snr_db in sorted(stats):
        for timing_name in timing_names:
            row = stats[snr_db]["timing_sec"][timing_name]
            print(f"{snr_db:>3.0f} | "
                  f"{timing_name:<28} | "
                  f"{row['mean']:>8.4f} | "
                  f"{row['sum']:>7.2f} | "
                  f"{int(row['count']):>5d}")


def main() -> None:
    params = {
        "sample_rate": 10000,
        "symbol_time": 1e-3,
        "h": 0.5,
        "pulse_shape_type": "rrc",
        "span_in_symbols": 12,
        "pulse_causal": False,
        "pulse_normalization": "cpfsk",
        "constellation_type": "PAM",
        "constellation_order": 4,
        "noise_type": "snr",
        "snr_db_values": tuple(range(10, 15, 1)),
        "n_samples_per_snr": 20,
        "signal_probability": 1,
        "min_frequency_offset_hz": -300.0,
        "max_frequency_offset_hz": 300.0,
        "min_phase_offset_rad": 0 * np.pi,
        "max_phase_offset_rad": 2 * np.pi,
        "simulate_timing_offset": True,
        "min_integer_timing_shift_samples": -10,
        "max_integer_timing_shift_samples": 10,
        "min_frac_timing_shift_samples": -0.4,
        "max_frac_timing_shift_samples": 0.4,
        "timing_offset_filter_taps": 81,
        "correct_timing_offset": True,
        "timing_correction_filter_taps": 81,
        "timing_complex_correlation_peak_search_radius_samples": 20,
        "corrected_correlation_peak_search_radius_samples": 2,
        "corrected_complex_correlation_peak_search_radius_samples": 20,
        "total_signal_duration_sec": 0.3500,
        "leading_noise_duration_sec": 0.050,
        "uw_spacing_sec": 0.100,
        "n_uw": 3,
        "detection_n_uw": None,
        "uw_bits": [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        "middle_crop_symbols_each_side": 3,
        "detection_threshold": 0.35,
        "detection_tolerance_samples": 8,
        "grid_tolerance_samples": 20,
        "grid_first_peak_weight": 1.0,
        "min_n_uw_for_detection": 3,
        "min_peak_distance_samples": 10,
        "complex_phase_correlation_shift_samples": 5,
        "initial_phase_future_guard_symbols": None,
        "viterbi_depth_symbols": 10,
        "viterbi_max_survivors": 256,
        "viterbi_metric_memory_symbols": 3,
        "viterbi_metric_type": "real",  # "real" or "abs"
        "pll_alpha": 0.05,
        "pll_beta": 0.004,
        "differential_ber_edge_trim_symbols": None,
        "plot_pll_lock": False,
        "max_pll_lock_plots": 5,
        "max_print_error_indices": 20,
        "ber_plot_floor": 0.002,
        "random_seed":42,
    }


    sps = int(round(params["sample_rate"] * params["symbol_time"]))
    bits_per_symbol = int(np.log2(params["constellation_order"]))
    full_uw = build_clean_full_uw(params)
    middle_uw, middle_start = crop_middle_uw(cpfsk_uw=full_uw, sps=sps,
                                             n_symbols_each_side=params["middle_crop_symbols_each_side"],
                                             n_uw_bits=len(params["uw_bits"]), bits_per_symbol=bits_per_symbol,
                                             span=params["span_in_symbols"])

    dataset = create_dataset(params)
    stats = evaluate_pipeline(dataset=dataset, params=params, middle_cpfsk_uw=middle_uw,
                              middle_cpfsk_uw_start=middle_start, full_uw=full_uw)
    print_summary(stats)
    print_timing_summary(stats)
    plot_fa_md_vs_snr(stats)
    plot_frequency_estimation_vs_snr(stats)
    plot_detection_complex_phase_estimation_vs_snr(stats)
    plot_timing_estimation_vs_snr(stats)
    plot_parabolic_timing_estimation_vs_snr(stats)
    plot_max_differential_correlation_vs_snr(stats)
    plot_ber_and_time_vs_snr(stats, ber_floor=params["ber_plot_floor"])


if __name__ == "__main__":
    main()
