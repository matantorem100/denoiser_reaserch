from pathlib import Path
import os
import sys
from typing import Any

import numpy as np

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError as exc:
    raise SystemExit("Install Hydra first: python3 -m pip install hydra-core") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-cache"))

import matplotlib.pyplot as plt

from demodulator_research.demodulator.demodulator import Demodulator, DemodulatorConfig
from demodulator_research.find_grid.find_grid import find_best_uw_grid
from demodulator_research.frequency_offset_estimation.frequency_offset_estimation import FrequencyOffsetEstimator
from demodulator_research.uw_detection.uw_detection import UwDetector
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.dataset_creation import Dataset, DatasetConfig
from synthetic_dataset_creation.modulator.modulator import Modulator, ModulatorConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig


EPS = 1e-12


def finite_mean(values: list[float], default: float = float("nan")) -> float:
    values = [float(value) for value in values if np.isfinite(value)]
    return default if not values else float(np.mean(values))


def exact_list(a: list[int], b: list[int]) -> bool:
    return list(a) == list(b)


def build_configs(dataset_config: DatasetConfig) -> tuple[ModulatorConfig, DemodulatorConfig]:
    pulse_shape_config = PulseShapeConfig(
        pulse_shape_type=dataset_config.pulse_shape_type,
        normalization_type=dataset_config.pulse_normalization,
        span_in_symbols=dataset_config.span_in_symbols,
    )
    constellation_config = ConstellationConfig(
        constellation_type=dataset_config.constellation_type,
        constellation_order=dataset_config.constellation_order,
    )
    return (
        ModulatorConfig(pulse_shape_config=pulse_shape_config, constellation_config=constellation_config),
        DemodulatorConfig(pulse_shape_config=pulse_shape_config, constellation_config=constellation_config),
    )


def build_cpfsk_uw(dataset_config: DatasetConfig) -> np.ndarray:
    modulator_config, _ = build_configs(dataset_config)
    signal, _ = Modulator(modulator_config).modulate(
        bits=np.asarray(dataset_config.uw_bits, dtype=np.uint8),
        symbol_time=dataset_config.symbol_time,
        sample_rate=dataset_config.sample_rate,
        frequency_offset=0.0,
        frequency_sensitivity=dataset_config.h,
        uw=None,
    )
    return signal


def extract_segment(signal: np.ndarray, start_sample: int, segment_len: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    start_sample = max(int(start_sample), 0)
    segment = signal[start_sample:start_sample + int(segment_len)]
    if len(segment) < segment_len:
        segment = np.pad(segment, (0, segment_len - len(segment)))
    return segment


def circular_mean(phases: list[float]) -> float:
    phases = [phase for phase in phases if np.isfinite(phase)]
    if not phases:
        return float("nan")
    return float(np.angle(np.mean(np.exp(1j * np.asarray(phases)))))


def estimate_phase(rx_uw: np.ndarray, cpfsk_uw: np.ndarray) -> float:
    n = min(len(rx_uw), len(cpfsk_uw))
    if n == 0:
        return float("nan")
    return float(np.angle(np.vdot(cpfsk_uw[:n], rx_uw[:n])))


def ber(rx_bits: np.ndarray, tx_bits: np.ndarray, bits_per_symbol: int) -> float:
    n = min(len(rx_bits), len(tx_bits))
    n = (n // bits_per_symbol) * bits_per_symbol
    if n == 0:
        return float("nan")
    return float(np.mean(rx_bits[:n] != tx_bits[:n]))


def run_receiver(
        rx_signal: np.ndarray,
        tx_bits: np.ndarray | None,
        dataset_config: DatasetConfig,
        receiver_config: DictConfig,
        expected_n_uw: int | None,
) -> dict[str, Any]:
    rx_signal = np.asarray(rx_signal, dtype=np.complex128).reshape(-1)
    sps = int(round(dataset_config.sample_rate * dataset_config.symbol_time))
    bits_per_symbol = int(np.log2(dataset_config.constellation_order))
    cpfsk_uw = build_cpfsk_uw(dataset_config)

    if dataset_config.uw_spacing_bits is None:
        raise ValueError("uw_spacing_bits is required")

    correlation = UwDetector(
        method_type="differential_correlation",
        differential_lag_samples=sps,
    ).estimate(signal=rx_signal, cpfsk_uw=cpfsk_uw)

    uw_spacing_samples = int((dataset_config.uw_spacing_bits // bits_per_symbol) * sps)
    grid = find_best_uw_grid(
        correlation=correlation,
        uw_spacing=uw_spacing_samples,
        threshold=float(receiver_config.uw_detection_threshold),
        n_uw=expected_n_uw if expected_n_uw and expected_n_uw > 0 else None,
        spacing_tolerance=int(receiver_config.grid_spacing_tolerance),
        min_n_uw=1,
    )

    peak_to_start_offset = len(cpfsk_uw) - sps - 1
    leading_noise_samples = int(dataset_config.leading_noise_samples)
    detected_starts_samples = [int(idx - peak_to_start_offset) for idx in grid.uw_positions]
    detected_starts_bits = [
        int(round((start - leading_noise_samples) / sps) * bits_per_symbol)
        for start in detected_starts_samples
    ]

    frequency_estimator = FrequencyOffsetEstimator("differential_data_aided", reference_signal=cpfsk_uw)
    per_uw_cfo = [
        frequency_estimator.estimate(
            signal=extract_segment(rx_signal, start, len(cpfsk_uw)),
            sample_rate=dataset_config.sample_rate,
            reference_signal=cpfsk_uw,
        )
        for start in detected_starts_samples
    ]
    coarse_cfo = finite_mean(per_uw_cfo)

    corrected_signal = None
    per_uw_phase = []
    phase_offset = float("nan")
    if np.isfinite(coarse_cfo):
        corrected_signal = frequency_estimator.correct_frequency_offset(
            signal=rx_signal,
            frequency_offset_hz=coarse_cfo,
            sample_rate=dataset_config.sample_rate,
        )
        per_uw_phase = [
            estimate_phase(extract_segment(corrected_signal, start, len(cpfsk_uw)), cpfsk_uw)
            for start in detected_starts_samples
        ]
        phase_offset = circular_mean(per_uw_phase)

    semi_ber = float("nan")
    diff_ber = float("nan")
    fine_cfo = float("nan")
    fine_cfo_abs = float("nan")
    pll_trace = {}

    if corrected_signal is not None and detected_starts_samples:
        _, demodulator_config = build_configs(dataset_config)
        demodulator = Demodulator(demodulator_config)
        demod_start_sample = max(detected_starts_samples[0], 0)
        demod_start_bit = max(detected_starts_bits[0], 0)
        aligned_signal = corrected_signal[demod_start_sample:]
        initial_phase = float(per_uw_phase[0]) if per_uw_phase else 0.0

        semi_bits, pll_trace = demodulator.semi_coherent_demodulate(
            rx_signal=aligned_signal,
            symbol_time=dataset_config.symbol_time,
            sample_rate=dataset_config.sample_rate,
            frequency_sensitivity=dataset_config.h,
            initial_phase=initial_phase,
            initial_frequency_hz=0.0,
            return_trace=True,
        )
        diff_bits = demodulator.differentiate_demodulate(
            rx_signal=aligned_signal,
            symbol_time=dataset_config.symbol_time,
            sample_rate=dataset_config.sample_rate,
            frequency_sensitivity=dataset_config.h,
        )

        residual_frequency = np.asarray(pll_trace["residual_frequency_hz"], dtype=float)
        if len(residual_frequency):
            tail_fraction = min(max(float(receiver_config.pll_tail_fraction), 0.0), 1.0)
            tail = residual_frequency[int((1.0 - tail_fraction) * len(residual_frequency)):]
            fine_cfo = finite_mean(list(tail))
            fine_cfo_abs = finite_mean(list(np.abs(tail)))

        if tx_bits is not None:
            tx_bits_for_ber = np.asarray(tx_bits, dtype=np.uint8)[demod_start_bit:]
            semi_ber = ber(semi_bits, tx_bits_for_ber, bits_per_symbol)
            diff_ber = ber(diff_bits, tx_bits_for_ber, bits_per_symbol)

    return {
        "correlation": correlation,
        "peak_to_start_offset": peak_to_start_offset,
        "uw_peak_indices": grid.uw_positions,
        "uw_peak_scores": grid.peak_values,
        "uw_start_samples": detected_starts_samples,
        "uw_start_bits": detected_starts_bits,
        "grid_score": grid.score,
        "coarse_cfo_hz": coarse_cfo,
        "per_uw_cfo_hz": per_uw_cfo,
        "phase_offset_rad": phase_offset,
        "fine_cfo_hz": fine_cfo,
        "fine_cfo_abs_hz": fine_cfo_abs,
        "semi_ber": semi_ber,
        "diff_ber": diff_ber,
        "pll_trace": pll_trace,
    }


def run_dataset(config: DictConfig) -> list[dict[str, Any]]:
    dataset_config = DatasetConfig(**OmegaConf.to_container(config.dataset, resolve=True))
    dataset = Dataset(dataset_config).generate_dataset()
    rows = []

    for snr_db, samples in dataset.items():
        for signal_idx, sample in samples.items():
            result = run_receiver(
                rx_signal=sample["rx_signal"],
                tx_bits=sample["tx_bits"] if sample["n_uw"] > 0 else None,
                dataset_config=dataset_config,
                receiver_config=config.receiver,
                expected_n_uw=sample["n_uw"],
            )
            coarse_error = (
                result["coarse_cfo_hz"] - float(sample["frequency_offset"])
                if np.isfinite(result["coarse_cfo_hz"]) else float("nan")
            )
            rows.append({
                "snr_db": float(snr_db),
                "signal_idx": int(signal_idx),
                "n_uw": int(sample["n_uw"]),
                "true_uw_bits": list(sample["uw_start_bits"]),
                "true_uw_samples": list(sample["uw_start_samples"]),
                "true_uw_peaks": [
                    int(start + result["peak_to_start_offset"])
                    for start in sample["uw_start_samples"]
                ],
                "detected_uw_bits": result["uw_start_bits"],
                "detected_uw_samples": result["uw_start_samples"],
                "detected_uw_peaks": result["uw_peak_indices"],
                "peak_scores": result["uw_peak_scores"],
                "grid_score": result["grid_score"],
                "true_cfo_hz": float(sample["frequency_offset"]),
                "coarse_cfo_hz": result["coarse_cfo_hz"],
                "coarse_cfo_error_hz": coarse_error,
                "fine_cfo_hz": result["fine_cfo_hz"],
                "remaining_cfo_hz": (
                    coarse_error + result["fine_cfo_hz"]
                    if np.isfinite(coarse_error) and np.isfinite(result["fine_cfo_hz"])
                    else float("nan")
                ),
                "phase_offset_rad": result["phase_offset_rad"],
                "semi_ber": result["semi_ber"],
                "diff_ber": result["diff_ber"],
                "correlation": result["correlation"],
                "pll_trace": result["pll_trace"],
            })

    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[float, dict[str, float]]:
    summary = {}
    for snr in sorted({row["snr_db"] for row in rows}):
        snr_rows = [row for row in rows if row["snr_db"] == snr]
        signal_rows = [row for row in snr_rows if row["n_uw"] > 0]
        noise_rows = [row for row in snr_rows if row["n_uw"] == 0]
        detected_rows = [row for row in signal_rows if row["detected_uw_bits"]]
        exact_rows = [row for row in signal_rows if exact_list(row["detected_uw_bits"], row["true_uw_bits"])]

        summary[snr] = {
            "p_detect": finite_mean([float(bool(row["detected_uw_bits"])) for row in signal_rows], 0.0),
            "p_exact": finite_mean([float(exact_list(row["detected_uw_bits"], row["true_uw_bits"])) for row in signal_rows], 0.0),
            "p_false_alarm": finite_mean([float(bool(row["detected_uw_bits"])) for row in noise_rows], 0.0),
            "frame_fail": finite_mean([float(not row["detected_uw_bits"]) for row in signal_rows], 0.0),
            "coarse_cfo_mae": finite_mean([abs(row["coarse_cfo_error_hz"]) for row in exact_rows]),
            "remaining_cfo_abs": finite_mean([abs(row["remaining_cfo_hz"]) for row in exact_rows]),
            "semi_ber_detected": finite_mean([row["semi_ber"] for row in detected_rows]),
            "diff_ber_detected": finite_mean([row["diff_ber"] for row in detected_rows]),
            "semi_ber_effective": finite_mean([row["semi_ber"] if row["detected_uw_bits"] else 0.5 for row in signal_rows]),
            "diff_ber_effective": finite_mean([row["diff_ber"] if row["detected_uw_bits"] else 0.5 for row in signal_rows]),
            "n_detected": float(len(detected_rows)),
            "n_signal": float(len(signal_rows)),
        }
    return summary


def print_summary(summary: dict[float, dict[str, float]]) -> None:
    print("SNR | det | fail | exact | FA | CFO MAE | rem CFO | semi eff BER | diff eff BER | n det/sig")
    print("-" * 104)
    for snr, stats in summary.items():
        print(
            f"{snr:>3.0f} | {stats['p_detect']:>4.2f} | {stats['frame_fail']:>4.2f} | "
            f"{stats['p_exact']:>5.2f} | {stats['p_false_alarm']:>4.2f} | "
            f"{stats['coarse_cfo_mae']:>7.2f} | {stats['remaining_cfo_abs']:>7.2f} | "
            f"{stats['semi_ber_effective']:>12.4g} | {stats['diff_ber_effective']:>12.4g} | "
            f"{stats['n_detected']:>4.0f}/{stats['n_signal']:<4.0f}"
        )


def plot_summary(summary: dict[float, dict[str, float]]) -> None:
    snrs = np.asarray(sorted(summary), dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    axes[0].plot(snrs, [summary[s]["p_detect"] for s in snrs], "o-", label="detect")
    axes[0].plot(snrs, [summary[s]["p_exact"] for s in snrs], "o-", label="exact grid")
    axes[0].plot(snrs, [summary[s]["p_false_alarm"] for s in snrs], "o-", label="false alarm")
    axes[0].set_ylabel("probability")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(snrs, [summary[s]["coarse_cfo_mae"] for s in snrs], "o-", label="coarse CFO MAE")
    axes[1].plot(snrs, [summary[s]["remaining_cfo_abs"] for s in snrs], "o-", label="remaining CFO abs")
    axes[1].set_ylabel("Hz")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].semilogy(snrs, np.maximum([summary[s]["semi_ber_effective"] for s in snrs], 1e-8), "o-", label="semi effective BER")
    axes[2].semilogy(snrs, np.maximum([summary[s]["diff_ber_effective"] for s in snrs], 1e-8), "o-", label="diff effective BER")
    axes[2].set_xlabel("SNR [dB]")
    axes[2].set_ylabel("BER")
    axes[2].grid(True, which="both")
    axes[2].legend()
    plt.show()


def plot_correlations(rows: list[dict[str, Any]], max_examples: int) -> None:
    for row in rows[:max_examples]:
        if row["n_uw"] == 0 and not row["detected_uw_peaks"]:
            continue
        plt.figure(figsize=(11, 4))
        plt.plot(row["correlation"], label="correlation")
        for idx in row["true_uw_peaks"]:
            plt.axvline(idx, color="tab:green", linestyle="--", label="true UW")
        for idx in row["detected_uw_peaks"]:
            plt.axvline(idx, color="tab:red", linestyle=":", label="detected UW")
        plt.title(f"UW correlation | SNR={row['snr_db']:g} dB | n_uw={row['n_uw']}")
        plt.xlabel("correlation index")
        plt.ylabel("score")
        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys())
        plt.tight_layout()
        plt.show()


def plot_remaining_offsets(rows: list[dict[str, Any]]) -> None:
    rows = [row for row in rows if row["n_uw"] > 0 and row["detected_uw_bits"]]
    if not rows:
        return
    snr = [row["snr_db"] for row in rows]
    plt.figure(figsize=(10, 5))
    plt.scatter(snr, [row["coarse_cfo_error_hz"] for row in rows], s=18, alpha=0.6, label="coarse error")
    plt.scatter(snr, [row["remaining_cfo_hz"] for row in rows], s=18, alpha=0.6, label="remaining after PLL")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("SNR [dB]")
    plt.ylabel("Hz")
    plt.title("Remaining frequency offsets")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pll_examples(rows: list[dict[str, Any]], max_examples: int) -> None:
    examples = [row for row in rows if row["pll_trace"]][:max_examples]
    for row in examples:
        trace = row["pll_trace"]
        k = np.arange(len(trace["phase_errors_rad"]))
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        axes[0].plot(k, trace["phase_errors_rad"])
        axes[0].set_title(f"PLL phase error | SNR={row['snr_db']:g} dB | BER={row['semi_ber']:.4g}")
        axes[0].set_ylabel("rad")
        axes[0].grid(True)
        axes[1].plot(k, trace["residual_frequency_hz"])
        axes[1].set_title("PLL residual frequency state")
        axes[1].set_xlabel("symbol")
        axes[1].set_ylabel("Hz")
        axes[1].grid(True)
        plt.show()


@hydra.main(version_base=None, config_path=".", config_name="pipeline_config")
def main(config: DictConfig) -> None:
    rows = run_dataset(config)
    summary = summarize(rows)
    print_summary(summary)

    if config.plots.summary:
        plot_summary(summary)
    if config.plots.correlations:
        plot_correlations(rows, int(config.plots.max_correlation_examples))
    if config.plots.remaining_offsets:
        plot_remaining_offsets(rows)
    if config.plots.pll_examples:
        plot_pll_examples(rows, int(config.plots.max_pll_examples))


if __name__ == "__main__":
    main()
