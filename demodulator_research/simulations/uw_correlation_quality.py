from dataclasses import dataclass
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from demodulator_research.uw_detection.uw_detection import UwDetector
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.modulator.modulator import Modulator, ModulatorConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig, PulseShape
from synthetic_dataset_creation.coding.coding import Coding, CodingConfig


@dataclass
class UwCorrelationQualityConfig:
    sample_rate: float = 10_000
    symbol_time: float = 1e-3
    frequency_sensitivity: float = 0.5

    pulse_shape_type: str = "rect"
    span_in_symbols: int = 1
    pulse_normalization: str = "cpfsk"

    constellation_type: str = "PAM"
    constellation_order: int = 4

    n_payload_bits: int = 4000
    leading_noise_samples: int = 0
    n_trials_per_snr: int = 20
    random_seed: int = 0

    snr_db_values: tuple[float | None, ...] = (None, 20.0, 10.0, 5.0, 0.0)
    frequency_offsets_hz: tuple[float, ...] = (0.0, 200.0)

    uw_bits: tuple[int, ...] = (
        1, 0, 0, 1, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    )

    plot_correlations: bool = True
    max_lags_to_plot: int = 900


def add_awgn_by_snr(
        signal: np.ndarray,
        snr_db: float,
        rng: np.random.Generator,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128)
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )

    return signal + noise


def apply_frequency_offset(
        signal: np.ndarray,
        frequency_offset_hz: float,
        sample_rate: float,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    n = np.arange(len(signal))
    return signal * np.exp(1j * 2.0 * np.pi * frequency_offset_hz * n / sample_rate)


def build_modulator(config: UwCorrelationQualityConfig) -> Modulator:
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


def build_unmodulated_uw(
        uw_bits: np.ndarray,
        config: UwCorrelationQualityConfig,
) -> np.ndarray:
    pulse_shape_config = PulseShapeConfig(
        pulse_shape_type=config.pulse_shape_type,
        normalization_type=config.pulse_normalization,
        span_in_symbols=config.span_in_symbols,
    )
    constellation_config = ConstellationConfig(
        constellation_type=config.constellation_type,
        constellation_order=config.constellation_order,
    )

    coding = Coding(CodingConfig(constellation=constellation_config))
    constellation_points = np.arange(config.constellation_order) * 2 + 1 - config.constellation_order
    symbols = coding.generate_bits_to_symbols(
        bits_array=np.asarray(uw_bits, dtype=np.uint8),
        constellation_points=constellation_points,
    )

    sps = int(round(config.sample_rate * config.symbol_time))
    upsampled = np.zeros((len(symbols) - 1) * sps + 1, dtype=symbols.dtype)
    upsampled[::sps] = symbols

    pulse = PulseShape(pulse_shape_config).generate_pulse_shape(
        sample_rate=config.sample_rate,
        symbol_time=config.symbol_time,
    )

    return np.convolve(upsampled, pulse, mode="full")


def build_uw_references(
        config: UwCorrelationQualityConfig,
) -> dict[str, np.ndarray]:
    uw_bits = np.asarray(config.uw_bits, dtype=np.uint8)
    modulator = build_modulator(config)

    cpfsk_uw, _ = modulator.modulate(
        bits=np.array([], dtype=np.uint8),
        symbol_time=config.symbol_time,
        sample_rate=config.sample_rate,
        uw=uw_bits,
        frequency_offset=0.0,
        frequency_sensitivity=config.frequency_sensitivity,
    )

    return {
        "cpfsk_uw": cpfsk_uw,
        "unmodulated_uw": build_unmodulated_uw(
            uw_bits=uw_bits,
            config=config,
        ),
    }


def generate_trial_signal(
        config: UwCorrelationQualityConfig,
        rng: np.random.Generator,
        frequency_offset_hz: float,
        snr_db: float | None,
) -> dict[str, Any]:
    bits_per_symbol = int(np.log2(config.constellation_order))
    n_payload_bits = (config.n_payload_bits // bits_per_symbol) * bits_per_symbol
    payload_bits = rng.integers(0, 2, size=n_payload_bits, dtype=np.uint8)
    uw_bits = np.asarray(config.uw_bits, dtype=np.uint8)

    tx_signal, _ = build_modulator(config).modulate(
        bits=payload_bits,
        symbol_time=config.symbol_time,
        sample_rate=config.sample_rate,
        uw=uw_bits,
        frequency_offset=0.0,
        frequency_sensitivity=config.frequency_sensitivity,
    )

    rx_signal = apply_frequency_offset(
        signal=tx_signal,
        frequency_offset_hz=frequency_offset_hz,
        sample_rate=config.sample_rate,
    )

    if config.leading_noise_samples > 0:
        leading_noise = (
                rng.standard_normal(config.leading_noise_samples) +
                1j * rng.standard_normal(config.leading_noise_samples)
        ) / np.sqrt(2.0)
        rx_signal = np.concatenate((leading_noise, rx_signal))

    if snr_db is not None:
        rx_signal = add_awgn_by_snr(
            signal=rx_signal,
            snr_db=snr_db,
            rng=rng,
        )

    return {
        "rx_signal": rx_signal,
        "true_uw_start_sample": config.leading_noise_samples,
    }


def expected_correlation_index(
        method_name: str,
        config: UwCorrelationQualityConfig,
        references: dict[str, np.ndarray],
) -> int:
    if method_name == "complex_correlation":
        reference_length = len(references["cpfsk_uw"])
    elif method_name == "regular_correlation":
        reference_length = len(references["unmodulated_uw"]) - 1
    elif method_name == "differential_correlation":
        lag = int(round(config.sample_rate * config.symbol_time))
        reference_length = len(references["cpfsk_uw"]) - lag
    else:
        raise NotImplementedError

    # UwDetector.normalized_correlation currently uses full convolution.
    return config.leading_noise_samples + reference_length - 1


def largest_false_peak(
        correlation: np.ndarray,
        true_peak_index: int,
        guard_samples: int,
) -> float:
    mask = np.ones(len(correlation), dtype=bool)
    start = max(0, true_peak_index - guard_samples)
    stop = min(len(correlation), true_peak_index + guard_samples + 1)
    mask[start:stop] = False

    if not np.any(mask):
        return float("nan")

    return float(np.nanmax(correlation[mask]))


def summarize_correlation(
        correlation: np.ndarray,
        expected_index: int,
        guard_samples: int,
) -> dict[str, float]:
    correlation = np.asarray(correlation, dtype=float).reshape(-1)
    max_peak_index = int(np.nanargmax(correlation))
    max_peak = float(correlation[max_peak_index])

    true_peak = float(correlation[expected_index]) if 0 <= expected_index < len(correlation) else float("nan")
    false_peak = largest_false_peak(
        correlation=correlation,
        true_peak_index=expected_index,
        guard_samples=guard_samples,
    )

    return {
        "true_peak": true_peak,
        "max_peak": max_peak,
        "largest_false_peak": false_peak,
        "peak_margin": true_peak - false_peak,
        "max_peak_index": float(max_peak_index),
        "peak_index_error": float(max_peak_index - expected_index),
    }


def run_detectors(
        rx_signal: np.ndarray,
        references: dict[str, np.ndarray],
        config: UwCorrelationQualityConfig,
) -> dict[str, np.ndarray]:
    return {
        "complex_correlation": UwDetector("complex_correlation").estimate(
            signal=rx_signal,
            cpfsk_uw=references["cpfsk_uw"],
        ),
        "regular_correlation": UwDetector("regular_correlation").estimate(
            signal=rx_signal,
            cpfsk_uw=references["cpfsk_uw"],
        ),
        "differential_correlation": UwDetector(
            "differential_correlation",
            differential_lag_samples=int(round(config.sample_rate * config.symbol_time)),
        ).estimate(
            signal=rx_signal,
            cpfsk_uw=references["cpfsk_uw"],
        ),
    }


def evaluate_uw_detection_quality(
        config: UwCorrelationQualityConfig,
) -> dict:
    rng = np.random.default_rng(config.random_seed)
    references = build_uw_references(config)
    methods = [
        "complex_correlation",
        "regular_correlation",
        "differential_correlation",
    ]
    metrics = [
        "true_peak",
        "max_peak",
        "largest_false_peak",
        "peak_margin",
        "max_peak_index",
        "peak_index_error",
    ]

    guard_samples = int(round(config.sample_rate * config.symbol_time))
    results: dict[float, dict[float | None, dict[str, dict[str, float]]]] = {}
    example_correlations: dict[tuple[float, float | None], dict[str, np.ndarray]] = {}

    for frequency_offset_hz in config.frequency_offsets_hz:
        results[frequency_offset_hz] = {}

        for snr_db in config.snr_db_values:
            per_method_trials = {
                method_name: {metric: [] for metric in metrics}
                for method_name in methods
            }

            for trial_index in range(config.n_trials_per_snr):
                trial = generate_trial_signal(
                    config=config,
                    rng=rng,
                    frequency_offset_hz=frequency_offset_hz,
                    snr_db=snr_db,
                )

                correlations = run_detectors(
                    rx_signal=trial["rx_signal"],
                    references=references,
                    config=config,
                )

                if trial_index == 0:
                    example_correlations[(frequency_offset_hz, snr_db)] = correlations

                for method_name, correlation in correlations.items():
                    expected_index = expected_correlation_index(
                        method_name=method_name,
                        config=config,
                        references=references,
                    )
                    summary = summarize_correlation(
                        correlation=correlation,
                        expected_index=expected_index,
                        guard_samples=guard_samples,
                    )

                    for metric in metrics:
                        per_method_trials[method_name][metric].append(summary[metric])

            results[frequency_offset_hz][snr_db] = {
                method_name: {
                    metric: float(np.nanmean(per_method_trials[method_name][metric]))
                    for metric in metrics
                }
                for method_name in methods
            }

            print_summary_for_condition(
                frequency_offset_hz=frequency_offset_hz,
                snr_db=snr_db,
                condition_results=results[frequency_offset_hz][snr_db],
            )

    return {
        "results": results,
        "example_correlations": example_correlations,
        "references": references,
    }


def snr_label(snr_db: float | None) -> str:
    return "clean" if snr_db is None else f"{snr_db:g} dB"


def print_summary_for_condition(
        frequency_offset_hz: float,
        snr_db: float | None,
        condition_results: dict[str, dict[str, float]],
) -> None:
    print()
    print("=" * 118)
    print(f"CFO = {frequency_offset_hz:g} Hz | SNR = {snr_label(snr_db)}")
    print("=" * 118)
    print(
        f"{'method':<26} | "
        f"{'true peak':>10} | "
        f"{'max peak':>10} | "
        f"{'false peak':>10} | "
        f"{'margin':>10} | "
        f"{'idx error':>10}"
    )
    print("-" * 118)

    for method_name, method_result in condition_results.items():
        print(
            f"{method_name:<26} | "
            f"{method_result['true_peak']:>10.4f} | "
            f"{method_result['max_peak']:>10.4f} | "
            f"{method_result['largest_false_peak']:>10.4f} | "
            f"{method_result['peak_margin']:>10.4f} | "
            f"{method_result['peak_index_error']:>10.1f}"
        )


def plot_example_correlations(
        config: UwCorrelationQualityConfig,
        example_correlations: dict[tuple[float, float | None], dict[str, np.ndarray]],
        references: dict[str, np.ndarray],
) -> None:
    if not config.plot_correlations:
        return

    for (frequency_offset_hz, snr_db), correlations in example_correlations.items():
        for method_name, correlation in correlations.items():
            plt.figure(figsize=(11, 5))

            stop = min(config.max_lags_to_plot, len(correlation))
            plt.plot(
                np.arange(stop),
                correlation[:stop],
                label=method_name,
            )

            expected_index = expected_correlation_index(
                method_name=method_name,
                config=config,
                references=references,
            )
            if expected_index < config.max_lags_to_plot:
                plt.axvline(expected_index, linestyle="--", label="expected UW peak")

            plt.grid(True)
            plt.xlabel("Correlation index")
            plt.ylabel("Normalized correlation magnitude")
            plt.title(
                f"{method_name} | "
                f"CFO={frequency_offset_hz:g} Hz | "
                f"SNR={snr_label(snr_db)}"
            )
            plt.legend()
            plt.tight_layout()
            plt.show()


def plot_metric_vs_snr(
        config: UwCorrelationQualityConfig,
        results: dict,
        metric: str,
) -> None:
    for frequency_offset_hz, results_for_cfo in results.items():
        plt.figure(figsize=(9, 5))

        x_values = [
            100.0 if snr_db is None else float(snr_db)
            for snr_db in config.snr_db_values
        ]

        for method_name in next(iter(results_for_cfo.values())).keys():
            y_values = [
                results_for_cfo[snr_db][method_name][metric]
                for snr_db in config.snr_db_values
            ]
            plt.plot(x_values, y_values, marker="o", label=method_name)

        plt.grid(True)
        plt.xlabel("SNR [dB] (clean shown as 100 dB)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs SNR | CFO={frequency_offset_hz:g} Hz")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    config = UwCorrelationQualityConfig(
        snr_db_values=(2.0, 1.0, 0),
        frequency_offsets_hz=(250.0, 0),
        n_trials_per_snr=5,
    )

    output = evaluate_uw_detection_quality(config)

    plot_example_correlations(
        config=config,
        example_correlations=output["example_correlations"],
        references=output["references"],
    )

    plot_metric_vs_snr(
        config=config,
        results=output["results"],
        metric="peak_margin",
    )

    plot_metric_vs_snr(
        config=config,
        results=output["results"],
        metric="largest_false_peak",
    )
