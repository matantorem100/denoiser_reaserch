import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from demodulator_research.frequency_offset_estimation.frequency_offset_estimation import (
    FrequencyOffsetEstimator,
)

from synthetic_dataset_creation.dataset_creation import DatasetConfig, Dataset
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.modulator.modulator import ModulatorConfig, Modulator
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig


def build_frequency_estimation_methods(
    reference_signal: np.ndarray | None = None,
) -> Dict[str, FrequencyOffsetEstimator]:
    """
    Register the frequency-estimation methods you want to compare.

    Method names must match FrequencyOffsetEstimator.method_type.
    """

    methods = {
        "phase_diff_coarse": FrequencyOffsetEstimator(method_type="phase_diff_coarse"),
        "differential_circular_coarse": FrequencyOffsetEstimator(method_type="differential_circular_coarse"),
    }

    if reference_signal is not None:
        methods["differential_data_aided"] = FrequencyOffsetEstimator(
            method_type="differential_data_aided",
            reference_signal=reference_signal,
        )

    return methods


def evaluate_remaining_frequency_offset_vs_snr(
    dataset: Dict[float, Dict[int, Dict[str, Any]]],
    sample_rate: float,
    reference_signal: np.ndarray | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compares frequency-correction methods by measuring:

        residual_hz = true_offset_hz - estimated_offset_hz

    Metrics per SNR:
        - mean residual signed error
        - mean absolute residual error
        - residual standard deviation
        - RMSE residual
        - mean estimated offset
        - mean true offset
    """

    methods = build_frequency_estimation_methods(reference_signal=reference_signal)

    snr_values = np.asarray(sorted(dataset.keys()), dtype=float)

    results: Dict[str, Dict[str, np.ndarray]] = {}

    for method_name in methods:
        results[method_name] = {
            "snr_db": snr_values,
            "mean_residual_hz": np.zeros(len(snr_values), dtype=float),
            "mean_abs_residual_hz": np.zeros(len(snr_values), dtype=float),
            "std_residual_hz": np.zeros(len(snr_values), dtype=float),
            "rmse_residual_hz": np.zeros(len(snr_values), dtype=float),
            "mean_estimated_offset_hz": np.zeros(len(snr_values), dtype=float),
            "mean_true_offset_hz": np.zeros(len(snr_values), dtype=float),
        }

    for i, snr_db in enumerate(snr_values):
        samples_for_snr = dataset[float(snr_db)]

        residuals_per_method = {method_name: [] for method_name in methods}
        estimates_per_method = {method_name: [] for method_name in methods}
        true_offsets = []

        for signal_idx, sample in samples_for_snr.items():
            rx_signal = np.asarray(sample["rx_signal"], dtype=np.complex128).reshape(-1)
            true_offset_hz = float(sample["frequency_offset"])

            true_offsets.append(true_offset_hz)

            for method_name, estimator in methods.items():
                estimated_offset_hz = float(estimator.estimate(rx_signal, sample_rate))

                _corrected_signal = estimator.correct_frequency_offset(
                    signal=rx_signal,
                    frequency_offset_hz=estimated_offset_hz,
                    sample_rate=sample_rate,
                )

                # But since this is simulation, the residual is known analytically:
                residual_hz = true_offset_hz - estimated_offset_hz

                residuals_per_method[method_name].append(residual_hz)
                estimates_per_method[method_name].append(estimated_offset_hz)

        true_offsets = np.asarray(true_offsets, dtype=float)

        print(f"\nSNR = {snr_db:.2f} dB")

        for method_name in methods:
            residuals = np.asarray(residuals_per_method[method_name], dtype=float)
            estimates = np.asarray(estimates_per_method[method_name], dtype=float)

            results[method_name]["mean_residual_hz"][i] = np.mean(residuals)
            results[method_name]["mean_abs_residual_hz"][i] = np.mean(np.abs(residuals))
            results[method_name]["std_residual_hz"][i] = np.std(residuals)
            results[method_name]["rmse_residual_hz"][i] = np.sqrt(np.mean(residuals**2))
            results[method_name]["mean_estimated_offset_hz"][i] = np.mean(estimates)
            results[method_name]["mean_true_offset_hz"][i] = np.mean(true_offsets)

            print(
                f"  method={method_name:26s} | "
                f"mean abs residual={results[method_name]['mean_abs_residual_hz'][i]:9.4f} Hz | "
                f"RMSE={results[method_name]['rmse_residual_hz'][i]:9.4f} Hz | "
                f"bias={results[method_name]['mean_residual_hz'][i]:9.4f} Hz | "
                f"std={results[method_name]['std_residual_hz'][i]:9.4f} Hz | "
                f"mean estimate={results[method_name]['mean_estimated_offset_hz'][i]:9.4f} Hz"
            )

    return results


def plot_metric_combined(
    results: Dict[str, Dict[str, np.ndarray]],
    metric_key: str,
    ylabel: str,
    title: str,
    true_value_hz: float | None = None,
) -> None:
    """
    Plot all methods together in one graph for a specific metric.
    """

    plt.figure(figsize=(9, 5))

    for method_name, result in results.items():
        plt.plot(
            result["snr_db"],
            result[metric_key],
            marker="o",
            label=method_name,
        )

    if true_value_hz is not None:
        plt.axhline(true_value_hz, linestyle="--", label="true offset")

    plt.grid(True)
    plt.xlabel("SNR [dB]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_metric_separate_per_method(
    results: Dict[str, Dict[str, np.ndarray]],
    metric_key: str,
    ylabel: str,
    title_prefix: str,
    true_value_hz: float | None = None,
) -> None:
    """
    Plot each method in a separate figure for a specific metric.
    """

    for method_name, result in results.items():
        plt.figure(figsize=(8, 5))

        plt.plot(
            result["snr_db"],
            result[metric_key],
            marker="o",
            label=method_name,
        )

        if true_value_hz is not None:
            plt.axhline(true_value_hz, linestyle="--", label="true offset")

        plt.grid(True)
        plt.xlabel("SNR [dB]")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix} - {method_name}")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_remaining_frequency_offset_results(
    results: Dict[str, Dict[str, np.ndarray]],
    const_frequency_offset: float,
    plot_combined: bool = True,
    plot_separate: bool = True,
) -> None:
    """
    Plot frequency-offset estimation results.

    You can control whether to plot:
        - combined graphs with all methods together
        - separate graphs, one per method

    Metrics:
        1. Mean absolute residual frequency offset
        2. RMSE residual frequency offset
        3. Mean signed residual / bias
        4. Mean estimated frequency offset
    """

    metrics_to_plot = [
        {
            "metric_key": "mean_abs_residual_hz",
            "ylabel": "Mean absolute remaining frequency offset [Hz]",
            "title": (
                f"Mean Absolute Remaining Frequency Offset vs SNR "
                f"- true offset = {const_frequency_offset} Hz"
            ),
            "true_value_hz": None,
        },
        {
            "metric_key": "rmse_residual_hz",
            "ylabel": "RMSE remaining frequency offset [Hz]",
            "title": (
                f"RMSE Remaining Frequency Offset vs SNR "
                f"- true offset = {const_frequency_offset} Hz"
            ),
            "true_value_hz": None,
        },
        {
            "metric_key": "mean_residual_hz",
            "ylabel": "Mean signed remaining offset [Hz]",
            "title": "Bias of Remaining Frequency Offset vs SNR",
            "true_value_hz": 0.0,
        },
        {
            "metric_key": "mean_estimated_offset_hz",
            "ylabel": "Mean estimated frequency offset [Hz]",
            "title": "Estimated Frequency Offset vs SNR",
            "true_value_hz": const_frequency_offset,
        },
    ]

    for metric in metrics_to_plot:
        if plot_combined:
            plot_metric_combined(
                results=results,
                metric_key=metric["metric_key"],
                ylabel=metric["ylabel"],
                title=metric["title"] + " - combined",
                true_value_hz=metric["true_value_hz"],
            )

        if plot_separate:
            plot_metric_separate_per_method(
                results=results,
                metric_key=metric["metric_key"],
                ylabel=metric["ylabel"],
                title_prefix=metric["title"],
                true_value_hz=metric["true_value_hz"],
            )


if __name__ == "__main__":

    uw_bits = [
        1, 0, 0, 1, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    ]

    const_frequency_offset = 500

    dataset_cfg = DatasetConfig(
        sample_rate=10_000,
        symbol_time=1e-3,
        n_bits_to_transmit=100000,

        min_frequency_offset=-0,
        max_frequency_offset=0,
        const_frequency_offset=const_frequency_offset,

        pulse_shape_type="rect",
        span_in_symbols=1,
        pulse_normalization="cpfsk",

        constellation_type="PAM",
        constellation_order=4,

        noise_type="snr",
        noise_db_values=list(np.arange(-10, 22, 2)),

        # Increase this for smoother curves.
        n_signals_per_snr=20,

        uw_bits=uw_bits,
    )

    dataset_generator = Dataset(dataset_cfg)
    dataset = dataset_generator.generate_dataset()

    pulse_shape_config = PulseShapeConfig(
        pulse_shape_type=dataset_cfg.pulse_shape_type,
        normalization_type=dataset_cfg.pulse_normalization,
        span_in_symbols=dataset_cfg.span_in_symbols,
    )
    constellation_config = ConstellationConfig(
        constellation_type=dataset_cfg.constellation_type,
        constellation_order=dataset_cfg.constellation_order,
    )
    modulator = Modulator(
        ModulatorConfig(
            pulse_shape_config=pulse_shape_config,
            constellation_config=constellation_config,
        )
    )
    reference_uw_signal, _ = modulator.modulate(
        bits=np.array([], dtype=np.uint8),
        symbol_time=dataset_cfg.symbol_time,
        sample_rate=dataset_cfg.sample_rate,
        uw=np.asarray(uw_bits, dtype=np.uint8),
        frequency_offset=0.0,
        frequency_sensitivity=dataset_cfg.h,
    )

    results = evaluate_remaining_frequency_offset_vs_snr(
        dataset=dataset,
        sample_rate=dataset_cfg.sample_rate,
        reference_signal=reference_uw_signal,
    )

    plot_remaining_frequency_offset_results(
        results=results,
        const_frequency_offset=const_frequency_offset,

        # Set these flags however you want:
        plot_combined=False,
        plot_separate=True,
    )
