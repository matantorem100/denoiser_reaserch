import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Callable

from demodulator_research.frequency_offset_estimation.frequency_offset_estimation import (
    FrequencyOffsetEstimator,
)

from synthetic_dataset_creation.dataset_creation import DatasetConfig, Dataset


def normalize_to_unit_magnitude(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a complex signal to unit magnitude.

    This is useful for differential phase estimation because we care mostly
    about phase rotation, not amplitude variations.
    """
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    return signal / np.maximum(np.abs(signal), eps)


def estimate_frequency_offset_diff_blind_circular_mean_hz(
    rx_signal: np.ndarray,
    sample_rate: float,
    normalize_amplitude: bool = True,
) -> float:
    """
    Blind differential frequency offset estimator using circular mean.

    Computes:

        d[n] = rx[n] * conj(rx[n-1])

    Then estimates the average phase increment using:

        angle(mean(d[n]))

    Finally converts phase increment per sample to Hz:

        f_hat = angle(mean(d)) * Fs / (2*pi)

    This is a circular/statistical mean of the differential phasors.
    """

    rx = np.asarray(rx_signal, dtype=np.complex128).reshape(-1)

    if len(rx) < 2:
        return 0.0

    if normalize_amplitude:
        rx = normalize_to_unit_magnitude(rx)

    differential_rx = rx[1:] * np.conj(rx[:-1])

    mean_differential_phasor = np.mean(differential_rx)

    phase_increment_rad = np.angle(mean_differential_phasor)

    estimated_offset_hz = phase_increment_rad * sample_rate / (2.0 * np.pi)

    return float(estimated_offset_hz)


def estimate_frequency_offset_diff_blind_linear_mean_hz(
    rx_signal: np.ndarray,
    sample_rate: float,
    normalize_amplitude: bool = True,
) -> float:
    """
    Blind differential frequency offset estimator using linear mean.

    Computes:

        d[n] = rx[n] * conj(rx[n-1])

    Then estimates phase increments using:

        phase_increment[n] = angle(d[n])

    Then averages the phase increments directly:

        mean_phase_increment = mean(angle(d[n]))

    Finally converts to Hz:

        f_hat = mean_phase_increment * Fs / (2*pi)

    For CPFSK, this estimator is often very intuitive because angle(d[n])
    is basically the instantaneous frequency in radians/sample.
    """

    rx = np.asarray(rx_signal, dtype=np.complex128).reshape(-1)

    if len(rx) < 2:
        return 0.0

    if normalize_amplitude:
        rx = normalize_to_unit_magnitude(rx)

    differential_rx = rx[1:] * np.conj(rx[:-1])

    phase_increments_rad = np.angle(differential_rx)

    mean_phase_increment_rad = np.mean(phase_increments_rad)

    estimated_offset_hz = mean_phase_increment_rad * sample_rate / (2.0 * np.pi)

    return float(estimated_offset_hz)


def build_frequency_estimation_methods(
    estimator: FrequencyOffsetEstimator,
    sample_rate: float,
) -> Dict[str, Callable[[np.ndarray], float]]:
    """
    Register the frequency-estimation methods you want to compare.

    All methods receive only rx_signal.

    Methods:
        coarse:
            Your existing estimator.coarse_estimate_hz method.

        diff_blind_circular_mean:
            Blind differential estimator based on angle(mean(d[n])).

        diff_blind_linear_mean:
            Blind differential estimator based on mean(angle(d[n])).
    """

    methods: Dict[str, Callable[[np.ndarray], float]] = {}

    methods["coarse"] = estimator.coarse_estimate_hz

    methods["diff_blind_circular_mean"] = lambda rx_signal: (
        estimate_frequency_offset_diff_blind_circular_mean_hz(
            rx_signal=rx_signal,
            sample_rate=sample_rate,
            normalize_amplitude=True,
        )
    )

    methods["diff_blind_linear_mean"] = lambda rx_signal: (
        estimate_frequency_offset_diff_blind_linear_mean_hz(
            rx_signal=rx_signal,
            sample_rate=sample_rate,
            normalize_amplitude=True,
        )
    )

    return methods


def evaluate_remaining_frequency_offset_vs_snr(
    dataset: Dict[float, Dict[int, Dict[str, Any]]],
    sample_rate: float,
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

    estimator = FrequencyOffsetEstimator(sample_rate=sample_rate)

    methods = build_frequency_estimation_methods(
        estimator=estimator,
        sample_rate=sample_rate,
    )

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

            for method_name, method_fn in methods.items():
                estimated_offset_hz = float(method_fn(rx_signal))

                # The correction itself would be:
                # corrected_signal = estimator.correct_frequency_offset(
                #     signal=rx_signal,
                #     frequency_offset_hz=estimated_offset_hz,
                # )

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

    results = evaluate_remaining_frequency_offset_vs_snr(
        dataset=dataset,
        sample_rate=dataset_cfg.sample_rate,
    )

    plot_remaining_frequency_offset_results(
        results=results,
        const_frequency_offset=const_frequency_offset,

        # Set these flags however you want:
        plot_combined=False,
        plot_separate=True,
    )