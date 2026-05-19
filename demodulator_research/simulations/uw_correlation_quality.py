from dataclasses import dataclass
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

    n_bits: int = 2000
    n_signals_per_condition: int = 1
    snr_db_values: tuple[float, ...] = (0, 2, 4, 6, 8, 10)
    frequency_offsets_hz: tuple[float, ...] = (-500, 0, 500)

    uw_bits: tuple[int, ...] = (
        1, 0, 0, 1, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    )
    uw_spacing_bits: int = 500
    leading_noise_samples: int = 1000

    detection_threshold: float = 0.45
    random_seed: int = 0

    n_context_symbols_before_crop: int = 6
    n_context_symbols_after_crop: int = 6
    n_random_context_trials: int = 200


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


def build_cpfsk_uw(config: Config) -> np.ndarray:
    return modulate_bits(config, np.asarray(config.uw_bits, dtype=np.uint8))


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

    for snr_db in config.snr_db_values:
        for frequency_offset_hz in config.frequency_offsets_hz:
            dataset_config = DatasetConfig(
                sample_rate=config.sample_rate,
                symbol_time=config.symbol_time,
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
                noise_type="snr",
                noise_db_values=[float(snr_db)],
                n_signals_per_snr=config.n_signals_per_condition,
                uw_bits=list(config.uw_bits),
                use_uw=True,
                uw_probability=1.0,
                n_uw_values=[3, 3],
                uw_spacing_bits=config.uw_spacing_bits,
                first_uw_after_noise=True,
                leading_noise_samples=config.leading_noise_samples,
                noise_only_when_no_uw=True,
                random_uw_start=False,
                uw_mode="overwrite",
                random_seed=int(rng.integers(0, 2**31 - 1)),
            )
            dataset[(float(snr_db), float(frequency_offset_hz))] = Dataset(dataset_config).generate_dataset()[float(snr_db)]

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
        score = float(np.max(correlation))

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
    fig, axes = plt.subplots(len(correlations), 1, figsize=(12, 2.8 * len(correlations)), constrained_layout=True)
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
    plt.show()


if __name__ == "__main__":
    config = Config()
    dataset = create_dataset(config)
    cpfsk_uw = build_cpfsk_uw(config)
    cpfsk_uw_middle, uw_crop_start_sample, uw_crop_stop_sample = crop_stable_uw_middle(config, cpfsk_uw)
    print(
        f"middle UW crop: start={uw_crop_start_sample}, stop={uw_crop_stop_sample}, "
        f"length={len(cpfsk_uw_middle)} samples"
    )

    rng = np.random.default_rng(config.random_seed)
    for (snr_db, frequency_offset_hz), samples in dataset.items():
        sample_index = int(rng.choice(list(samples.keys())))
        sample = samples[sample_index]
        correlations = run_all_correlation_methods(
            rx_signal=sample["rx_signal"],
            config=config,
            cpfsk_uw=cpfsk_uw,
            rng=rng,
        )
        plot_correlations_and_uw_positions(
            config=config,
            sample=sample,
            correlations=correlations,
            cpfsk_uw=cpfsk_uw,
            title=f"SNR={snr_db:g} dB | CFO={frequency_offset_hz:g} Hz | n_uw={sample['n_uw']}",
        )
