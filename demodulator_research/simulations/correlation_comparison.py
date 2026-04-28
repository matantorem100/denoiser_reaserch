import numpy as np
import matplotlib.pyplot as plt
import pydantic
from typing import Dict, List, Any, Optional

from demodulator_research.demodulator.demodulator import DemodulatorConfig, Demodulator
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.dataset_creation import DatasetConfig, Dataset
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig

# Import the Dataset code we built before
# from your_module import Dataset, DatasetConfig


class UwCorrelationEvaluatorConfig(pydantic.BaseModel):
    sample_rate: float = 10_000
    symbol_time: float = 1e-3

    pulse_shape_type: str = "rect"
    span_in_symbols: int = 1
    pulse_normalization: str = "cpfsk"

    constellation_type: str = "PAM"
    constellation_order: int = 4

    noise_type: str = "snr"

    demodulation_methods: List[str] = [
        "differentiate",
        "coherent",
    ]


class UwCorrelationEvaluator:
    def __init__(self, config: UwCorrelationEvaluatorConfig):
        self.config = config

        self.sps = int(round(config.sample_rate * config.symbol_time))
        self.bits_per_symbol = int(np.log2(config.constellation_order))

        pulse_shape_config = PulseShapeConfig(
            pulse_shape_type=config.pulse_shape_type,
            normalization_type=config.pulse_normalization,
            span_in_symbols=config.span_in_symbols,
        )

        constellation_config = ConstellationConfig(
            constellation_type=config.constellation_type,
            constellation_order=config.constellation_order,
        )

        demodulator_config = DemodulatorConfig(
            pulse_shape_config=pulse_shape_config,
            constellation_config=constellation_config,
        )

        self.demodulator = Demodulator(demodulator_config)

    def _call_demodulator(self, method_name: str, rx_signal: np.ndarray) -> np.ndarray:
        candidate_names = [
            f"{method_name}_demodulate",
            method_name,
        ]

        for candidate in candidate_names:
            if hasattr(self.demodulator, candidate):
                fn = getattr(self.demodulator, candidate)

                return fn(
                    rx_signal=rx_signal,
                    symbol_time=self.config.symbol_time,
                    sample_rate=self.config.sample_rate,
                )

        raise AttributeError(
            f"Demodulator does not have method '{method_name}'. "
            f"Tried: {candidate_names}"
        )

    @staticmethod
    def _phase_diff(x: np.ndarray) -> np.ndarray:
        """
        diff(unwrap(angle(x)))
        """
        return np.diff(np.unwrap(np.angle(x)))

    @staticmethod
    def _bits_to_bipolar(bits: np.ndarray) -> np.ndarray:
        """
        Convert bits from {0,1} to {-1,+1}.
        This is much better for correlation than raw 0/1 bits.
        """
        bits = np.asarray(bits).astype(np.float64)
        return 2.0 * bits - 1.0

    @staticmethod
    def _normalized_valid_correlation(
        signal: np.ndarray,
        uw: np.ndarray,
        use_abs: bool = True,
    ) -> Dict[str, Any]:

        signal = np.asarray(signal)
        uw = np.asarray(uw)

        if len(signal) < len(uw):
            return {
                "true_peak": np.nan,
                "max_peak": np.nan,
                "largest_false_peak": np.nan,
                "peak_margin": np.nan,
                "max_peak_index": -1,
                "corr": np.array([]),
            }

        corr = np.correlate(signal, np.conj(uw[::-1])[::-1], mode="valid")

        # Equivalent and clearer:
        corr = np.array([
            np.vdot(uw, signal[i:i + len(uw)])
            for i in range(len(signal) - len(uw) + 1)
        ])

        uw_energy = np.linalg.norm(uw)

        sliding_energy = np.sqrt(
            np.convolve(
                np.abs(signal) ** 2,
                np.ones(len(uw)),
                mode="valid",
            )
        )

        denom = uw_energy * sliding_energy
        denom = np.maximum(denom, 1e-12)

        norm_corr = corr / denom

        if use_abs:
            metric_corr = np.abs(norm_corr)
        else:
            metric_corr = np.real(norm_corr)

        true_peak = float(metric_corr[0])
        max_peak = float(np.max(metric_corr))
        max_peak_index = int(np.argmax(metric_corr))

        if len(metric_corr) > 1:
            largest_false_peak = float(np.max(metric_corr[1:]))
        else:
            largest_false_peak = float("-inf")

        peak_margin = true_peak - largest_false_peak

        return {
            "true_peak": true_peak,
            "max_peak": max_peak,
            "largest_false_peak": largest_false_peak,
            "peak_margin": peak_margin,
            "max_peak_index": max_peak_index,
            "corr": metric_corr,
        }

    def evaluate_dataset(
        self,
        dataset: Dict[float, Dict[int, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, np.ndarray]]:

        noise_db_values = np.asarray(sorted(dataset.keys()), dtype=float)

        result_names = [
            "fm_waveform",
            "phase_diff",
        ]

        for method in self.config.demodulation_methods:
            result_names.append(f"bits_{method}")

        results = {}

        for name in result_names:
            results[name] = {
                "noise_db": noise_db_values,
                "true_peak": np.zeros(len(noise_db_values)),
                "max_peak": np.zeros(len(noise_db_values)),
                "largest_false_peak": np.zeros(len(noise_db_values)),
                "peak_margin": np.zeros(len(noise_db_values)),
                "max_peak_index": np.zeros(len(noise_db_values)),
            }

        for i, noise_db in enumerate(noise_db_values):
            samples_for_snr = dataset[float(noise_db)]

            metric_trials = {
                name: {
                    "true_peak": [],
                    "max_peak": [],
                    "largest_false_peak": [],
                    "peak_margin": [],
                    "max_peak_index": [],
                }
                for name in result_names
            }

            for signal_idx, sample in samples_for_snr.items():
                rx_signal = sample["rx_signal"]
                tx_signal = sample["tx_signal"]
                uw_bits = np.asarray(sample["uw_bits"], dtype=np.uint8)

                n_uw_symbols = len(uw_bits) // self.bits_per_symbol
                n_uw_samples = n_uw_symbols * self.sps

                clean_uw_fm = tx_signal[:n_uw_samples]

                # -------------------------------------------------
                # 1. Correlation directly on FM-modulated waveform
                # -------------------------------------------------
                fm_metrics = self._normalized_valid_correlation(
                    signal=rx_signal,
                    uw=clean_uw_fm,
                    use_abs=True,
                )

                for key in metric_trials["fm_waveform"]:
                    metric_trials["fm_waveform"][key].append(fm_metrics[key])

                # -------------------------------------------------
                # 2. Correlation after diff(unwrap(angle()))
                # -------------------------------------------------
                rx_phase_diff = self._phase_diff(rx_signal)
                uw_phase_diff = self._phase_diff(clean_uw_fm)

                phase_metrics = self._normalized_valid_correlation(
                    signal=rx_phase_diff,
                    uw=uw_phase_diff,
                    use_abs=True,
                )

                for key in metric_trials["phase_diff"]:
                    metric_trials["phase_diff"][key].append(phase_metrics[key])

                # -------------------------------------------------
                # 3. Correlation after demodulation, on bits
                # -------------------------------------------------
                uw_bits_bipolar = self._bits_to_bipolar(uw_bits)

                for method in self.config.demodulation_methods:
                    rx_bits = self._call_demodulator(
                        method_name=method,
                        rx_signal=rx_signal,
                    )

                    rx_bits = np.asarray(rx_bits, dtype=np.uint8)
                    rx_bits_bipolar = self._bits_to_bipolar(rx_bits)

                    bit_metrics = self._normalized_valid_correlation(
                        signal=rx_bits_bipolar,
                        uw=uw_bits_bipolar,
                        use_abs=True,
                    )

                    result_key = f"bits_{method}"

                    for key in metric_trials[result_key]:
                        metric_trials[result_key][key].append(bit_metrics[key])

            for result_key in result_names:
                for metric_name in metric_trials[result_key]:
                    values = np.asarray(metric_trials[result_key][metric_name], dtype=float)
                    results[result_key][metric_name][i] = np.nanmean(values)

            print(f"Finished SNR = {noise_db:.2f} dB")

        return results

    def plot_metric(
        self,
        results: Dict[str, Dict[str, np.ndarray]],
        metric_name: str = "peak_margin",
        title: Optional[str] = None,
    ) -> None:

        plt.figure(figsize=(9, 5))

        for name, result in results.items():
            plt.plot(
                result["noise_db"],
                result[metric_name],
                marker="o",
                label=name,
            )

        plt.grid(True)
        plt.xlabel(f"{self.config.noise_type.upper()} [dB]")
        plt.ylabel(metric_name)
        plt.legend()

        if title is None:
            title = f"UW normalized correlation: {metric_name}"

        plt.title(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    dataset_cfg = DatasetConfig(
        sample_rate=10_000,
        symbol_time=1e-3,
        n_bits_to_transmit=100_000,

        min_frequency_offset=-0,
        max_frequency_offset=0,

        pulse_shape_type="rect",
        span_in_symbols=1,
        pulse_normalization="cpfsk",

        constellation_type="PAM",
        constellation_order=4,

        noise_type="snr",
        noise_db_values=list(np.arange(-5, 15, 2)),

        n_signals_per_snr=3,

        uw_bits=[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    )

    dataset_generator = Dataset(dataset_cfg)
    dataset = dataset_generator.generate_dataset()

    evaluator_cfg = UwCorrelationEvaluatorConfig(
        sample_rate=dataset_cfg.sample_rate,
        symbol_time=dataset_cfg.symbol_time,

        pulse_shape_type=dataset_cfg.pulse_shape_type,
        span_in_symbols=dataset_cfg.span_in_symbols,
        pulse_normalization=dataset_cfg.pulse_normalization,

        constellation_type=dataset_cfg.constellation_type,
        constellation_order=dataset_cfg.constellation_order,

        noise_type=dataset_cfg.noise_type,

        demodulation_methods=[
            "differentiate",
            "coherent",
            # "non_coherent",
            # "pll",
        ],
    )

    evaluator = UwCorrelationEvaluator(evaluator_cfg)

    uw_results = evaluator.evaluate_dataset(dataset)

    evaluator.plot_metric(
        uw_results,
        metric_name="true_peak",
        title="UW normalized correlation: true peak",
    )

    evaluator.plot_metric(
        uw_results,
        metric_name="largest_false_peak",
        title="UW normalized correlation: largest false peak",
    )

    evaluator.plot_metric(
        uw_results,
        metric_name="peak_margin",
        title="UW normalized correlation: true peak - largest false peak",
    )

    evaluator.plot_metric(
        uw_results,
        metric_name="max_peak",
        title="UW normalized correlation: max peak",
    )