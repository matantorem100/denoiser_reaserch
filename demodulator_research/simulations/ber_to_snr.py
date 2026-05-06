import numpy as np
import matplotlib.pyplot as plt
import pydantic
from typing import Dict, List, Optional, Any

from demodulator_research.demodulator.demodulator import DemodulatorConfig, Demodulator
from demodulator_research.frequency_offset_estimation.frequency_offset_estimation import FrequencyOffsetEstimator
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.dataset_creation import DatasetConfig, Dataset
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig


class BerEvaluatorConfig(pydantic.BaseModel):
    sample_rate: float = 10_000
    symbol_time: float = 1e-3

    pulse_shape_type: str = "rect"
    span_in_symbols: int = 1
    pulse_normalization: str = "cpfsk"

    constellation_type: str = "PAM"
    constellation_order: int = 4

    noise_type: str = "snr"

    demodulation_methods: List[str] = ["differentiate", "coherent"]


class BerEvaluator:
    def __init__(self, config: BerEvaluatorConfig):
        self.config = config

        self.bits_per_symbol = int(np.log2(self.config.constellation_order))
        self.sps = int(round(self.config.sample_rate * self.config.symbol_time))

        pulse_shape_config = PulseShapeConfig(pulse_shape_type=self.config.pulse_shape_type,
                                              normalization_type=self.config.pulse_normalization,
                                              span_in_symbols=self.config.span_in_symbols)

        constellation_config = ConstellationConfig(constellation_type=self.config.constellation_type,
                                                   constellation_order=self.config.constellation_order)

        demodulator_config = DemodulatorConfig(pulse_shape_config=pulse_shape_config,
                                               constellation_config=constellation_config)

        self.demodulator = Demodulator(demodulator_config)

    def _call_demodulator(self, method_name: str, rx_signal: np.ndarray) -> np.ndarray:
        """
        Tries:
            demodulator.differentiate_demodulate(...)
            demodulator.differentiate(...)
        """

        candidate_names = [f"{method_name}_demodulate", method_name]

        for candidate in candidate_names:
            if hasattr(self.demodulator, candidate):
                fn = getattr(self.demodulator, candidate)

                return fn(rx_signal=rx_signal, symbol_time=self.config.symbol_time, sample_rate=self.config.sample_rate)

        raise AttributeError(f"Demodulator does not have method for '{method_name}'. "f"Tried: {candidate_names}")

    @staticmethod
    def _calculate_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
        min_len = min(len(tx_bits), len(rx_bits))

        if min_len == 0:
            return 1.0

        tx_bits = np.asarray(tx_bits[:min_len], dtype=np.uint8)
        rx_bits = np.asarray(rx_bits[:min_len], dtype=np.uint8)

        n_errors = np.sum(tx_bits != rx_bits)
        ber = n_errors / min_len

        return float(ber)

    def compare_ber_vs_noise(self, dataset: Dict[float, Dict[int, Dict[str, Any]]]) -> Dict[str, Dict[str, np.ndarray]]:

        noise_db_values = np.asarray(sorted(dataset.keys()), dtype=float)

        results: Dict[str, Dict[str, np.ndarray]] = {}

        for method in self.config.demodulation_methods:
            results[method] = {
                "noise_db": noise_db_values,
                "ber": np.zeros(len(noise_db_values), dtype=float),
                "ber_std": np.zeros(len(noise_db_values), dtype=float),
            }

        for i, noise_db in enumerate(noise_db_values):
            samples_for_snr = dataset[float(noise_db)]

            ber_trials_per_method = {method: [] for method in self.config.demodulation_methods}

            for signal_idx, sample in samples_for_snr.items():
                tx_bits = sample["tx_bits"]
                rx_signal = sample["rx_signal"]

                for method in self.config.demodulation_methods:
                    estimated_frequency_offset = FrequencyOffsetEstimator(self.config.sample_rate).estimate(rx_signal)
                    real_offset = dataset[noise_db][signal_idx]["frequency_offset"]
                    print(f"estimated frequency offset: {estimated_frequency_offset}, the real frequency offset: {real_offset}")
                    corrected_rx = FrequencyOffsetEstimator(self.config.sample_rate).correct_frequency_offset(rx_signal, estimated_frequency_offset)
                    rx_bits = self._call_demodulator(method_name=method, rx_signal=rx_signal)
                    ber = self._calculate_ber(tx_bits=tx_bits, rx_bits=rx_bits)

                    ber_trials_per_method[method].append(ber)

            for method in self.config.demodulation_methods:
                ber_values = np.asarray(ber_trials_per_method[method], dtype=float)

                results[method]["ber"][i] = np.mean(ber_values)
                results[method]["ber_std"][i] = np.std(ber_values)

                print(f"SNR={noise_db:6.2f} dB | " f"method={method:15s} | " f"BER={results[method]['ber'][i]:.6e}")

        return results

    def plot_ber_comparison(self, ber_results: Dict[str, Dict[str, np.ndarray]], title: Optional[str] = None) -> None:

        plt.figure(figsize=(8, 5))

        for method, result in ber_results.items():
            y = np.maximum(result["ber"], 1e-8)

            plt.semilogy(result["noise_db"], y, marker="o", label=method)

        plt.grid(True, which="both")
        plt.xlabel(f"{self.config.noise_type.upper()} [dB]")
        plt.ylabel("BER")
        plt.legend()

        if title is None:
            title = f"BER vs {self.config.noise_type.upper()}"

        plt.title(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    dataset_cfg = DatasetConfig(
        sample_rate=10_0000,
        symbol_time=1e-3,
        n_bits_to_transmit=100000,

        min_frequency_offset=-0,
        max_frequency_offset=0,

        pulse_shape_type="rrc",
        span_in_symbols=10,
        pulse_normalization="cpfsk",

        constellation_type="PAM",
        constellation_order=4,

        noise_type="snr",
        noise_db_values=list(np.arange(0, 20, 2)),

        n_signals_per_snr=3,

        uw_bits=[1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    )

    dataset_generator = Dataset(dataset_cfg)
    dataset = dataset_generator.generate_dataset()

    evaluator_cfg = BerEvaluatorConfig(
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
                              # "semi_coherent",
                              # "coherent_fm_viterbi"
                              ],
    )

    evaluator = BerEvaluator(evaluator_cfg)

    ber_results = evaluator.compare_ber_vs_noise(dataset)

    evaluator.plot_ber_comparison(ber_results, title=f"{dataset_cfg.constellation_order}-PAM CPFSK: BER vs {dataset_cfg.noise_type.upper()}")
