import numpy as np
from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-cache"))

import matplotlib.pyplot as plt
import pydantic
from typing import Dict, List, Optional, Any

from demodulator_research.demodulator.demodulator import DemodulatorConfig, Demodulator
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.dataset_creation import DatasetConfig, Dataset
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig


class BerEvaluatorConfig(pydantic.BaseModel):
    sample_rate: float = 10_000
    symbol_time: float = 1e-3
    frequency_sensitivity: float = 0.5

    pulse_shape_type: str = "rect"
    span_in_symbols: int = 1
    pulse_normalization: str = "cpfsk"
    pulse_causal: bool = False

    constellation_type: str = "PAM"
    constellation_order: int = 4

    noise_type: str = "snr"

    demodulation_methods: List[str] = ["differentiate", "coherent"]
    viterbi_n_phase_states: int = 16
    viterbi_memory_depth_symbols: Optional[int] = None
    viterbi_beam_width: int = 512
    viterbi_use_abs_metric: bool = True
    viterbi_alpha: float = 0.05
    viterbi_beta: float = 0.02
    viterbi_decision_delay_symbols: Optional[int] = None
    viterbi_max_survivors: Optional[int] = None
    align_ber_by_symbol_delay: bool = True
    ber_edge_trim_symbols: Optional[int] = None


class BerEvaluator:
    def __init__(self, config: BerEvaluatorConfig):
        self.config = config

        self.bits_per_symbol = int(np.log2(self.config.constellation_order))
        self.sps = int(round(self.config.sample_rate * self.config.symbol_time))

        pulse_shape_config = PulseShapeConfig(pulse_shape_type=self.config.pulse_shape_type,
                                              normalization_type=self.config.pulse_normalization,
                                              span_in_symbols=self.config.span_in_symbols,
                                              pulse_causal=self.config.pulse_causal)

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

                kwargs = {
                    "rx_signal": rx_signal,
                    "symbol_time": self.config.symbol_time,
                    "sample_rate": self.config.sample_rate,
                    "frequency_sensitivity": self.config.frequency_sensitivity,
                }
                if method_name in ("viterbi", "cpfsk_viterbi"):
                    kwargs.update({
                        "n_phase_states": self.config.viterbi_n_phase_states,
                        "memory_depth_symbols": self.config.viterbi_memory_depth_symbols,
                        "beam_width": self.config.viterbi_beam_width,
                        "use_abs_metric": self.config.viterbi_use_abs_metric,
                        "alpha": self.config.viterbi_alpha,
                        "beta": self.config.viterbi_beta,
                        "decision_delay_symbols": self.config.viterbi_decision_delay_symbols,
                        "max_survivors": self.config.viterbi_max_survivors,
                    })
                return fn(**kwargs)

        raise AttributeError(f"Demodulator does not have method for '{method_name}'. "f"Tried: {candidate_names}")

    def _expected_bit_delay(self, method_name: str) -> int:
        """
        Expected deterministic bit delay from the TX/RX filtering.

        The modulator does:
            upsampled_symbols (*) pulse_shape, mode="same"

        The differentiate receiver does:
            diff(unwrap(angle(.))) (*) pulse_shape[::-1], mode="full"

        With mode="same", the TX pulse is centered on the symbol grid. The RX
        matched filter adds half the pulse delay, and diff(...) advances by one
        sample. The differentiate demodulator samples at center - 1, so the
        nominal bit delay is zero. Keeping this explicit avoids hiding timing
        assumptions in the BER calculation.
        """
        if method_name not in ("differentiate",):
            return 0

        pulse_shape = self.demodulator._pulse_shape_instance.generate_pulse_shape(
            self.config.sample_rate,
            self.config.symbol_time,
        )
        pulse_length = len(np.asarray(pulse_shape).reshape(-1))
        matched_filter_delay = (pulse_length - 1) / 2.0
        discriminator_advance = 1.0
        receiver_sampling_offset = (pulse_length - 1) // 2 - 1

        delay_samples = matched_filter_delay - discriminator_advance - receiver_sampling_offset
        delay_symbols = int(round(delay_samples / self.sps))
        return delay_symbols * self.bits_per_symbol

    def _alignment_search_window_bits(self) -> list[int]:
        max_symbol_shift = max(int(np.ceil(self.config.span_in_symbols)) + 2, 2)
        return [shift * self.bits_per_symbol for shift in range(-max_symbol_shift, max_symbol_shift + 1)]

    def _ber_edge_trim_bits(self, method_name: str) -> int:
        if method_name not in ("differentiate",):
            return 0
        if self.config.ber_edge_trim_symbols is not None:
            trim_symbols = int(self.config.ber_edge_trim_symbols)
        else:
            trim_symbols = max(1, int(np.ceil(float(self.config.span_in_symbols) / 2.0)))
        return max(0, trim_symbols) * self.bits_per_symbol

    @staticmethod
    def _calculate_ber_for_offset(tx_bits: np.ndarray, rx_bits: np.ndarray,
                                  bit_offset: int, edge_trim_bits: int = 0) -> tuple[float, int]:
        if bit_offset >= 0:
            tx_aligned = tx_bits[bit_offset:]
            rx_aligned = rx_bits
        else:
            tx_aligned = tx_bits
            rx_aligned = rx_bits[-bit_offset:]

        min_len = min(len(tx_aligned), len(rx_aligned))

        if min_len == 0:
            return 1.0, 0

        tx_aligned = np.asarray(tx_aligned[:min_len], dtype=np.uint8)
        rx_aligned = np.asarray(rx_aligned[:min_len], dtype=np.uint8)

        edge_trim_bits = int(edge_trim_bits)
        if edge_trim_bits > 0:
            if min_len <= 2 * edge_trim_bits:
                return 1.0, 0
            tx_aligned = tx_aligned[edge_trim_bits:-edge_trim_bits]
            rx_aligned = rx_aligned[edge_trim_bits:-edge_trim_bits]
            min_len = len(tx_aligned)

        n_errors = np.sum(tx_aligned != rx_aligned)
        ber = n_errors / min_len

        return float(ber), int(min_len)

    def _calculate_aligned_ber(self, tx_bits: np.ndarray, rx_bits: np.ndarray, method_name: str) -> tuple[float, int, int]:
        tx_bits = np.asarray(tx_bits, dtype=np.uint8).reshape(-1)
        rx_bits = np.asarray(rx_bits, dtype=np.uint8).reshape(-1)

        expected_delay = self._expected_bit_delay(method_name)
        edge_trim_bits = self._ber_edge_trim_bits(method_name)
        if not self.config.align_ber_by_symbol_delay:
            ber, n_bits = self._calculate_ber_for_offset(tx_bits, rx_bits, expected_delay, edge_trim_bits)
            return ber, expected_delay, n_bits

        best_ber = 1.0
        best_offset = expected_delay
        best_n_bits = 0

        for relative_offset in self._alignment_search_window_bits():
            candidate_offset = expected_delay + relative_offset
            ber, n_bits = self._calculate_ber_for_offset(tx_bits, rx_bits, candidate_offset, edge_trim_bits)
            if n_bits == 0:
                continue
            if ber < best_ber or (ber == best_ber and abs(candidate_offset - expected_delay) < abs(best_offset - expected_delay)):
                best_ber = ber
                best_offset = candidate_offset
                best_n_bits = n_bits

        return float(best_ber), int(best_offset), int(best_n_bits)

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
            alignment_trials_per_method = {method: [] for method in self.config.demodulation_methods}

            for signal_idx, sample in samples_for_snr.items():
                tx_bits = sample["tx_bits"]
                rx_signal = sample["rx_signal"]

                for method in self.config.demodulation_methods:
                    # estimated_frequency_offset = FrequencyOffsetEstimator(self.config.sample_rate).estimate(rx_signal)
                    # real_offset = dataset[noise_db][signal_idx]["frequency_offset"]
                    # print(f"estimated frequency offset: {estimated_frequency_offset}, the real frequency offset: {real_offset}")
                    # corrected_rx = FrequencyOffsetEstimator(self.config.sample_rate).correct_frequency_offset(rx_signal, estimated_frequency_offset)
                    rx_bits = self._call_demodulator(method_name=method, rx_signal=rx_signal)
                    ber, bit_offset, _n_bits = self._calculate_aligned_ber(tx_bits=tx_bits,
                                                                           rx_bits=rx_bits,
                                                                           method_name=method)

                    ber_trials_per_method[method].append(ber)
                    alignment_trials_per_method[method].append(bit_offset)

            for method in self.config.demodulation_methods:
                ber_values = np.asarray(ber_trials_per_method[method], dtype=float)

                results[method]["ber"][i] = np.mean(ber_values)
                results[method]["ber_std"][i] = np.std(ber_values)
                alignment_offsets = np.asarray(alignment_trials_per_method[method], dtype=float)

                sample_snr_db = (
                        float(noise_db) +
                        10.0 * np.log10(self.bits_per_symbol / self.sps)
                )
                print(
                    f"{self.config.noise_type.upper()}={noise_db:6.2f} dB | "
                    f"sample SNR~{sample_snr_db:7.2f} dB | "
                    f"method={method:15s} | "
                    f"BER={results[method]['ber'][i]:.6e} | "
                    f"align bits mean={np.mean(alignment_offsets):6.2f}"
                )

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
        sample_rate=10000,
        symbol_time=1e-3,
        n_bits_to_transmit=1000,

        min_frequency_offset=-0,
        max_frequency_offset=0,

        pulse_shape_type="rrc",
        span_in_symbols=10,
        pulse_normalization="cpfsk",
        pulse_causal=False,

        constellation_type="PAM",
        constellation_order=4,

        noise_type="snr",
        noise_db_values=list(np.arange(0, 20, 1)),

        n_signals_per_snr=100,

        use_uw=False,
        uw_bits=[1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    )

    dataset_generator = Dataset(dataset_cfg)
    dataset = dataset_generator.generate_dataset()

    evaluator_cfg = BerEvaluatorConfig(
        sample_rate=dataset_cfg.sample_rate,
        symbol_time=dataset_cfg.symbol_time,
        frequency_sensitivity=dataset_cfg.h,

        pulse_shape_type=dataset_cfg.pulse_shape_type,
        span_in_symbols=dataset_cfg.span_in_symbols,
        pulse_normalization=dataset_cfg.pulse_normalization,
        pulse_causal=dataset_cfg.pulse_causal,

        constellation_type=dataset_cfg.constellation_type,
        constellation_order=dataset_cfg.constellation_order,

        noise_type=dataset_cfg.noise_type,

        demodulation_methods=[
                              "differentiate",
                              # "viterbi"
                              ],
        viterbi_n_phase_states=16,
        viterbi_memory_depth_symbols=None,
        viterbi_beam_width=128,
        viterbi_use_abs_metric=True,
        viterbi_alpha=0.05,
        viterbi_beta=0.02,
        viterbi_decision_delay_symbols=None,
        viterbi_max_survivors=128,
        align_ber_by_symbol_delay=True,
    )

    evaluator = BerEvaluator(evaluator_cfg)

    ber_results = evaluator.compare_ber_vs_noise(dataset)

    evaluator.plot_ber_comparison(ber_results, title=f"{dataset_cfg.constellation_order}-PAM CPFSK: BER vs {dataset_cfg.noise_type.upper()}")
