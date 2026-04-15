import numpy as np
import matplotlib.pyplot as plt
import pydantic
from typing import Optional, Dict, List

from synthetic_dataset_creation.channel.channel import ChannelConfig, Channel
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from demodulator_research.demodulator.demodulator import DemodulatorConfig, Demodulator
from synthetic_dataset_creation.modulator.modulator import ModulatorConfig, Modulator, FMConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig


class SimulationConfig(pydantic.BaseModel):
    sample_rate: float = 10000
    symbol_time: float = 1e-3
    n_bits_to_transmit: int = 100000

    apply_fm: bool = True
    frequency_offset: float = 0.0
    h: float = 0.5

    pulse_shape_type: str = "rect"
    pulse_normalization: str = "cpfsk"

    constellation_type: str = "PAM"
    constellation_order: int = 4

    noise_type: str = "snr"   # "snr" or "eb_to_n0"
    noise_db_values: List[float] = list(np.arange(0, 20, 2))

    n_trials_per_point: int = 3
    random_seed: Optional[int] = 1234

    # methods to compare. These must exist in your Demodulator class.
    demodulation_methods: List[str] = ["differentiate"]

    # UW used for correlation peak tests
    uw_bits: List[int] = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]


class CommunicationEvaluator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.sps = int(round(self.config.sample_rate * self.config.symbol_time))

        if self.sps <= 0:
            raise ValueError("sps must be positive")

        self._validate_config()

        pulse_shape_config = PulseShapeConfig(
            pulse_shape_type=self.config.pulse_shape_type,
            normalization_type=self.config.pulse_normalization
        )

        constellation_config = ConstellationConfig(
            constellation_type=self.config.constellation_type,
            constellation_order=self.config.constellation_order
        )

        fm_config = None
        if self.config.apply_fm:
            fm_config = FMConfig(
                frequency_offset=self.config.frequency_offset,
                frequency_sensitivity=self.config.h
            )

        modulator_config = ModulatorConfig(
            pulse_shape_config=pulse_shape_config,
            constellation_config=constellation_config,
            fm_config=fm_config
        )

        demodulator_config = DemodulatorConfig(
            pulse_shape_config=pulse_shape_config,
            constellation_config=constellation_config,
            fm_config=fm_config
        )

        self.modulator = Modulator(modulator_config)
        self.demodulator = Demodulator(demodulator_config)

        self.bits_per_symbol = int(np.log2(self.config.constellation_order))
        self.rng = np.random.default_rng(self.config.random_seed)

        self.uw_bits = np.asarray(self.config.uw_bits, dtype=np.uint8)

    def _validate_config(self) -> None:
        if self.config.apply_fm and self.config.pulse_normalization != "cpfsk":
            raise ValueError("For FM/CPFSK, pulse_normalization must be 'cpfsk'")

        if (not self.config.apply_fm) and self.config.pulse_normalization != "norm_2":
            raise ValueError("For linear modulation, pulse_normalization must be 'norm_2'")

        if self.config.noise_type not in {"snr", "eb_to_n0"}:
            raise ValueError("noise_type must be 'snr' or 'eb_to_n0'")

        if self.config.n_trials_per_point <= 0:
            raise ValueError("n_trials_per_point must be positive")

        if len(self.config.demodulation_methods) == 0:
            raise ValueError("demodulation_methods must not be empty")

    def _random_bits(self, n_bits: int) -> np.ndarray:
        return self.rng.integers(0, 2, size=n_bits, dtype=np.uint8)

    def _build_channel(self, noise_db: float) -> Channel:
        channel_config = ChannelConfig(
            channel_type="awgn",
            bits_per_symbol=self.bits_per_symbol,
            sps=self.sps,
            noise_type=self.config.noise_type,
            noise_db=float(noise_db)
        )
        return Channel(channel_config)

    def _call_demodulator(self, method_name: str, rx_signal: np.ndarray) -> np.ndarray:
        """
        Calls demodulator.<method_name>_demodulate(...) if it exists,
        otherwise demodulator.<method_name>(...) if it exists.
        """
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
                    apply_fm=self.config.apply_fm
                )

        raise AttributeError(
            f"Demodulator does not have a method for '{method_name}'. "
            f"Tried: {candidate_names}"
        )

    def _generate_one_trial(self, noise_db: float, uw: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Generate one common transmitted sequence and one common noisy received waveform.
        All demodulation methods are tested on the exact same rx signal.
        """
        bits = self._random_bits(self.config.n_bits_to_transmit)

        tx_signal, tx_bits = self.modulator.modulate(
            bits=bits,
            symbol_time=self.config.symbol_time,
            sample_rate=self.config.sample_rate,
            apply_fm=self.config.apply_fm,
            uw=uw
        )

        channel = self._build_channel(noise_db)
        rx_signal = channel.transmit(tx_signal)

        return {
            "tx_bits": tx_bits,
            "tx_signal": tx_signal,
            "rx_signal": rx_signal,
        }

    @staticmethod
    def _compute_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
        min_len = min(len(tx_bits), len(rx_bits))
        if min_len == 0:
            return 1.0

        bit_errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
        return bit_errors / min_len

    @staticmethod
    def _uw_normalized_correlation(rx_bits: np.ndarray, uw_bits: np.ndarray) -> Dict[str, float]:
        if uw_bits is None or len(uw_bits) == 0:
            raise ValueError("uw_bits must be provided and non-empty")

        if len(rx_bits) < len(uw_bits):
            return {
                "true_peak": -1.0,
                "max_peak": -1.0,
                "largest_false_peak": -1.0,
                "peak_margin": -1.0,
                "max_peak_index": -1,
            }

        uw_bipolar = 2.0 * uw_bits.astype(np.float64) - 1.0
        rx_bipolar = 2.0 * rx_bits.astype(np.float64) - 1.0

        L = len(uw_bipolar)

        corr = np.correlate(rx_bipolar, uw_bipolar, mode="valid")

        uw_energy = np.sum(uw_bipolar ** 2)
        rx_energy = np.convolve(rx_bipolar ** 2, np.ones(L, dtype=np.float64), mode="valid")

        denom = np.sqrt(rx_energy * uw_energy)
        denom = np.maximum(denom, 1e-12)

        normalized_corr = corr / denom

        true_peak = float(normalized_corr[0])
        max_peak = float(np.max(normalized_corr))
        max_peak_index = int(np.argmax(normalized_corr))

        if len(normalized_corr) > 1:
            sidelobes = normalized_corr[1:]
            largest_false_peak = float(np.max(sidelobes))
        else:
            largest_false_peak = float("-inf")

        peak_margin = true_peak - largest_false_peak

        return {
            "true_peak": true_peak,
            "max_peak": max_peak,
            "largest_false_peak": largest_false_peak,
            "peak_margin": peak_margin,
            "max_peak_index": max_peak_index,
        }

    def compare_ber_vs_noise(self, uw: Optional[np.ndarray] = None) -> Dict[str, Dict[str, np.ndarray]]:
        results: Dict[str, Dict[str, np.ndarray]] = {}

        for method in self.config.demodulation_methods:
            results[method] = {
                "noise_db": np.asarray(self.config.noise_db_values, dtype=float),
                "ber": np.zeros(len(self.config.noise_db_values), dtype=float),
            }

        for i, noise_db in enumerate(self.config.noise_db_values):
            ber_trials_per_method = {method: [] for method in self.config.demodulation_methods}

            for _ in range(self.config.n_trials_per_point):
                common_trial = self._generate_one_trial(noise_db=noise_db, uw=uw)
                tx_bits = common_trial["tx_bits"]
                rx_signal = common_trial["rx_signal"]

                for method in self.config.demodulation_methods:
                    rx_bits = self._call_demodulator(method, rx_signal)
                    ber = self._compute_ber(tx_bits, rx_bits)
                    ber_trials_per_method[method].append(ber)

            for method in self.config.demodulation_methods:
                results[method]["ber"][i] = np.mean(ber_trials_per_method[method])

        return results

    def compare_uw_metrics_vs_noise(self, uw: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        if uw is None or len(uw) == 0:
            raise ValueError("uw must be provided")

        results: Dict[str, Dict[str, np.ndarray]] = {}

        for method in self.config.demodulation_methods:
            results[method] = {
                "noise_db": np.asarray(self.config.noise_db_values, dtype=float),
                "true_peak": np.zeros(len(self.config.noise_db_values), dtype=float),
                "largest_false_peak": np.zeros(len(self.config.noise_db_values), dtype=float),
                "peak_margin": np.zeros(len(self.config.noise_db_values), dtype=float),
                "max_peak": np.zeros(len(self.config.noise_db_values), dtype=float),
            }

        for i, noise_db in enumerate(self.config.noise_db_values):
            metric_trials = {
                method: {
                    "true_peak": [],
                    "largest_false_peak": [],
                    "peak_margin": [],
                    "max_peak": [],
                }
                for method in self.config.demodulation_methods
            }

            for _ in range(self.config.n_trials_per_point):
                common_trial = self._generate_one_trial(noise_db=noise_db, uw=uw)
                rx_signal = common_trial["rx_signal"]

                for method in self.config.demodulation_methods:
                    rx_bits = self._call_demodulator(method, rx_signal)
                    corr_metrics = self._uw_normalized_correlation(rx_bits, uw)

                    metric_trials[method]["true_peak"].append(corr_metrics["true_peak"])
                    metric_trials[method]["largest_false_peak"].append(corr_metrics["largest_false_peak"])
                    metric_trials[method]["peak_margin"].append(corr_metrics["peak_margin"])
                    metric_trials[method]["max_peak"].append(corr_metrics["max_peak"])

            for method in self.config.demodulation_methods:
                results[method]["true_peak"][i] = np.mean(metric_trials[method]["true_peak"])
                results[method]["largest_false_peak"][i] = np.mean(metric_trials[method]["largest_false_peak"])
                results[method]["peak_margin"][i] = np.mean(metric_trials[method]["peak_margin"])
                results[method]["max_peak"][i] = np.mean(metric_trials[method]["max_peak"])

        return results

    def plot_ber_comparison(self, ber_results: Dict[str, Dict[str, np.ndarray]], title: Optional[str] = None) -> None:
        plt.figure(figsize=(8, 5))

        for method, result in ber_results.items():
            y = np.maximum(result["ber"], 1e-6)
            plt.semilogy(result["noise_db"], y, marker="o", label=method)

        plt.grid(True, which="both")
        plt.xlabel(f"{self.config.noise_type} [dB]")
        plt.ylabel("BER")
        plt.legend()

        if title is None:
            title = f"BER vs {self.config.noise_type.upper()}"

        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_uw_metric_comparison(
        self,
        uw_results: Dict[str, Dict[str, np.ndarray]],
        metric_name: str,
        ylabel: str,
        title: Optional[str] = None
    ) -> None:
        plt.figure(figsize=(8, 5))

        for method, result in uw_results.items():
            plt.plot(result["noise_db"], result[metric_name], marker="o", label=method)

        plt.grid(True)
        plt.xlabel(f"{self.config.noise_type} [dB]")
        plt.ylabel(ylabel)
        plt.legend()

        if title is None:
            title = f"{metric_name} vs {self.config.noise_type.upper()}"

        plt.title(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    cfg = SimulationConfig(
        sample_rate=10000,
        symbol_time=1e-3,
        n_bits_to_transmit=100000,
        apply_fm=True,
        frequency_offset=0.0,
        h=0.5,
        pulse_shape_type="rect",
        pulse_normalization="cpfsk",
        constellation_type="PAM",
        constellation_order=4,
        noise_type="snr",
        noise_db_values=list(np.arange(0, 20, 2)),
        n_trials_per_point=3,
        random_seed=1234,

        # Put here the methods that exist in your Demodulator class
        demodulation_methods=[
            "differentiate",
            "coherent",
            "non_coherent"
            # "pll",
        ],

        uw_bits=[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0],
    )

    evaluator = CommunicationEvaluator(cfg)

    # BER comparison
    ber_results = evaluator.compare_ber_vs_noise(uw=None)
    evaluator.plot_ber_comparison(
        ber_results,
        title=f"{cfg.constellation_order}-PAM CPFSK: BER vs {cfg.noise_type.upper()}"
    )

    # UW comparison
    uw = np.asarray(cfg.uw_bits, dtype=np.uint8)

    uw_results = evaluator.compare_uw_metrics_vs_noise(uw)

    evaluator.plot_uw_metric_comparison(
        uw_results,
        metric_name="true_peak",
        ylabel="Normalized true UW peak",
        title=f"{cfg.constellation_order}-PAM CPFSK: true UW peak vs {cfg.noise_type.upper()}"
    )

    evaluator.plot_uw_metric_comparison(
        uw_results,
        metric_name="largest_false_peak",
        ylabel="Largest false normalized peak",
        title=f"{cfg.constellation_order}-PAM CPFSK: false UW peak vs {cfg.noise_type.upper()}"
    )

    evaluator.plot_uw_metric_comparison(
        uw_results,
        metric_name="peak_margin",
        ylabel="True peak - largest false peak",
        title=f"{cfg.constellation_order}-PAM CPFSK: UW peak margin vs {cfg.noise_type.upper()}"
    )