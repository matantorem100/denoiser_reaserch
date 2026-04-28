import numpy as np
import matplotlib.pyplot as plt
import pydantic
from typing import Optional, Dict, List, Any

from demodulator_research.frequency_offset_estimation.frequency_offset_estimation import (
    FrequencyOffsetEstimator,
)
from demodulator_research.demodulator.demodulator import DemodulatorConfig, Demodulator
from synthetic_dataset_creation.dataset_creation import (
    SignalDatasetConfig,
    SignalDatasetGenerator,
)
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.modulator.modulator import FMConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig


class SimulationConfig(pydantic.BaseModel):
    sample_rate: float = 10_000
    symbol_time: float = 1e-3

    n_signals_per_snr: int = 1
    n_bits_per_signal: int = 100_000

    apply_fm: bool = True
    h: float = 0.5

    random_frequency_offset: bool = True
    frequency_offset_min: float = -100.0
    frequency_offset_max: float = 100.0
    fixed_frequency_offset: float = 0.0

    pulse_shape_type: str = "rect"
    pulse_normalization: str = "cpfsk"

    constellation_type: str = "PAM"
    constellation_order: int = 4

    noise_type: str = "snr"
    noise_db_values: List[float] = list(np.arange(0, 20, 2))

    uw_n_bits: int = 32  # PAM4: 32 bits = 16 UW symbols

    random_seed: Optional[int] = 1234

    demodulation_methods: List[str] = ["differentiate"]

    estimator_coarse_search_width_hz: float = 200.0
    estimator_n_grid_points: int = 201


class CommunicationEvaluator:
    def __init__(self, config: SimulationConfig):
        self.config = config

        self.sps = int(round(config.sample_rate * config.symbol_time))

        self.bits_per_symbol = int(np.log2(config.constellation_order))

        if config.uw_n_bits % self.bits_per_symbol != 0:
            raise ValueError("uw_n_bits must be divisible by bits_per_symbol")

        self.dataset_generator = self._build_dataset_generator()

        self.estimator = FrequencyOffsetEstimator(sample_rate=config.sample_rate)

        self.demodulator = self._build_demodulator()

    def _build_dataset_generator(self) -> SignalDatasetGenerator:
        dataset_cfg = SignalDatasetConfig(
            sample_rate=self.config.sample_rate,
            symbol_time=self.config.symbol_time,

            n_signals_per_snr=self.config.n_signals_per_snr,
            n_bits_per_signal=self.config.n_bits_per_signal,

            apply_fm=self.config.apply_fm,
            h=self.config.h,

            uw_n_bits=self.config.uw_n_bits,

            random_frequency_offset=self.config.random_frequency_offset,
            frequency_offset_min=self.config.frequency_offset_min,
            frequency_offset_max=self.config.frequency_offset_max,
            fixed_frequency_offset=self.config.fixed_frequency_offset,

            pulse_shape_type=self.config.pulse_shape_type,
            pulse_normalization=self.config.pulse_normalization,

            constellation_type=self.config.constellation_type,
            constellation_order=self.config.constellation_order,

            noise_type=self.config.noise_type,
            noise_db_values=self.config.noise_db_values,
        )

        return SignalDatasetGenerator(dataset_cfg)


    def _build_demodulator(self) -> Demodulator:
        pulse_shape_config = PulseShapeConfig(pulse_shape_type=self.config.pulse_shape_type,
                                              normalization_type=self.config.pulse_normalization)

        constellation_config = ConstellationConfig(constellation_type=self.config.constellation_type,
                                                   constellation_order=self.config.constellation_order)

        fm_config = None
        if self.config.apply_fm:
            fm_config = FMConfig(frequency_offset=0.0, frequency_sensitivity=self.config.h)

        demodulator_config = DemodulatorConfig(pulse_shape_config=pulse_shape_config,
                                               constellation_config=constellation_config, fm_config=fm_config)

        return Demodulator(demodulator_config)

    def _call_demodulator(self, method_name: str, rx_signal: np.ndarray) -> np.ndarray:
        candidate_names = [f"{method_name}_demodulate", method_name]

        for candidate in candidate_names:
            if hasattr(self.demodulator, candidate):
                fn = getattr(self.demodulator, candidate)
                return fn(rx_signal=rx_signal, symbol_time=self.config.symbol_time, sample_rate=self.config.sample_rate)

        raise AttributeError(f"Demodulator does not have method '{method_name}'. Tried {candidate_names}")

    @staticmethod
    def _compute_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
        min_len = min(len(tx_bits), len(rx_bits))
        if min_len == 0:
            return 1.0

        return float(np.mean(tx_bits[:min_len] != rx_bits[:min_len]))


    def _estimate_and_correct(self, rx_signal: np.ndarray, clean_uw_signal: np.ndarray) -> Dict[str, Any]:
        estimated_cfo = self.estimator.estimate(signal=rx_signal, uw=clean_uw_signal)

        corrected_signal = self.estimator.correct_frequency_offset(rx_signal, estimated_cfo)

        corr = self.estimator.calculate_normalized_correlation(corrected_signal, clean_uw_signal)

        abs_corr = np.abs(corr)
        timing_index = int(np.argmax(abs_corr))
        corr_score = float(abs_corr[timing_index])
        print(corr_score)


        return {
            "estimated_cfo": estimated_cfo,
            "timing_index": timing_index,
            "corr_score": corr_score,
            "corrected_signal": corrected_signal,
        }

    def evaluate(self) -> Dict[str, Dict[str, np.ndarray]]:
        dataset = self.dataset_generator.generate_dataset()

        results: Dict[str, Dict[str, np.ndarray]] = {}

        for method in self.config.demodulation_methods:
            results[method] = {
                "noise_db": np.asarray(self.config.noise_db_values, dtype=float),
                "ber": np.zeros(len(self.config.noise_db_values), dtype=float),
                "cfo_mae": np.zeros(len(self.config.noise_db_values), dtype=float),
                "cfo_rmse": np.zeros(len(self.config.noise_db_values), dtype=float),
                "cfo_bias": np.zeros(len(self.config.noise_db_values), dtype=float),
                "corr_score": np.zeros(len(self.config.noise_db_values), dtype=float),
                "timing_index": np.zeros(len(self.config.noise_db_values), dtype=float),
            }

        for i, noise_db in enumerate(self.config.noise_db_values):
            samples = dataset[float(noise_db)]

            per_method_ber = {method: [] for method in self.config.demodulation_methods}
            cfo_errors = []
            corr_scores = []
            timing_indices = []

            for sample in samples:
                rx_signal = sample["noised_signal"]
                tx_bits = sample["bits"]
                true_cfo = float(sample["frequency_offset"])
                clean_uw_signal = sample["clean_uw"]

                correction_result = self._estimate_and_correct(rx_signal=rx_signal, clean_uw_signal=clean_uw_signal)

                estimated_cfo = correction_result["estimated_cfo"]
                corrected_signal = correction_result["corrected_signal"]

                cfo_errors.append(estimated_cfo - true_cfo)
                corr_scores.append(correction_result["corr_score"])
                timing_indices.append(correction_result["timing_index"])

                for method in self.config.demodulation_methods:
                    rx_bits = self._call_demodulator(method_name=method, rx_signal=corrected_signal)
                    ber = self._compute_ber(tx_bits, rx_bits)
                    per_method_ber[method].append(ber)

            cfo_errors = np.asarray(cfo_errors)
            corr_scores = np.asarray(corr_scores)
            timing_indices = np.asarray(timing_indices)

            for method in self.config.demodulation_methods:
                results[method]["ber"][i] = np.mean(per_method_ber[method])
                results[method]["cfo_mae"][i] = np.mean(np.abs(cfo_errors))
                results[method]["corr_score"][i] = np.mean(corr_scores)
                results[method]["timing_index"][i] = np.mean(timing_indices)

        return results

    def plot_results(self, results: Dict[str, Dict[str, np.ndarray]]) -> None:
        plt.figure(figsize=(8, 5))
        for method, r in results.items():
            plt.semilogy(r["noise_db"], np.maximum(r["ber"], 1e-6), marker="o", label=method)

        plt.grid(True, which="both")
        plt.xlabel(f"{self.config.noise_type} [dB]")
        plt.ylabel("BER")
        plt.title("BER after CFO estimation and correction")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        for method, r in results.items():
            plt.plot(r["noise_db"], r["cfo_mae"], marker="o", label=f"{method} MAE")
            plt.plot(r["noise_db"], r["cfo_rmse"], marker="x", label=f"{method} RMSE")

        plt.grid(True)
        plt.xlabel(f"{self.config.noise_type} [dB]")
        plt.ylabel("CFO error [Hz]")
        plt.title("CFO estimation error")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        for method, r in results.items():
            plt.plot(r["noise_db"], r["corr_score"], marker="o", label=method)

        plt.grid(True)
        plt.xlabel(f"{self.config.noise_type} [dB]")
        plt.ylabel("Max normalized UW correlation")
        plt.title("UW correlation peak after CFO grid search")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        for method, r in results.items():
            plt.plot(r["noise_db"], r["timing_index"], marker="o", label=method)

        plt.grid(True)
        plt.xlabel(f"{self.config.noise_type} [dB]")
        plt.ylabel("Estimated UW start index")
        plt.title("Detected UW timing index")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    cfg = SimulationConfig(
        sample_rate=10_000,
        symbol_time=1e-3,

        n_signals_per_snr=1,
        n_bits_per_signal=100_000,

        apply_fm=True,
        h=0.5,

        random_frequency_offset=True,
        frequency_offset_min=-500,
        frequency_offset_max=500,

        pulse_shape_type="rrc",
        pulse_normalization="cpfsk",

        constellation_type="PAM",
        constellation_order=4,

        noise_type="snr",
        noise_db_values=list(np.arange(0, 30, 0.5)),

        # PAM4: 32 bits = 16 UW symbols
        uw_n_bits=32,

        random_seed=1234,

        demodulation_methods=[
            "differentiate",
        ],
    )

    evaluator = CommunicationEvaluator(cfg)
    results = evaluator.evaluate()

    for method, r in results.items():
        print(f"\nMethod: {method}")
        for i, snr_db in enumerate(r["noise_db"]):
            print(
                f"SNR={snr_db:>5.1f} dB | "
                f"BER={r['ber'][i]:.6f} | "
                f"CFO_MAE={r['cfo_mae'][i]:.3f} Hz | "
                f"CFO_RMSE={r['cfo_rmse'][i]:.3f} Hz | "
                f"CFO_Bias={r['cfo_bias'][i]:.3f} Hz | "
                f"Corr={r['corr_score'][i]:.4f} | "
                f"Timing={r['timing_index'][i]:.1f}"
            )

    evaluator.plot_results(results)