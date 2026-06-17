import numpy as np
import pydantic

from demodulator_research.pipeline.demod.demod import CpfskViterbiConfig, cpfsk_viterbi_demodulate
from synthetic_dataset_creation.coding.coding import Coding, CodingConfig
from synthetic_dataset_creation.constellation.constellation import Constellation, ConstellationConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShape, PulseShapeConfig


TWO_PI = 2.0 * np.pi


def wrap_phase(phase: float | np.ndarray) -> float | np.ndarray:
    return np.angle(np.exp(1j * phase))


def differentiate(rx_signal: np.ndarray, sample_rate: float, frequency_sensitivity: float) -> np.ndarray:
    """
    FM discriminator:
        instantaneous_frequency ~= diff(unwrap(angle(rx_signal))) * Fs / (2*pi*h)

    The output is still sample-rate data. Symbol decisions are made later by
    matched filtering and sampling at the symbol rate.
    """
    rx_signal = np.asarray(rx_signal, dtype=np.complex128).reshape(-1)
    if len(rx_signal) < 2:
        return np.array([], dtype=np.float64)

    phase = np.unwrap(np.angle(rx_signal))
    phase_delta = np.diff(phase)
    return phase_delta * float(sample_rate) / (TWO_PI * float(frequency_sensitivity))


class DemodulatorConfig(pydantic.BaseModel):
    pulse_shape_config: PulseShapeConfig
    constellation_config: ConstellationConfig


class Demodulator:
    def __init__(self, config: DemodulatorConfig):
        self.config = config
        self._pulse_shape_instance = PulseShape(config.pulse_shape_config)
        self._constellation_instance = Constellation(config.constellation_config)
        self._coding_instance = Coding(CodingConfig(constellation=config.constellation_config))

    def demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float,
                   method: str = "differentiate", frequency_sensitivity: float = 0.5,
                   **kwargs) -> np.ndarray:
        if method == "differentiate":
            return self.differentiate_demodulate(rx_signal, symbol_time, sample_rate, frequency_sensitivity)
        if method in ("viterbi", "cpfsk_viterbi"):
            return self.viterbi_demodulate(rx_signal=rx_signal,
                                           symbol_time=symbol_time,
                                           sample_rate=sample_rate,
                                           frequency_sensitivity=frequency_sensitivity,
                                           **kwargs)
        raise ValueError(f"Unknown demodulation method: {method}")

    def differentiate_demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float,
                                 frequency_sensitivity: float = 0.5) -> np.ndarray:
        sps = int(round(float(symbol_time) * float(sample_rate)))
        pulse_shape = np.asarray(self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time),
                                 dtype=np.float64).reshape(-1)
        constellation_points = np.asarray(self._constellation_instance.generate_constellation_points(),
                                          dtype=np.float64)

        discriminator = differentiate(rx_signal=rx_signal,
                                      sample_rate=sample_rate,
                                      frequency_sensitivity=frequency_sensitivity)
        if len(discriminator) == 0:
            return np.array([], dtype=np.uint8)

        matched_filter = pulse_shape[::-1]
        matched_energy = np.sum(np.abs(matched_filter) ** 2)
        if matched_energy <= 0.0:
            return np.array([], dtype=np.uint8)

        filtered = np.convolve(discriminator, matched_filter / matched_energy, mode="full")
        # TX currently uses np.convolve(..., mode="same"), so the transmit pulse
        # is already centered on each symbol sample. The RX matched filter adds
        # half the pulse delay; diff(...) advances the discriminator by one
        # sample, so sample one point before the matched-filter center.
        sampling_offset = max((len(pulse_shape) - 1) // 2 - 1, 0)
        sample_positions = np.arange(sampling_offset, len(filtered), sps, dtype=int)

        decided_indices = []
        for sample_position in sample_positions:
            value = filtered[sample_position]
            decided_indices.append(int(np.argmin(np.abs(value - constellation_points))))

        if not decided_indices:
            return np.array([], dtype=np.uint8)

        decided_symbols = constellation_points[np.asarray(decided_indices, dtype=np.int64)]
        return self._coding_instance.generate_symbols_to_bits(decided_symbols, constellation_points)

    def viterbi_demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float,
                           frequency_sensitivity: float = 0.5,
                           n_phase_states: int = 32,
                           memory_depth_symbols: int | None = None,
                           beam_width: int = 256,
                           use_abs_metric: bool = False,
                           initial_phase: float = 0.0,
                           initial_frequency_hz: float = 0.0,
                           alpha: float = 0.15,
                           beta: float = 0.01,
                           decision_delay_symbols: int | None = None,
                           max_survivors: int | None = None,
                           return_trace: bool = False) -> np.ndarray:
        sps = int(round(float(symbol_time) * float(sample_rate)))
        rx_signal = np.asarray(rx_signal, dtype=np.complex128).reshape(-1)
        constellation_points = np.asarray(self._constellation_instance.generate_constellation_points(),
                                          dtype=np.float64)
        pulse_shape = np.asarray(self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time),
                                 dtype=np.float64).reshape(-1)
        if decision_delay_symbols is None:
            decision_delay_symbols = max((len(pulse_shape) - 1) // (2 * sps), 0)
        if max_survivors is None:
            delay_factor = int(len(constellation_points) ** min(int(decision_delay_symbols), 4))
            max_survivors = max(int(beam_width), min(int(beam_width) * delay_factor, 8192))

        if alpha != 0.0 or beta != 0.0:
            phase_mode = "pll"
        elif use_abs_metric:
            phase_mode = "chunk_phase"
        else:
            phase_mode = "coherent"

        depth_symbols = int(memory_depth_symbols) if memory_depth_symbols is not None else int(decision_delay_symbols)
        depth_symbols = max(depth_symbols, 1)
        n_symbols = int(np.ceil(len(rx_signal) / sps))

        result = cpfsk_viterbi_demodulate(
            rx_signal=rx_signal,
            config=CpfskViterbiConfig(
                sample_rate=float(sample_rate),
                symbol_time=float(symbol_time),
                h=float(frequency_sensitivity),
                pulse_shape_type=self.config.pulse_shape_config.pulse_shape_type,
                pulse_normalization=self.config.pulse_shape_config.normalization_type,
                span_in_symbols=int(self.config.pulse_shape_config.span_in_symbols),
                pulse_causal=bool(self.config.pulse_shape_config.pulse_causal),
                constellation_type=self.config.constellation_config.constellation_type,
                constellation_order=int(self.config.constellation_config.constellation_order),
                depth_symbols=depth_symbols,
                phase_mode=phase_mode,
                initial_phase=float(initial_phase),
                initial_frequency_hz=float(initial_frequency_hz),
                pll_alpha=float(alpha),
                pll_beta=float(beta),
                max_survivors=int(max_survivors),
                n_symbols=int(n_symbols),
            ),
        )
        bits = result["bits"]
        trace = {
            "phase_errors_rad": result.get("pll_phase_errors_rad", np.array([], dtype=np.float64)),
            "residual_frequency_hz": result.get("pll_frequencies_hz", np.array([], dtype=np.float64)),
            "branch_metrics": np.asarray(result.get("metrics", []), dtype=np.float64),
        }

        if return_trace:
            return bits, trace
        return bits
