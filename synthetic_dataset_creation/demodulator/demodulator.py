import numpy as np
import pydantic

from synthetic_dataset_creation.coding.coding import Coding, CodingConfig
from synthetic_dataset_creation.constellation.constellation import Constellation, ConstellationConfig
from synthetic_dataset_creation.pulse_shape import PulseShape, PulseShapeConfig


class DemodulatorConfig(pydantic.BaseModel):
    pulse_shape_config: PulseShapeConfig
    constellation_config: ConstellationConfig


class Demodulator:
    def __init__(self, config: DemodulatorConfig):
        self.config = config
        self._pulse_shape_instance = PulseShape(config.pulse_shape_config)
        self._constellation_instance = Constellation(config.constellation_config)
        self._coding_instance = Coding(CodingConfig(constellation=config.constellation_config))

    def demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float) -> np.ndarray:
        rx_signal = np.asarray(rx_signal)

        if rx_signal.ndim != 1:
            raise ValueError("rx_signal must be a 1D numpy array")
        if rx_signal.size == 0:
            raise ValueError("rx_signal must not be empty")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if symbol_time <= 0:
            raise ValueError("symbol_time must be positive")

        sps = round(symbol_time * sample_rate)
        if sps < 1:
            raise ValueError("sample_rate * symbol_time must be at least 1")

        constellation_points = self._constellation_instance.generate_constellation_points()
        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)

        matched_filter = pulse_shape[::-1]
        filtered_signal = np.convolve(rx_signal, matched_filter, mode="full")

        sampling_offset = len(pulse_shape) - 1
        detected_symbols = filtered_signal[sampling_offset::sps]

        n_expected_symbols = int(np.ceil((len(rx_signal) - len(pulse_shape) + 1) / sps))
        n_expected_symbols = max(n_expected_symbols, 0)
        detected_symbols = detected_symbols[:n_expected_symbols]

        bits = self._coding_instance.generate_symbols_to_bits(
            detected_symbols,
            constellation_points,
        )

        return bits