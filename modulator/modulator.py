import numpy as np
import pydantic

from coding.coding import CodingConfig, Coding
from constellation.constellation import ConstellationConfig, Constellation
from pulse_shape.pulse_shape import PulseShapeConfig, PulseShape


class ModulatorConfig(pydantic.BaseModel):
    pulse_shape_config: PulseShapeConfig
    constellation_config: ConstellationConfig


class Modulator:
    def __init__(self, config: ModulatorConfig):
        self.config = config
        self._pulse_shape_instance = PulseShape(config.pulse_shape_config)
        self._constellation_instance = Constellation(config.constellation_config)
        self._coding_instance = Coding(CodingConfig(constellation=config.constellation_config))

    def modulate(self, bits: np.ndarray, symbol_time: float, sample_rate: float) -> np.ndarray:
        if bits.ndim != 1:
            raise ValueError("bits must be a 1D numpy array")

        constellation_points = self._constellation_instance.generate_constellation_points()

        symbols = self._coding_instance.generate_bits_to_symbols(bits, constellation_points)

        sps = round(symbol_time * sample_rate)

        upsampled_symbols = np.zeros(symbols.shape[0] * sps, dtype=symbols.dtype)
        upsampled_symbols[::sps] = symbols

        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)

        modulated_signal = np.convolve(upsampled_symbols, pulse_shape, mode="full")
        return modulated_signal