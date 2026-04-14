from typing import Optional

import numpy as np
import pydantic

from synthetic_dataset_creation.coding.coding import Coding, CodingConfig
from synthetic_dataset_creation.constellation.constellation import Constellation, ConstellationConfig
from synthetic_dataset_creation.modulator.modulator import FMConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShape, PulseShapeConfig


class DemodulatorConfig(pydantic.BaseModel):
    pulse_shape_config: PulseShapeConfig
    constellation_config: ConstellationConfig
    fm_config: Optional[FMConfig] = None


class Demodulator:
    def __init__(self, config: DemodulatorConfig):
        self.config = config
        self._pulse_shape_instance = PulseShape(config.pulse_shape_config)
        self._constellation_instance = Constellation(config.constellation_config)
        self._coding_instance = Coding(CodingConfig(constellation=config.constellation_config))

    def demodulate(self, rx_signal: np.ndarray, symbol_time: float, sample_rate: float, apply_fm: bool = False) -> np.ndarray:

        sps = int(symbol_time * sample_rate)

        # FM demod if needed
        if apply_fm:
            rx_signal = (np.diff(np.unwrap(np.angle(rx_signal))) /
                         (2.0 * np.pi * self.config.fm_config.frequency_sensitivity) * sample_rate)

        constellation_points = self._constellation_instance.generate_constellation_points()
        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)

        # For real recovered message, matched filtering is real
        matched_filter = pulse_shape[::-1].conj()
        filtered_signal = np.convolve(rx_signal, matched_filter, mode="full")

        sampling_offset = ((len(pulse_shape) - 1) // 2) * 2
        detected_symbols = filtered_signal[sampling_offset::sps]

        # Estimate expected number of symbols from input duration
        n_expected_symbols = int(np.ceil((len(rx_signal) - len(pulse_shape) + 1) / sps))
        n_expected_symbols = max(n_expected_symbols, 0)
        detected_symbols = detected_symbols[:n_expected_symbols]

        bits = self._coding_instance.generate_symbols_to_bits(detected_symbols, constellation_points)

        return bits