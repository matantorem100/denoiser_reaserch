from typing import Optional

import numpy as np
import pydantic

from synthetic_dataset_creation.channel.channel import ChannelConfig, Channel
from synthetic_dataset_creation.coding.coding import CodingConfig, Coding
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig, Constellation
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig, PulseShape


class FMConfig(pydantic.BaseModel):
    frequency_offset: float = 0.0
    frequency_sensitivity: float = 1.0


class ModulatorConfig(pydantic.BaseModel):
    pulse_shape_config: PulseShapeConfig
    constellation_config: ConstellationConfig
    channel_config: ChannelConfig
    fm_config: Optional[FMConfig] = None


class Modulator:
    def __init__(self, config: ModulatorConfig):
        self.config = config
        self._pulse_shape_instance = PulseShape(config.pulse_shape_config)
        self._constellation_instance = Constellation(config.constellation_config)
        self._channel_instance = Channel(config.channel_config)
        self._coding_instance = Coding(CodingConfig(constellation=config.constellation_config))

    def _fm_modulate_baseband(self, message: np.ndarray, sample_rate: float) -> np.ndarray:
        if self.config.fm_config is None:
            raise ValueError("FM config is not provided")

        fm_cfg = self.config.fm_config

        if np.iscomplexobj(message):
            raise ValueError("FM must be real-valued.")

        t = np.arange(len(message), dtype=np.float32) / sample_rate

        # Residual/baseband frequency offset term 2*pi*f_offset*t
        phase_offset = 2.0 * np.pi * fm_cfg.frequency_offset * t

        # FM message term 2*pi*kf * integral(message dt)
        phase_message = 2.0 * np.pi * fm_cfg.frequency_sensitivity * np.cumsum(message) / sample_rate

        total_phase = phase_offset + phase_message

        # Complex baseband output
        signal = np.exp(1j * total_phase)
        return signal

    def modulate(self, bits: np.ndarray, symbol_time: float, sample_rate: float, apply_fm: bool = False) -> np.ndarray:
        if bits.ndim != 1:
            raise ValueError("bits must be a 1D numpy array")

        constellation_points = self._constellation_instance.generate_constellation_points()
        symbols = self._coding_instance.generate_bits_to_symbols(bits, constellation_points)

        sps = int(symbol_time * sample_rate)

        upsampled_symbols = np.zeros(symbols.shape[0] * sps, dtype=symbols.dtype)
        upsampled_symbols[::sps] = symbols

        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)
        modulated_signal = np.convolve(upsampled_symbols, pulse_shape, mode="full")

        if apply_fm:
            modulated_signal =  self._fm_modulate_baseband(modulated_signal, sample_rate)

        transmitted_signal = self._channel_instance.transmit(modulated_signal)

        return transmitted_signal


