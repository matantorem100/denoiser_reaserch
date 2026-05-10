from typing import Tuple

import numpy as np
import pydantic

from synthetic_dataset_creation.coding.coding import CodingConfig, Coding
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig, Constellation
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig, PulseShape


class ModulatorConfig(pydantic.BaseModel):
    pulse_shape_config: PulseShapeConfig
    constellation_config: ConstellationConfig


class Modulator:
    def __init__(self, config: ModulatorConfig):
        self.config = config
        self._pulse_shape_instance = PulseShape(config.pulse_shape_config)
        self._constellation_instance = Constellation(config.constellation_config)
        self._coding_instance = Coding(CodingConfig(constellation=config.constellation_config))

    @staticmethod
    def _fm_modulate_baseband(message: np.ndarray, sample_rate: float, frequency_offset: float = None,
                              frequency_sensitivity: float = 0.5) -> np.ndarray:
        """
        this function apply signal over fm to create cpfsk modulation
        :param message: the signal
        :param sample_rate: the sample rate of the signal
        :param frequency_offset: the simulated deviation
        :param frequency_sensitivity: modulation index h
        :return: the cpfsk modulated signal
        """
        if np.iscomplexobj(message):
            raise ValueError("FM must be real-valued.")

        t = np.arange(len(message), dtype=np.float32) / sample_rate

        # Residual/baseband frequency offset term 2*pi*f_offset*t
        phase_offset = 2.0 * np.pi * frequency_offset * t

        # FM message term 2*pi*kf * integral(message dt)
        phase_message = 2.0 * np.pi * frequency_sensitivity * np.cumsum(message) / sample_rate

        total_phase = phase_offset + phase_message

        # Complex baseband output
        signal = np.exp(1j * total_phase)
        return signal

    def modulate(self, bits: np.ndarray, symbol_time: float, sample_rate: float, uw: np.ndarray = None,
                 frequency_offset: float = None, frequency_sensitivity: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function aims to create cpfsk modulation
        :param bits: the bits to transmit
        :param symbol_time: the time that each symbol is transmitted
        :param sample_rate: the sample rate of the signal
        :param uw: the unique word bits
        :param frequency_offset: the simulated deviation
        :param frequency_sensitivity: the modulation index h
        :return: the modulated signal
        """

        if uw is not None:
            bits = np.concatenate((uw, bits))

        constellation_points = self._constellation_instance.generate_constellation_points()
        symbols = self._coding_instance.generate_bits_to_symbols(bits, constellation_points)

        sps = int(symbol_time * sample_rate)

        upsampled_symbols = np.zeros((symbols.shape[0] - 1) * sps + 1, dtype=symbols.dtype)
        upsampled_symbols[::sps] = symbols

        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)
        modulated_signal = np.convolve(upsampled_symbols, pulse_shape, mode="full")

        modulated_signal =  self._fm_modulate_baseband(modulated_signal, sample_rate, frequency_offset, frequency_sensitivity)

        return modulated_signal, bits


