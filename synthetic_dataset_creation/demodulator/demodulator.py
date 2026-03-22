from typing import Optional

import numpy as np
import pydantic

from synthetic_dataset_creation.coding.coding import Coding, CodingConfig
from synthetic_dataset_creation.constellation.constellation import Constellation, ConstellationConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShape, PulseShapeConfig


class FMConfig(pydantic.BaseModel):
    frequency_offset: float = 0.0
    modulation_index: float = 0.5


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

    def _fm_demodulate_baseband(self, rx_signal: np.ndarray, sample_rate: float) -> np.ndarray:
        if self.config.fm_config is None:
            raise ValueError("FM config is not provided")

        fm_cfg = self.config.fm_config

        rx_signal = np.asarray(rx_signal)
        if rx_signal.ndim != 1:
            raise ValueError("rx_signal must be a 1D numpy array")
        if rx_signal.size == 0:
            raise ValueError("rx_signal must not be empty")
        if not np.iscomplexobj(rx_signal):
            raise ValueError("FM baseband demod expects a complex-valued signal")
        if fm_cfg.modulation_index == 0:
            raise ValueError("modulation_index must be non-zero")

        rx_signal = rx_signal.astype(np.complex128, copy=False)

        # Remove amplitude variation before phase extraction
        mag = np.abs(rx_signal)
        valid = mag > 0
        normalized = np.zeros_like(rx_signal, dtype=np.complex128)
        normalized[valid] = rx_signal[valid] / mag[valid]

        phase = np.unwrap(np.angle(normalized))

        # Per-sample phase increment. Keep same output length.
        dphase = np.diff(phase, prepend=phase[0])

        # Remove constant residual frequency offset term
        phase_offset_per_sample = 2.0 * np.pi * fm_cfg.frequency_offset / sample_rate
        dphase = dphase - phase_offset_per_sample

        # Recover the real-valued shaped message m[n]
        message = dphase / (2.0 * np.pi * fm_cfg.modulation_index)

        return message.astype(np.float64)

    def demodulate(
        self,
        rx_signal: np.ndarray,
        symbol_time: float,
        sample_rate: float,
        apply_fm: bool = False,
    ) -> np.ndarray:
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

        # Step 1: FM demod if needed
        if apply_fm:
            rx_signal = self._fm_demodulate_baseband(rx_signal, sample_rate)

        constellation_points = self._constellation_instance.generate_constellation_points()
        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)

        # For real recovered message, matched filtering is real
        matched_filter = pulse_shape[::-1].conj()
        filtered_signal = np.convolve(rx_signal, matched_filter, mode="full")

        sampling_offset = len(pulse_shape) - 1
        detected_symbols = filtered_signal[sampling_offset::sps]

        # Estimate expected number of symbols from input duration
        n_expected_symbols = int(np.ceil((len(rx_signal) - len(pulse_shape) + 1) / sps))
        n_expected_symbols = max(n_expected_symbols, 0)
        detected_symbols = detected_symbols[:n_expected_symbols]

        bits = self._coding_instance.generate_symbols_to_bits(
            detected_symbols,
            constellation_points,
        )

        return bits