from typing import Tuple, Optional

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
        self.last_uw_start_bits: list[int] = []

    @staticmethod
    def _fm_modulate_baseband(message: np.ndarray, sample_rate: float, frequency_offset: float = None,
                              frequency_sensitivity: float = 0.5, phase_offset: float = 0.0) -> np.ndarray:
        """
        this function apply signal over fm to create cpfsk modulation
        :param message: the signal
        :param sample_rate: the sample rate of the signal
        :param frequency_offset: the simulated deviation
        :param frequency_sensitivity: modulation index h
        :param phase_offset: constant phase offset in radians
        :return: the cpfsk modulated signal
        """
        if np.iscomplexobj(message):
            raise ValueError("FM must be real-valued.")

        t = np.arange(len(message), dtype=np.float32) / sample_rate

        if frequency_offset is None:
            frequency_offset = 0.0

        # Residual/baseband frequency offset term 2*pi*f_offset*t
        frequency_phase = 2.0 * np.pi * frequency_offset * t

        # FM message term 2*pi*kf * integral(message dt)
        phase_message = 2.0 * np.pi * frequency_sensitivity * np.cumsum(message) / sample_rate

        total_phase = frequency_phase + phase_message + phase_offset

        # Complex baseband output
        signal = np.exp(1j * total_phase)
        return signal

    @staticmethod
    def _build_uw_start_bits(n_bits: int, n_uw: int, uw_len: int, uw_spacing_bits: int | None,
                             uw_start_bit: int | None, random_uw_start: bool, rng: np.random.Generator | None,
                             alignment_bits: int = 1) -> list[int] | None:

        if n_uw <= 0:
            raise ValueError("n_uw must be positive")
        if alignment_bits <= 0:
            raise ValueError("alignment_bits must be positive")
        if uw_len % alignment_bits != 0:
            raise ValueError("UW length must be divisible by bits_per_symbol")

        if uw_start_bit is None and not random_uw_start:
            return None

        if n_uw == 1:
            required_span = uw_len
        else:
            if uw_spacing_bits is None:
                raise ValueError("uw_spacing_bits is required when n_uw > 1")
            if int(uw_spacing_bits) % alignment_bits != 0:
                raise ValueError("uw_spacing_bits must be divisible by bits_per_symbol")
            required_span = (n_uw - 1) * int(uw_spacing_bits) + uw_len

        if required_span > n_bits:
            raise ValueError("UWs do not fit inside bits")

        if random_uw_start:
            if uw_start_bit is not None:
                raise ValueError("Use either random_uw_start or uw_start_bit, not both")
            if rng is None:
                rng = np.random.default_rng()
            max_start_symbol = (n_bits - required_span) // alignment_bits
            first_start = int(rng.integers(0, max_start_symbol + 1)) * alignment_bits
        else:
            first_start = int(uw_start_bit)

        if first_start % alignment_bits != 0:
            raise ValueError("uw_start_bit must be divisible by bits_per_symbol")
        if first_start < 0 or first_start + required_span > n_bits:
            raise ValueError("UW group does not fit inside bits")

        if n_uw == 1:
            return [first_start]

        return [first_start + i * int(uw_spacing_bits) for i in range(n_uw)]


    def _apply_unique_words(self, bits: np.ndarray, uw: Optional[np.ndarray], uw_start_bit: Optional[int],
                            uw_spacing_bits: Optional[int], n_uw: Optional[int] = 1, random_uw_start: Optional[bool] = False,
                            rng: np.random.Generator | None = None, uw_mode: Optional[str] = "overwrite") -> np.ndarray:

        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        self.last_uw_start_bits = []

        if uw is None:
            return bits

        uw = np.asarray(uw, dtype=np.uint8).reshape(-1)
        if uw.size == 0:
            return bits

        if not np.all((uw == 0) | (uw == 1)):
            raise ValueError("uw must contain only 0/1 values")

        bits_per_symbol = int(np.log2(self.config.constellation_config.constellation_order))
        starts = self._build_uw_start_bits(n_bits=len(bits), n_uw=int(n_uw), uw_len=len(uw),
                                           uw_spacing_bits=uw_spacing_bits, uw_start_bit=uw_start_bit,
                                           random_uw_start=random_uw_start, rng=rng,
                                           alignment_bits=bits_per_symbol)

        if starts is None:
            if uw_mode != "prepend":
                raise ValueError("uw_start_bit or random_uw_start is required unless uw_mode='prepend'")
            self.last_uw_start_bits = [0]
            return np.concatenate((uw, bits)).astype(np.uint8)

        if uw_mode == "overwrite":
            output_bits = bits.copy()
            for start in starts:
                stop = start + len(uw)
                if stop > len(output_bits):
                    raise ValueError("UW does not fit in bits for uw_mode='overwrite'")
                output_bits[start:stop] = uw
            self.last_uw_start_bits = starts
            return output_bits.astype(np.uint8)

        if uw_mode == "insert":
            output_bits = bits.copy()
            inserted_bits = 0
            inserted_starts = []
            for start in sorted(starts):
                adjusted_start = start + inserted_bits
                if adjusted_start > len(output_bits):
                    raise ValueError("UW insert position is beyond the end of bits")
                inserted_starts.append(adjusted_start)
                output_bits = np.concatenate((output_bits[:adjusted_start], uw, output_bits[adjusted_start:]))
                inserted_bits += len(uw)
            self.last_uw_start_bits = inserted_starts
            return output_bits.astype(np.uint8)

        if uw_mode == "prepend":
            raise ValueError("Use uw_mode='insert' or 'overwrite' when UW positions are provided")

        raise ValueError("uw_mode must be one of: 'prepend', 'insert', 'overwrite'")


    def modulate(self, bits: np.ndarray, symbol_time: float, sample_rate: float, uw: np.ndarray = None,
                 frequency_offset: float = None, frequency_sensitivity: float = 0.5,
                 n_uw: int = 1, uw_spacing_bits: int | None = None,
                 uw_start_bit: int | None = None, random_uw_start: bool = False,
                 rng: np.random.Generator | None = None, uw_mode: str = "prepend",
                 phase_offset: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function aims to create cpfsk modulation
        :param bits: the bits to transmit
        :param symbol_time: the time that each symbol is transmitted
        :param sample_rate: the sample rate of the signal
        :param uw: the unique word bits
        :param frequency_offset: the simulated deviation
        :param frequency_sensitivity: the modulation index h
        :param n_uw: number of UWs to place
        :param uw_spacing_bits: bit spacing between UWs when n_uw > 1
        :param uw_start_bit: first UW bit index. If None and random_uw_start=True,
                             the first UW is chosen randomly so all UWs fit.
        :param random_uw_start: randomly choose the first UW position
        :param rng: random generator used when random_uw_start=True
        :param uw_mode: "prepend" for old behavior, "insert" to grow the bitstream, or
                        "overwrite" to keep the bitstream length fixed
        :param phase_offset: constant phase offset in radians
        :return: the modulated signal
        """

        bits = self._apply_unique_words(bits=bits, uw=uw, uw_start_bit=uw_start_bit,
                                        uw_spacing_bits=uw_spacing_bits, n_uw=n_uw,
                                        random_uw_start=random_uw_start, rng=rng, uw_mode=uw_mode)

        constellation_points = self._constellation_instance.generate_constellation_points()
        symbols = self._coding_instance.generate_bits_to_symbols(bits, constellation_points)

        sps = int(symbol_time * sample_rate)

        upsampled_symbols = np.zeros((symbols.shape[0] - 1) * sps + 1, dtype=symbols.dtype)
        upsampled_symbols[::sps] = symbols

        pulse_shape = self._pulse_shape_instance.generate_pulse_shape(sample_rate, symbol_time)
        modulated_signal = np.convolve(upsampled_symbols, pulse_shape, mode="full")

        modulated_signal = self._fm_modulate_baseband(
            message=modulated_signal,
            sample_rate=sample_rate,
            frequency_offset=frequency_offset,
            frequency_sensitivity=frequency_sensitivity,
            phase_offset=phase_offset,
        )

        return modulated_signal, bits
