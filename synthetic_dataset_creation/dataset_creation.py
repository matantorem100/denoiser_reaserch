import numpy as np
import pydantic
from typing import Optional, Dict, List, Any

from synthetic_dataset_creation.channel.channel import ChannelConfig, Channel
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from demodulator_research.demodulator.demodulator import DemodulatorConfig, Demodulator
from synthetic_dataset_creation.modulator.modulator import ModulatorConfig, Modulator
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig


class DatasetConfig(pydantic.BaseModel):
    sample_rate: float = 10_000
    symbol_time: float = 1e-3
    n_bits_to_transmit: int = 100_000

    min_frequency_offset: float = -100
    max_frequency_offset: float = 100
    const_frequency_offset: Optional[float] = None
    frequency_offset_values: Optional[List[float]] = None
    min_phase_offset: float = 0.0
    max_phase_offset: float = 0.0
    const_phase_offset: Optional[float] = 0.0
    phase_offset_values: Optional[List[float]] = None
    h: float = 0.5

    pulse_shape_type: str = "rect"
    span_in_symbols: int = 1
    pulse_normalization: str = "cpfsk"

    constellation_type: str = "PAM"
    constellation_order: int = 4

    noise_type: str = "snr"  # "snr" or "eb_to_n0"
    noise_db_values: List[float] = list(np.arange(0, 20, 2))

    n_signals_per_snr: int = 3

    demodulation_methods: List[str] = ["differentiate"]

    uw_bits: List[int] = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
    uw_probability: float = 1.0
    n_uw: int = 1
    uw_spacing_bits: Optional[int] = None
    uw_start_bit: Optional[int] = None
    random_uw_start: bool = False
    uw_mode: str = "prepend"
    random_seed: Optional[int] = None

    @pydantic.field_validator("uw_probability")
    @classmethod
    def validate_uw_probability(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("uw_probability must be in [0, 1]")
        return value

    @pydantic.field_validator("n_uw")
    @classmethod
    def validate_n_uw(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n_uw must be positive")
        return value


class Dataset:
    def __init__(self, config: DatasetConfig):
        self.config = config

        self.sps = int(round(self.config.sample_rate * self.config.symbol_time))
        self.bits_per_symbol = int(np.log2(self.config.constellation_order))

        pulse_shape_config = PulseShapeConfig(pulse_shape_type=self.config.pulse_shape_type,
                                              normalization_type=self.config.pulse_normalization,
                                              span_in_symbols=self.config.span_in_symbols)

        constellation_config = ConstellationConfig(constellation_type=self.config.constellation_type,
                                                   constellation_order=self.config.constellation_order)

        modulator_config = ModulatorConfig(pulse_shape_config=pulse_shape_config,
                                           constellation_config=constellation_config)

        demodulator_config = DemodulatorConfig(pulse_shape_config=pulse_shape_config,
                                               constellation_config=constellation_config)

        self.modulator = Modulator(modulator_config)
        self.demodulator = Demodulator(demodulator_config)

        self.uw_bits = np.asarray(self.config.uw_bits, dtype=np.uint8)
        self.rng = np.random.default_rng(self.config.random_seed)


    def _random_bits(self, n_bits: int) -> np.ndarray:
        n_bits = (n_bits // self.bits_per_symbol) * self.bits_per_symbol
        return self.rng.integers(0, 2, size=n_bits, dtype=np.uint8)


    def _build_channel(self, noise_db: float) -> Channel:
        channel_config = ChannelConfig(channel_type="awgn", bits_per_symbol=self.bits_per_symbol, sps=self.sps,
                                       noise_type=self.config.noise_type, noise_db=float(noise_db))
        return Channel(channel_config)

    def _choose_frequency_offset(self) -> float:
        if self.config.frequency_offset_values is not None:
            return float(self.rng.choice(np.asarray(self.config.frequency_offset_values, dtype=float)))
        if self.config.const_frequency_offset is not None:
            return float(self.config.const_frequency_offset)
        return float(self.rng.uniform(self.config.min_frequency_offset, self.config.max_frequency_offset))

    def _choose_phase_offset(self) -> float:
        if self.config.phase_offset_values is not None:
            return float(self.rng.choice(np.asarray(self.config.phase_offset_values, dtype=float)))
        if self.config.const_phase_offset is not None:
            return float(self.config.const_phase_offset)
        return float(self.rng.uniform(self.config.min_phase_offset, self.config.max_phase_offset))

    def _choose_has_uw(self) -> bool:
        if not self.config.use_uw:
            return False
        return bool(self.rng.random() < self.config.uw_probability)

    def generate_one_signal(self, noise_db: float) -> Dict[str, Any]:
        """
        This function generates signal in the given snr, n_bits size and:
        1. frequency offset - one value from frequency_offset_values list if the list in the config is not None OR
        const_frequency_offset from the config OR random value between min_frequency_offset to max_frequency_offset
        2. phase_offset - one value from phase_offset_values list if the list in the config is not None OR
        const_phase_offset from the config OR random value between min_phase_offset to max_phase_offset
        3. uw - decide whether the signal will have unique word or not. If not then pass None ub the uw field of the
        modulator. If there is so pass the uw to the modulator
        :param noise_db: the noise of the current simulated signal
        :return: the simulated signal
        """

        frequency_offset = self._choose_frequency_offset()
        phase_offset = self._choose_phase_offset()
        has_uw = self._choose_has_uw()

        bits = self._random_bits(self.config.n_bits_to_transmit)
        uw_to_modulate = self.uw_bits
        if not has_uw:
            uw_to_modulate = None

        tx_signal, tx_bits = self.modulator.modulate(bits=bits, symbol_time=self.config.symbol_time,
                                                     sample_rate=self.config.sample_rate,
                                                     frequency_offset=frequency_offset,
                                                     frequency_sensitivity=self.config.h,
                                                     phase_offset=phase_offset,
                                                     uw=uw_to_modulate,
                                                     n_uw=self.config.n_uw,
                                                     uw_spacing_bits=self.config.uw_spacing_bits,
                                                     uw_start_bit=self.config.uw_start_bit,
                                                     random_uw_start=self.config.random_uw_start,
                                                     rng=self.rng,
                                                     uw_mode=self.config.uw_mode)

        channel = self._build_channel(noise_db)
        rx_signal = channel.transmit(tx_signal)

        return {
            "noise_db": float(noise_db),
            "frequency_offset": float(frequency_offset),
            "phase_offset": float(phase_offset),
            "has_uw": bool(has_uw),
            "uw_bits": None if uw_to_modulate is None else uw_to_modulate.copy(),
            "uw_start_bits": list(self.modulator.last_uw_start_bits),
            "tx_bits": tx_bits,
            "tx_signal": tx_signal,
            "rx_signal": rx_signal,
        }

    def generate_dataset(self) -> Dict[float, Dict[int, Dict[str, Any]]]:
        """
        Returns:

        dataset[snr_db][signal_index] = {
            "noise_db": ...,
            "frequency_offset": ...,
            "uw_bits": ...,
            "tx_bits": ...,
            "tx_signal": ...,
            "rx_signal": ...
        }

        Example:
            dataset[10.0][0]["rx_signal"]
            dataset[10.0][0]["frequency_offset"]
        """

        dataset: Dict[float, Dict[int, Dict[str, Any]]] = {}

        for noise_db in self.config.noise_db_values:
            noise_db = float(noise_db)
            dataset[noise_db] = {}

            for signal_idx in range(self.config.n_signals_per_snr):
                sample = self.generate_one_signal(noise_db=noise_db)
                dataset[noise_db][signal_idx] = sample

        return dataset
