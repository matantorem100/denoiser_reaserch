from typing import Optional, List, Dict, Any
import numpy as np
import pydantic

from synthetic_dataset_creation.channel.channel import ChannelConfig, Channel
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.modulator.modulator import ModulatorConfig, Modulator, FMConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig


class SignalDatasetConfig(pydantic.BaseModel):
    sample_rate: float = 10_000
    symbol_time: float = 1e-3

    n_signals_per_snr: int = 100
    n_bits_per_signal: int = 512

    apply_fm: bool = True
    h: float = 0.5

    uw_n_bits: int = 16

    random_frequency_offset: bool = True
    frequency_offset_min: Optional[float] = None
    frequency_offset_max: Optional[float] = None
    fixed_frequency_offset: float = 0.0

    pulse_shape_type: str = "rrc"
    pulse_normalization: str = "cpfsk"

    constellation_type: str = "PAM"
    constellation_order: int = 4

    noise_type: str = "snr"
    noise_db_values: List[float] = list(np.arange(0, 20, 2))


class SignalDatasetGenerator:
    def __init__(self, config: SignalDatasetConfig):
        self.config = config
        self.sps = int(round(config.sample_rate * config.symbol_time))
        self.bits_per_symbol = int(np.log2(config.constellation_order))
        self._validate_config()

        self.pulse_shape_config = PulseShapeConfig(pulse_shape_type=config.pulse_shape_type,
                                                   normalization_type=config.pulse_normalization)

        self.constellation_config = ConstellationConfig(constellation_type=config.constellation_type,
                                                        constellation_order=config.constellation_order)

    def _validate_config(self):
        if self.config.n_signals_per_snr <= 0:
            raise ValueError("n_signals_per_snr must be positive")

        if self.config.n_bits_per_signal <= 0:
            raise ValueError("n_bits_per_signal must be positive")

        if self.config.uw_n_bits <= 0:
            raise ValueError("uw_n_bits must be positive")

        if self.config.n_bits_per_signal % self.bits_per_symbol != 0:
            raise ValueError("n_bits_per_signal must be divisible by bits_per_symbol")

        if self.config.uw_n_bits % self.bits_per_symbol != 0:
            raise ValueError("uw_n_bits must be divisible by bits_per_symbol")

        if len(self.config.noise_db_values) == 0:
            raise ValueError("noise_db_values must not be empty")

        if self.config.apply_fm and self.config.pulse_normalization != "cpfsk":
            raise ValueError("For CPFSK use pulse_normalization='cpfsk'")

        if not self.config.apply_fm and self.config.pulse_normalization != "norm_2":
            raise ValueError("For linear modulation use pulse_normalization='norm_2'")


    def _sample_frequency_offset(self) -> float:
        if not self.config.random_frequency_offset:
            return float(self.config.fixed_frequency_offset)

        f_min = self.config.frequency_offset_min
        f_max = self.config.frequency_offset_max

        if f_min is None:
            f_min = -self.config.sample_rate / 2

        if f_max is None:
            f_max = self.config.sample_rate / 2

        frequency_offset = np.random.uniform(f_min, f_max)
        return frequency_offset

    def _build_modulator(self, frequency_offset: float) -> Modulator:
        fm_config = None

        if self.config.apply_fm:
            fm_config = FMConfig(frequency_offset=frequency_offset, frequency_sensitivity=self.config.h)

        modulator_config = ModulatorConfig(pulse_shape_config=self.pulse_shape_config,
                                           constellation_config=self.constellation_config, fm_config=fm_config)

        return Modulator(modulator_config)


    def _build_channel(self, noise_db: float) -> Channel:
        channel_config = ChannelConfig(channel_type="awgn", bits_per_symbol=self.bits_per_symbol, sps=self.sps,
                                       noise_type=self.config.noise_type, noise_db=float(noise_db))
        return Channel(channel_config)


    def generate_one(self, noise_db: float, signal_id: int) -> Dict[str, Any]:
        bits = np.random.randint(low=0, high=2, size=self.config.n_bits_per_signal, dtype=np.uint8)

        uw_bits = np.random.randint(low=0, high=2, size=self.config.uw_n_bits, dtype=np.uint8)

        frequency_offset = self._sample_frequency_offset()

        modulator = self._build_modulator(frequency_offset)
        channel = self._build_channel(noise_db)

        modulated_clean_signal, modulated_bits = modulator.modulate(bits=bits, symbol_time=self.config.symbol_time,
                                                sample_rate=self.config.sample_rate, apply_fm=self.config.apply_fm,
                                                uw=uw_bits)

        modulated_noised_signal = channel.transmit(modulated_clean_signal)

        n_uw_symbols = len(uw_bits) // self.bits_per_symbol
        n_uw_samples = n_uw_symbols * self.sps

        uw_clean_signal = modulated_clean_signal[:n_uw_samples]

        return {
            "signal_id": signal_id,
            "snr_db": float(noise_db),
            "frequency_offset": frequency_offset,

            "bits": modulated_bits,
            "payload_bits": bits,
            "uw_bits": uw_bits,

            "clean_signal": modulated_clean_signal,
            "noised_signal": modulated_noised_signal,

            "clean_uw": uw_clean_signal,

            "n_uw_symbols": n_uw_symbols,
            "n_uw_samples": n_uw_samples,
        }

    def generate_dataset(self) -> Dict[float, List[Dict[str, Any]]]:
        dataset = {}
        global_signal_id = 0

        for noise_db in self.config.noise_db_values:
            noise_db = float(noise_db)
            dataset[noise_db] = []

            for _ in range(self.config.n_signals_per_snr):
                sample = self.generate_one(noise_db=noise_db, signal_id=global_signal_id)
                dataset[noise_db].append(sample)
                global_signal_id += 1

        return dataset