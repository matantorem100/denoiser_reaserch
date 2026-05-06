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

    def _random_bits(self, n_bits: int) -> np.ndarray:
        """
        Make sure number of bits is divisible by bits_per_symbol.
        For PAM4, bits_per_symbol = 2.
        """
        n_bits = (n_bits // self.bits_per_symbol) * self.bits_per_symbol
        return np.random.randint(0, 2, size=n_bits, dtype=np.uint8)


    def _build_channel(self, noise_db: float) -> Channel:
        channel_config = ChannelConfig(channel_type="awgn", bits_per_symbol=self.bits_per_symbol, sps=self.sps,
                                       noise_type=self.config.noise_type, noise_db=float(noise_db))
        return Channel(channel_config)

    def generate_one_signal(self, noise_db: float, frequency_offset: Optional[float] = None,
                            uw: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate one signal with a specific SNR and frequency offset.
        """

        bits = self._random_bits(self.config.n_bits_to_transmit)

        tx_signal, tx_bits = self.modulator.modulate(bits=bits, symbol_time=self.config.symbol_time,
                                                     sample_rate=self.config.sample_rate,
                                                     frequency_offset=frequency_offset, uw=uw)

        channel = self._build_channel(noise_db)
        rx_signal = channel.transmit(tx_signal)

        return {
            "noise_db": float(noise_db),
            "frequency_offset": float(frequency_offset),
            "uw_bits": uw.copy(),
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
                if self.config.const_frequency_offset is not None:
                    frequency_offset = self.config.const_frequency_offset
                else:
                    frequency_offset = np.random.uniform(self.config.min_frequency_offset, self.config.max_frequency_offset)

                sample = self.generate_one_signal(noise_db=noise_db, frequency_offset=frequency_offset, uw=self.uw_bits)

                dataset[noise_db][signal_idx] = sample

        return dataset
