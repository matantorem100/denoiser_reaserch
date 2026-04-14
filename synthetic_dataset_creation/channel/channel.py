from typing import Literal, Union

import numpy as np
import pydantic


class ChannelConfig(pydantic.BaseModel):
    channel_type: Literal["awgn"]
    bits_per_symbol: int
    sps: int
    snr_db: float


class Channel:
    def __init__(self, config: ChannelConfig):
        self.config = config

    def transmit(self, signal: np.ndarray) -> np.ndarray:

        if self.config.channel_type == "awgn":
            return self._apply_awgn(signal)

        raise ValueError(f"Unknown channel type: {self.config.channel_type}")

    def _apply_awgn(self, signal: np.ndarray) -> np.ndarray:

        bits_per_symbol = self.config.bits_per_symbol
        sps = self.config.sps
        n_symbols = signal.shape[0] / sps

        es = np.sum(np.abs(signal) ** 2) / n_symbols
        eb = es / bits_per_symbol
        eb_n0_linear = 10 ** (self.config.snr_db / 10.0)
        n0 = eb / eb_n0_linear

        if np.iscomplexobj(signal):
            noise_std = np.sqrt(n0 / 2.0)
            noise = noise_std * (np.random.randn(signal.shape[0]) + 1j * np.random.randn(signal.shape[0]))
        else:
            noise_std = np.sqrt(n0 / 2.0)
            noise = noise_std * np.random.randn(signal.shape[0])

        return signal + noise