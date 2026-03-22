from typing import Literal

import numpy as np
import pydantic


class ChannelConfig(pydantic.BaseModel):
    channel_type: Literal["awgn"]
    snr_db: float
    random_seed: int | None = None


class Channel:
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def transmit(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal)

        if signal.ndim != 1:
            raise ValueError("signal must be a 1D numpy array")

        if self.config.channel_type == "awgn":
            return self._apply_awgn(signal)

        raise ValueError(f"Unknown channel type: {self.config.channel_type}")


    def _apply_awgn(self, signal: np.ndarray) -> np.ndarray:
        signal_power = np.mean(np.abs(signal) ** 2)

        if signal_power == 0:
            raise ValueError("Cannot add AWGN to a zero-power signal")

        snr_linear = 10 ** (self.config.snr_db / 10)
        noise_power = signal_power / snr_linear

        if np.iscomplexobj(signal):
            noise_std = np.sqrt(noise_power / 2)
            noise = noise_std * (self.rng.standard_normal(signal.shape) + 1j * self.rng.standard_normal(signal.shape))
        else:
            noise_std = np.sqrt(noise_power)
            noise = noise_std * self.rng.standard_normal(signal.shape)

        return signal + noise