import numpy as np
import pydantic
from typing import Tuple


class ChannelConfig(pydantic.BaseModel):
    channel_type: str
    bits_per_symbol: int
    sps: int
    noise_type: str
    noise_db: float


class Channel:
    def __init__(self, config: ChannelConfig):
        self.config = config

    def transmit(self, signal: np.ndarray, return_noise: bool = False,
                 rng: np.random.Generator | None = None) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:

        if self.config.channel_type == "awgn":
            return self._apply_awgn(signal=signal, return_noise=return_noise, rng=rng)

        raise ValueError(f"Unknown channel type: {self.config.channel_type}")

    def noise_power_from_signal(self, signal: np.ndarray) -> float:
        signal = np.asarray(signal)

        if self.config.noise_type == "eb_to_n0":
            bits_per_symbol = self.config.bits_per_symbol
            sps = self.config.sps

            n_symbols = signal.shape[0] // sps
            if n_symbols == 0:
                return 0.0

            # Symbol energy
            es = np.sum(np.abs(signal) ** 2) / n_symbols

            # Bit energy
            eb = es / bits_per_symbol

            eb_n0_linear = 10 ** (self.config.noise_db / 10.0)
            n0 = eb / eb_n0_linear

            return float(n0)

        elif self.config.noise_type == "snr":
            # Average signal power
            signal_power = np.mean(np.abs(signal) ** 2)

            snr_linear = 10 ** (self.config.noise_db / 10.0)

            noise_power = signal_power / snr_linear

            return float(noise_power)

        else:
            raise ValueError(f"Unknown noise_type: {self.config.noise_type}")

    def generate_awgn(self, reference_signal: np.ndarray, shape: tuple[int, ...] | int | None = None,
                      complex_noise: bool | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
        reference_signal = np.asarray(reference_signal)
        if shape is None:
            shape = reference_signal.shape
        if isinstance(shape, int):
            shape = (shape,)
        if complex_noise is None:
            complex_noise = bool(np.iscomplexobj(reference_signal))
        if rng is None:
            real_normal = np.random.randn
        else:
            real_normal = rng.standard_normal

        noise_power = self.noise_power_from_signal(reference_signal)

        if complex_noise:
            noise_std = np.sqrt(noise_power / 2.0)
            return noise_std * (real_normal(shape) + 1j * real_normal(shape))

        noise_std = np.sqrt(noise_power)
        return noise_std * real_normal(shape)

    def _apply_awgn(self, signal: np.ndarray, return_noise: bool = False,
                    rng: np.random.Generator | None = None) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:

        signal = np.asarray(signal)
        noise = self.generate_awgn(reference_signal=signal, rng=rng)
        rx_signal = signal + noise

        if return_noise:
            return rx_signal, noise

        return rx_signal
