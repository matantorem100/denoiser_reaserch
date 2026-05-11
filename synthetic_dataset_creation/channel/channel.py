import numpy as np
import pydantic


class ChannelConfig(pydantic.BaseModel):
    channel_type: str
    bits_per_symbol: int
    sps: int
    noise_type: str
    noise_db: float


class Channel:
    def __init__(self, config: ChannelConfig):
        self.config = config

    def transmit(self, signal: np.ndarray) -> np.ndarray:

        if self.config.channel_type == "awgn":
            return self._apply_awgn(signal)

        raise ValueError(f"Unknown channel type: {self.config.channel_type}")

    def _apply_awgn(self, signal: np.ndarray) -> np.ndarray:

        signal = np.asarray(signal)

        if self.config.noise_type == "eb_to_n0":
            bits_per_symbol = self.config.bits_per_symbol
            sps = self.config.sps

            n_symbols = signal.shape[0] // sps

            # Symbol energy
            es = np.sum(np.abs(signal) ** 2) / n_symbols

            # Bit energy
            eb = es / bits_per_symbol

            eb_n0_linear = 10 ** (self.config.noise_db / 10.0)
            n0 = eb / eb_n0_linear

            if np.iscomplexobj(signal):
                noise_std = np.sqrt(n0 / 2.0)
                noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
            else:
                noise_std = np.sqrt(n0)
                noise = noise_std * np.random.randn(*signal.shape)

            return signal + noise

        elif self.config.noise_type == "snr":
            # Average signal power
            signal_power = np.mean(np.abs(signal) ** 2)

            snr_linear = 10 ** (self.config.noise_db / 10.0)

            noise_power = signal_power / snr_linear

            if np.iscomplexobj(signal):
                noise_std = np.sqrt(noise_power / 2.0)
                noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
            else:
                noise_std = np.sqrt(noise_power)
                noise = noise_std * np.random.randn(*signal.shape)

            return signal + noise

        else:
            raise ValueError(f"Unknown noise_type: {self.config.noise_type}")