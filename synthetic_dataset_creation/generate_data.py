import copy
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from synthetic_dataset_creation.modulator.modulator import Modulator, ModulatorConfig
from synthetic_dataset_creation.coding.coding import Coding, CodingConfig
from synthetic_dataset_creation.constellation.constellation import Constellation


class PamOverFmDataset(Dataset):
    def __init__(
        self,
        modulator_config: ModulatorConfig,
        split: str = "train",
        dataset_size: int = 100_000,
        n_bits: int = 128,
        symbol_time: float = 0.1,
        sample_rate: float = 100.0,
        snr_db_min: float = 0.0,
        snr_db_max: float = 14.0,
        fixed_snr_db: Optional[float] = None,
        segment_seconds: Optional[float] = None,
        random_crop: Optional[bool] = None,
        normalize: bool = False,
        return_bits: bool = False,
        return_symbols: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.base_modulator_config = copy.deepcopy(modulator_config)
        self.split = split
        self.dataset_size = dataset_size
        self.n_bits = n_bits
        self.symbol_time = symbol_time
        self.sample_rate = sample_rate
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.fixed_snr_db = fixed_snr_db
        self.normalize = normalize
        self.return_bits = return_bits
        self.return_symbols = return_symbols

        self.segment_seconds = segment_seconds
        self.segment_length = (int(sample_rate * segment_seconds) if segment_seconds is not None else None)

        if random_crop is None:
            random_crop = (split == "train")
        self.random_crop = random_crop

        self.rng = np.random.default_rng(seed)
        self.py_random = random.Random(seed)

        constellation = Constellation(self.base_modulator_config.constellation_config)
        self.constellation_points = constellation.generate_constellation_points()
        self.coding = Coding(CodingConfig(constellation=self.base_modulator_config.constellation_config))

        self.bits_per_symbol = int(np.log2(len(self.constellation_points)))
        if self.n_bits % self.bits_per_symbol != 0:
            raise ValueError(f"n_bits={self.n_bits} must be divisible by bits_per_symbol={self.bits_per_symbol}")

    def __len__(self):
        return self.dataset_size

    def _sample_bits(self) -> np.ndarray:
        return self.rng.integers(0, 2, size=self.n_bits, dtype=np.int64)

    def _sample_snr_db(self) -> float:
        if self.fixed_snr_db is not None:
            return float(self.fixed_snr_db)
        return float(self.rng.uniform(self.snr_db_min, self.snr_db_max))

    def _make_modulator(self, snr_db: float, random_seed: Optional[int]) -> Modulator:
        cfg = copy.deepcopy(self.base_modulator_config)
        cfg.channel_config.snr_db = float(snr_db)
        cfg.channel_config.random_seed = random_seed
        return Modulator(cfg)

    def _fm_demodulate(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal)

        if signal.ndim != 1:
            raise ValueError("signal must be a 1D numpy array")
        if len(signal) < 2:
            raise ValueError("signal length must be at least 2")
        if not np.iscomplexobj(signal):
            raise ValueError("signal must be complex for FM demodulation")

        # First discriminator pass
        fm_demodulated_1 = np.angle(signal[1:] * np.conj(signal[:-1]))  # radians/sample

        # Estimate constant offset
        mean_phase_increment = np.mean(fm_demodulated_1)

        # Cancel offset on original signal
        n = np.arange(len(signal))
        correction = np.exp(-1j * mean_phase_increment * n)
        signal_corrected = signal * correction

        # Second discriminator pass
        fm_demodulated_2 = np.angle(signal_corrected[1:] * np.conj(signal_corrected[:-1]))

        return fm_demodulated_2.astype(np.float32)

    def _normalize_waveform(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - x.mean()) / (x.std() + eps)

    def _crop_or_pad(self, noisy: np.ndarray, clean: np.ndarray):
        L = min(len(noisy), len(clean))
        noisy = noisy[:L]
        clean = clean[:L]

        if self.segment_length is not None:
            if L >= self.segment_length:
                if self.random_crop:
                    start = self.py_random.randint(0, L - self.segment_length)
                else:
                    start = (L - self.segment_length) // 2
                end = start + self.segment_length
                noisy = noisy[start:end]
                clean = clean[start:end]
            else:
                pad = self.segment_length - L
                noisy = np.pad(noisy, (0, pad))
                clean = np.pad(clean, (0, pad))

        return noisy, clean

    def __getitem__(self, idx):
        bits = self._sample_bits()
        snr_db = self._sample_snr_db()

        noisy_seed = int(self.rng.integers(0, 2**31 - 1))
        clean_seed = int(self.rng.integers(0, 2**31 - 1))

        noisy_modulator = self._make_modulator(snr_db=snr_db, random_seed=noisy_seed)
        clean_modulator = self._make_modulator(snr_db=120.0, random_seed=clean_seed)

        noisy_fm = noisy_modulator.modulate(
            bits=bits,
            symbol_time=self.symbol_time,
            sample_rate=self.sample_rate,
            apply_fm=True,
        )

        clean_fm = clean_modulator.modulate(
            bits=bits,
            symbol_time=self.symbol_time,
            sample_rate=self.sample_rate,
            apply_fm=True,
        )

        noisy = self._fm_demodulate(noisy_fm)
        clean = self._fm_demodulate(clean_fm)

        noisy, clean = self._crop_or_pad(noisy, clean)

        if self.normalize:
            noisy = self._normalize_waveform(noisy)
            clean = self._normalize_waveform(clean)

        noisy = torch.from_numpy(noisy).float().unsqueeze(0)   # [1, T]
        clean = torch.from_numpy(clean).float().unsqueeze(0)   # [1, T]

        out = {
            "noisy": noisy,
            "clean": clean,
        }

        if self.return_bits:
            out["bits"] = torch.from_numpy(bits).long()

        if self.return_symbols:
            symbols = self.coding.generate_bits_to_symbols(bits, self.constellation_points)
            out["symbols"] = torch.from_numpy(np.asarray(symbols)).float()

        out["snr_db"] = torch.tensor(snr_db, dtype=torch.float32)

        return out

# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#
#     from synthetic_dataset_creation.modulator.modulator import ModulatorConfig, FMConfig
#     from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig
#     from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
#     from synthetic_dataset_creation.channel.channel import ChannelConfig
#
#     pulse_shape_config = PulseShapeConfig(
#         pulse_shape_type="rect",
#         normalization_type="cpfsk",
#     )
#     constellation_config = ConstellationConfig(
#         constellation_type="PAM",
#         constellation_order=2,
#     )
#     channel_config = ChannelConfig(
#         channel_type="awgn",
#         snr_db=30,
#         random_seed=None,
#     )
#     fm_config = FMConfig(
#         frequency_offset=0,
#         frequency_sensitivity=1,
#         amplitude=1,
#         normalize_message=True,
#     )
#
#     modulator_config = ModulatorConfig(
#         pulse_shape_config=pulse_shape_config,
#         constellation_config=constellation_config,
#         channel_config=channel_config,
#         fm_config=fm_config,
#     )
#
#     train_dataset = PamOverFmDataset(
#         modulator_config=modulator_config,
#         split="train",
#         dataset_size=100_000,
#         n_bits=128,
#         symbol_time=0.1,
#         sample_rate=100.0,
#         snr_db_min=0.0,
#         snr_db_max=14.0,
#         segment_seconds=None,  # full signal
#         random_crop=True,
#         normalize=False,
#         return_bits=False,
#         return_symbols=False,
#         seed=1234,
#     )
#
#     val_dataset = PamOverFmDataset(
#         modulator_config=modulator_config,
#         split="validation",
#         dataset_size=10_000,
#         n_bits=128,
#         symbol_time=0.1,
#         sample_rate=100.0,
#         snr_db_min=0.0,
#         snr_db_max=14.0,
#         segment_seconds=None,
#         random_crop=False,
#         normalize=False,
#         seed=2026,
#     )
#
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
#     batch = next(iter(train_loader))
#
#     print(batch["noisy"].shape)  # [B, 1, T]
#     print(batch["clean"].shape)  # [B, 1, T]
#     print(batch["snr_db"].shape)  # [B]