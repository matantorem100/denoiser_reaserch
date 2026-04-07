import copy
import random
from typing import Callable, Optional

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
        normalize: bool = True,
        normalization_eps: float = 1e-8,
        target_mode: str = "post_demod_clean",
        target_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        clean_snr_db: float = 120.0,
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
        self.normalization_eps = normalization_eps

        self.target_mode = target_mode
        self.target_transform = target_transform
        self.clean_snr_db = clean_snr_db

        self.return_bits = return_bits
        self.return_symbols = return_symbols

        self.segment_seconds = segment_seconds
        self.segment_length = (int(sample_rate * segment_seconds) if segment_seconds is not None else None)

        if random_crop is None:
            random_crop = (split == "train")
        self.random_crop = random_crop

        self.rng = np.random.default_rng(seed)
        self.py_random = random.Random(seed)

        self.constellation = Constellation(self.base_modulator_config.constellation_config)
        self.constellation_points = self.constellation.generate_constellation_points()
        self.coding = Coding(CodingConfig(constellation=self.base_modulator_config.constellation_config))

        self.bits_per_symbol = int(np.log2(len(self.constellation_points)))
        if self.n_bits % self.bits_per_symbol != 0:
            raise ValueError( f"n_bits={self.n_bits} must be divisible by bits_per_symbol={self.bits_per_symbol}")

        allowed_target_modes = {"post_demod_clean", "transformed_clean"}
        if self.target_mode not in allowed_target_modes:
            raise ValueError(f"target_mode must be one of {allowed_target_modes}, got {self.target_mode}")

        if self.target_mode == "transformed_clean" and self.target_transform is None:
            raise ValueError("target_transform must be provided when target_mode='transformed_clean'")

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

        if not np.iscomplexobj(signal):
            raise ValueError("signal must be complex for FM demodulation")

        fm_demodulated_1 = np.diff(np.unwrap(np.angle(signal)))
        mean_phase_increment = np.mean(fm_demodulated_1)

        n = np.arange(len(signal))
        correction = np.exp(-1j * mean_phase_increment * n)
        signal_corrected = signal * correction

        fm_demodulated_2 = np.diff(np.unwrap(np.angle(signal_corrected)))
        return fm_demodulated_2.astype(np.float32)

    def _build_target(self, clean_post_demod: np.ndarray) -> np.ndarray:
        if self.target_mode == "post_demod_clean":
            return clean_post_demod.astype(np.float32)

        if self.target_mode == "transformed_clean":
            target = self.target_transform(clean_post_demod)
            return np.asarray(target, dtype=np.float32)

        raise RuntimeError(f"Unhandled target_mode: {self.target_mode}")

    def _crop_or_pad_pair(self, x1: np.ndarray, x2: np.ndarray):
        L = min(len(x1), len(x2))
        x1 = x1[:L]
        x2 = x2[:L]

        if self.segment_length is not None:
            if L >= self.segment_length:
                if self.random_crop:
                    start = self.py_random.randint(0, L - self.segment_length)
                else:
                    start = (L - self.segment_length) // 2
                end = start + self.segment_length
                x1 = x1[start:end]
                x2 = x2[start:end]
            else:
                pad = self.segment_length - L
                x1 = np.pad(x1, (0, pad))
                x2 = np.pad(x2, (0, pad))

        return x1, x2

    def _normalize_pair_with_noisy_stats(self, noisy: np.ndarray, target: np.ndarray):
        noisy = np.asarray(noisy, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)

        mean = noisy.mean()
        std = noisy.std() + self.normalization_eps

        noisy = (noisy - mean) / std
        target = (target - mean) / std
        return noisy, target

    def __getitem__(self, idx):
        bits = self._sample_bits()
        snr_db = self._sample_snr_db()

        signal_seed = int(self.rng.integers(0, 2**31 - 1))

        noisy_modulator = self._make_modulator(snr_db=snr_db, random_seed=signal_seed)
        clean_modulator = self._make_modulator(snr_db=self.clean_snr_db, random_seed=signal_seed)

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

        noisy_post_demod = self._fm_demodulate(noisy_fm)
        clean_post_demod = self._fm_demodulate(clean_fm)

        target = self._build_target(clean_post_demod)

        noisy_raw, target_raw = self._crop_or_pad_pair(noisy_post_demod, target)

        if self.normalize:
            noisy_norm, target_norm = self._normalize_pair_with_noisy_stats(noisy_raw, target_raw)
        else:
            noisy_norm, target_norm = noisy_raw.copy(), target_raw.copy()

        out = {
            "noisy": torch.from_numpy(noisy_norm).float().unsqueeze(0),         # training input
            "clean": torch.from_numpy(target_norm).float().unsqueeze(0),         # training target
            "noisy_raw": torch.from_numpy(noisy_raw).float().unsqueeze(0),       # metric input
            "clean_raw": torch.from_numpy(target_raw).float().unsqueeze(0),      # metric target
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }

        if self.return_bits:
            out["bits"] = torch.from_numpy(bits).long()

        if self.return_symbols:
            symbols = self.coding.generate_bits_to_symbols(bits, self.constellation_points)
            out["symbols"] = torch.from_numpy(np.asarray(symbols)).float()

        return out