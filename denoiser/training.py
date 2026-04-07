import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from denoiser.dataset import PamOverFmDataset
from denoiser.losses import CombinedDenoiseLoss
from denoiser.model import FMDenoiserNet
from synthetic_dataset_creation.channel.channel import ChannelConfig
from synthetic_dataset_creation.constellation.constellation import ConstellationConfig
from synthetic_dataset_creation.modulator.modulator import FMConfig, ModulatorConfig
from synthetic_dataset_creation.pulse_shape.pulse_shape import PulseShapeConfig, PulseShape


def build_configs():
    pulse_shape_config = PulseShapeConfig(
        pulse_shape_type="rect",
        normalization_type="cpfsk",
    )
    constellation_config = ConstellationConfig(
        constellation_type="PAM",
        constellation_order=2,
    )
    channel_config = ChannelConfig(
        channel_type="awgn",
        snr_db=30.0,
        random_seed=None,
    )
    fm_config = FMConfig(
        frequency_offset=0.0,
        frequency_sensitivity=1.0,
        amplitude=1.0,
        normalize_message=True,
    )

    modulator_config = ModulatorConfig(
        pulse_shape_config=pulse_shape_config,
        constellation_config=constellation_config,
        channel_config=channel_config,
        fm_config=fm_config,
    )
    return modulator_config


def build_dataloaders(modulator_config):
    train_dataset = PamOverFmDataset(
        modulator_config=modulator_config,
        split="train",
        dataset_size=100_000,
        n_bits=128,
        symbol_time=(1/1000),
        sample_rate=10000,
        snr_db_min=0.0,
        snr_db_max=14.0,
        fixed_snr_db=30,
        segment_seconds=None,
        random_crop=True,
        normalize=False,
        target_mode="post_demod_clean",
        clean_snr_db=120.0,
        return_bits=True,
        return_symbols=True,
        seed=1234,
    )

    val_dataset = PamOverFmDataset(
        modulator_config=modulator_config,
        split="val",
        dataset_size=20,
        n_bits=100_000,
        symbol_time=0.1,
        sample_rate=100.0,
        snr_db_min=0.0,
        snr_db_max=14.0,
        fixed_snr_db=None,
        segment_seconds=None,
        random_crop=False,
        normalize=False,
        target_mode="post_demod_clean",
        clean_snr_db=120.0,
        return_bits=True,
        return_symbols=True,
        seed=2026,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def build_matched_filter_kernel(
    pulse_shape_config: PulseShapeConfig,
    sample_rate: float,
    symbol_time: float,
    device: torch.device,
):
    pulse = PulseShape(pulse_shape_config).generate_pulse_shape(sample_rate, symbol_time)
    pulse = np.asarray(pulse, dtype=np.float32)
    kernel = torch.from_numpy(pulse[::-1].copy()).float().view(1, 1, -1).to(device)
    return kernel, pulse


def apply_matched_filter(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    k = kernel.shape[-1]
    return F.conv1d(x, kernel, padding=k - 1)


def sample_symbol_outputs(y: torch.Tensor, sps: int, n_symbols: int, mf_delay: int) -> torch.Tensor:
    z = y[:, 0, mf_delay::sps]
    z = z[:, :n_symbols]
    return z


def pam_hard_decision_to_bits(samples: torch.Tensor) -> torch.Tensor:
    return (samples > 0).long()


def compute_bit_errors_2pam(
    mf_samples: torch.Tensor,
    bits: torch.Tensor,
    pulse_energy: float,
):
    mf_samples_norm = mf_samples / pulse_energy
    pred_bits = pam_hard_decision_to_bits(mf_samples_norm)

    target_bits = bits.long()
    if pred_bits.shape != target_bits.shape:
        raise ValueError(
            f"Shape mismatch in BER: pred_bits {pred_bits.shape}, target_bits {target_bits.shape}"
        )

    bit_errors = (pred_bits != target_bits).sum().item()
    total_bits = target_bits.numel()
    return bit_errors, total_bits


def compute_waveform_snr_db(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    signal_power = torch.mean(reference ** 2, dim=(1, 2))
    noise_power = torch.mean((estimate - reference) ** 2, dim=(1, 2))
    snr_db = 10.0 * torch.log10((signal_power + eps) / (noise_power + eps))
    return snr_db.mean().item()


def denormalize_with_raw_stats(
    pred_norm: torch.Tensor,
    noisy_raw: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mean_raw = noisy_raw.mean(dim=(1, 2), keepdim=True)
    std_raw = noisy_raw.std(dim=(1, 2), keepdim=True, unbiased=False) + eps
    pred_raw = pred_norm * std_raw + mean_raw
    return pred_raw


def plot_epoch_example(
    epoch: int,
    noisy_raw: torch.Tensor,
    denoised_raw: torch.Tensor,
    clean_raw: torch.Tensor,
    mf_noisy_raw: torch.Tensor,
    mf_denoised_raw: torch.Tensor,
    mf_clean_raw: torch.Tensor,
    save_dir: Path,
    max_points_raw: int = 2000,
    max_points_mf: int = 2000,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    noisy_np = noisy_raw[0, 0].detach().cpu().numpy()
    denoised_np = denoised_raw[0, 0].detach().cpu().numpy()
    clean_np = clean_raw[0, 0].detach().cpu().numpy()

    mf_noisy_np = mf_noisy_raw[0, 0].detach().cpu().numpy()
    mf_denoised_np = mf_denoised_raw[0, 0].detach().cpu().numpy()
    mf_clean_np = mf_clean_raw[0, 0].detach().cpu().numpy()

    raw_len = min(max_points_raw, len(noisy_np))
    mf_len = min(max_points_mf, len(mf_noisy_np))

    fig, axes = plt.subplots(2, 1, figsize=(16, 9))

    axes[0].plot(noisy_np[:raw_len], label="Noisy", linewidth=1.0)
    axes[0].plot(denoised_np[:raw_len], label="Enhanced", linewidth=1.0)
    axes[0].plot(clean_np[:raw_len], label="Clean", linewidth=1.0)
    axes[0].set_title(f"Epoch {epoch} - Raw post-demod signals")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(mf_noisy_np[:mf_len], label="Noisy + MF", linewidth=1.0)
    axes[1].plot(mf_denoised_np[:mf_len], label="Enhanced + MF", linewidth=1.0)
    axes[1].plot(mf_clean_np[:mf_len], label="Clean + MF", linewidth=1.0)
    axes[1].set_title(f"Epoch {epoch} - Matched-filtered signals")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()

    out_path = save_dir / f"epoch_{epoch:03d}_signals.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.001)
    print(f"Saved example plot to {out_path}")


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    mf_kernel,
    sps,
    mf_delay,
    pulse_energy,
    waveform_criterion,
    train: bool,
    lambda_wave: float = 0.3,
    lambda_mf: float = 0.4,
    lambda_sym: float = 0.3,
    grad_clip: float = 1.0,
    epoch_idx: int = 0,
    num_epochs: int = 0,
):
    if train:
        model.train()
        desc = f"Train {epoch_idx:03d}/{num_epochs:03d}"
    else:
        model.eval()
        desc = f"Val   {epoch_idx:03d}/{num_epochs:03d}"

    total_loss = 0.0
    total_wave = 0.0
    total_mf = 0.0
    total_sym = 0.0

    total_noisy_bit_errors = 0
    total_denoised_bit_errors = 0
    total_bits = 0

    total_snr_in = 0.0
    total_snr_out = 0.0
    n_batches = 0

    example_batch = None

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    start_time = time.time()

    for batch_idx, batch in enumerate(pbar):
        noisy = batch["noisy"].to(device, non_blocking=True)            # normalized
        clean = batch["clean"].to(device, non_blocking=True)            # normalized
        noisy_raw = batch["noisy_raw"].to(device, non_blocking=True)    # raw
        clean_raw = batch["clean_raw"].to(device, non_blocking=True)    # raw
        symbols = batch["symbols"].to(device, non_blocking=True)
        bits = batch["bits"].to(device, non_blocking=True)
        snr_db = batch["snr_db"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            denoised = model(noisy, snr_db)

            mf_denoised = apply_matched_filter(denoised, mf_kernel)
            mf_clean = apply_matched_filter(clean, mf_kernel)

            # n_symbols = symbols.shape[1]
            # mf_denoised_s = sample_symbol_outputs(
            #     mf_denoised, sps=sps, n_symbols=n_symbols, mf_delay=mf_delay
            # )
            # mf_clean_s = sample_symbol_outputs(
            #     mf_clean, sps=sps, n_symbols=n_symbols, mf_delay=mf_delay
            # )

            # loss_wave = waveform_criterion(denoised, clean)
            # loss_mf = F.mse_loss(mf_denoised, mf_clean)
            # loss_sym = F.mse_loss(mf_denoised_s, mf_clean_s)
            #
            # loss = (
            #     lambda_wave * loss_wave
            #     + lambda_mf * loss_mf
            #     + lambda_sym * loss_sym
            # )
            loss = F.mse_loss(mf_denoised, mf_clean)

            if train:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        denoised_raw = denormalize_with_raw_stats(
            pred_norm=denoised.detach(),
            noisy_raw=noisy_raw,
        )

        mf_noisy_raw = apply_matched_filter(noisy_raw, mf_kernel)
        mf_denoised_raw = apply_matched_filter(denoised_raw, mf_kernel)
        mf_clean_raw = apply_matched_filter(clean_raw, mf_kernel)

        mf_noisy_raw_s = sample_symbol_outputs(
            mf_noisy_raw, sps=sps, n_symbols=symbols.shape[1], mf_delay=mf_delay
        )
        mf_denoised_raw_s = sample_symbol_outputs(
            mf_denoised_raw, sps=sps, n_symbols=symbols.shape[1], mf_delay=mf_delay
        )

        noisy_bit_errors, noisy_total_bits = compute_bit_errors_2pam(
            mf_samples=mf_noisy_raw_s.detach(),
            bits=bits.detach(),
            pulse_energy=pulse_energy,
        )

        denoised_bit_errors, denoised_total_bits = compute_bit_errors_2pam(
            mf_samples=mf_denoised_raw_s.detach(),
            bits=bits.detach(),
            pulse_energy=pulse_energy,
        )

        if noisy_total_bits != denoised_total_bits:
            raise ValueError("Mismatch between noisy and denoised total bits")

        total_noisy_bit_errors += noisy_bit_errors
        total_denoised_bit_errors += denoised_bit_errors
        total_bits += noisy_total_bits

        snr_in_db_val = compute_waveform_snr_db(noisy_raw.detach(), clean_raw.detach())
        snr_out_db_val = compute_waveform_snr_db(denoised_raw.detach(), clean_raw.detach())

        total_snr_in += snr_in_db_val
        total_snr_out += snr_out_db_val

        total_loss += loss.item()
        # total_wave += loss_wave.item()
        # total_mf += loss_mf.item()
        # total_sym += loss_sym.item()
        n_batches += 1

        if (not train) and example_batch is None:
            example_batch = {
                "noisy_raw": noisy_raw[:1].detach().cpu(),
                "denoised_raw": denoised_raw[:1].detach().cpu(),
                "clean_raw": clean_raw[:1].detach().cpu(),
                "mf_noisy_raw": mf_noisy_raw[:1].detach().cpu(),
                "mf_denoised_raw": mf_denoised_raw[:1].detach().cpu(),
                "mf_clean_raw": mf_clean_raw[:1].detach().cpu(),
                "snr_db": snr_db[:1].detach().cpu(),
            }

        avg_loss = total_loss / n_batches
        ber_noisy = total_noisy_bit_errors / total_bits
        ber_denoised = total_denoised_bit_errors / total_bits
        avg_snr_in = total_snr_in / n_batches
        avg_snr_out = total_snr_out / n_batches
        elapsed = time.time() - start_time

        pbar.set_postfix({
            "loss": f"{avg_loss:.4e}",
            "ber_noisy": f"{ber_noisy:.3e}",
            "ber_den": f"{ber_denoised:.3e}",
            "snr_in": f"{avg_snr_in:.2f}",
            "snr_out": f"{avg_snr_out:.2f}",
            "gain": f"{(avg_snr_out - avg_snr_in):.2f}dB",
            "sec": f"{elapsed:.1f}",
        })

    avg_snr_in = total_snr_in / n_batches
    avg_snr_out = total_snr_out / n_batches

    metrics = {
        "loss": total_loss / n_batches,
        "loss_wave": total_wave / n_batches,
        "loss_mf": total_mf / n_batches,
        "loss_sym": total_sym / n_batches,
        "ber_noisy": total_noisy_bit_errors / total_bits,
        "ber_denoised": total_denoised_bit_errors / total_bits,
        "snr_in_db": avg_snr_in,
        "snr_out_db": avg_snr_out,
        "snr_gain_db": avg_snr_out - avg_snr_in,
        "total_bits": total_bits,
    }

    return metrics, example_batch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    modulator_config = build_configs()

    symbol_time = 0.1
    sample_rate = 100.0
    sps = round(symbol_time * sample_rate)

    mf_kernel, pulse = build_matched_filter_kernel(
        pulse_shape_config=modulator_config.pulse_shape_config,
        sample_rate=sample_rate,
        symbol_time=symbol_time,
        device=device,
    )
    pulse_energy = float(np.sum(np.asarray(pulse, dtype=np.float32) ** 2))
    mf_delay = len(pulse) - 1

    print("sps:", sps)
    print("pulse_len:", len(pulse))
    print("mf_delay:", mf_delay)
    print("pulse_energy:", pulse_energy)

    train_loader, val_loader = build_dataloaders(modulator_config)

    model = FMDenoiserNet(
        in_channels=1,
        hidden_channels=64,
        kernel_size=7,
        dilations=(1, 2, 4, 8, 16, 32),
        snr_hidden_dim=32,
        num_groups=8,
    ).to(device)

    waveform_criterion = CombinedDenoiseLoss(
        w_mse=1.0,
        w_l1=0.2,
        w_diff=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    num_epochs = 10
    best_val_ber = float("inf")
    save_dir = Path("checkpoints")
    plots_dir = Path("plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    total_train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics, _ = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mf_kernel=mf_kernel,
            sps=sps,
            mf_delay=mf_delay,
            pulse_energy=pulse_energy,
            waveform_criterion=waveform_criterion,
            train=True,
            lambda_wave=0.3,
            lambda_mf=0.4,
            lambda_sym=0.3,
            grad_clip=1.0,
            epoch_idx=epoch,
            num_epochs=num_epochs,
        )

        val_metrics, val_example = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            mf_kernel=mf_kernel,
            sps=sps,
            mf_delay=mf_delay,
            pulse_energy=pulse_energy,
            waveform_criterion=waveform_criterion,
            train=False,
            lambda_wave=0.3,
            lambda_mf=0.4,
            lambda_sym=0.3,
            grad_clip=1.0,
            epoch_idx=epoch,
            num_epochs=num_epochs,
        )

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.6f} "
            f"(wave={train_metrics['loss_wave']:.6f}, mf={train_metrics['loss_mf']:.6f}, sym={train_metrics['loss_sym']:.6f}) | "
            f"train BER noisy={train_metrics['ber_noisy']:.6e} denoised={train_metrics['ber_denoised']:.6e} | "
            f"train SNR in={train_metrics['snr_in_db']:.2f}dB out={train_metrics['snr_out_db']:.2f}dB gain={train_metrics['snr_gain_db']:.2f}dB | "
            f"val loss={val_metrics['loss']:.6f} "
            f"(wave={val_metrics['loss_wave']:.6f}, mf={val_metrics['loss_mf']:.6f}, sym={val_metrics['loss_sym']:.6f}) | "
            f"val BER noisy={val_metrics['ber_noisy']:.6e} denoised={val_metrics['ber_denoised']:.6e} | "
            f"val SNR in={val_metrics['snr_in_db']:.2f}dB out={val_metrics['snr_out_db']:.2f}dB gain={val_metrics['snr_gain_db']:.2f}dB | "
            f"val bits={val_metrics['total_bits']} | "
            f"lr={current_lr:.3e} | "
            f"time={epoch_time:.1f}s"
        )

        if val_example is not None:
            print(f"Validation example SNR_db label: {val_example['snr_db'].item():.3f}")
            plot_epoch_example(
                epoch=epoch,
                noisy_raw=val_example["noisy_raw"],
                denoised_raw=val_example["denoised_raw"],
                clean_raw=val_example["clean_raw"],
                mf_noisy_raw=val_example["mf_noisy_raw"],
                mf_denoised_raw=val_example["mf_denoised_raw"],
                mf_clean_raw=val_example["mf_clean_raw"],
                save_dir=plots_dir,
                max_points_raw=2000,
                max_points_mf=2000,
            )

        last_ckpt = save_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
            last_ckpt,
        )

        if val_metrics["ber_denoised"] < best_val_ber:
            best_val_ber = val_metrics["ber_denoised"]
            best_ckpt = save_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                best_ckpt,
            )
            print(f"Saved new best checkpoint to {best_ckpt}")

    total_time = time.time() - total_train_start
    print(f"Training finished in {total_time:.1f} seconds")
    print(f"Best val denoised BER = {best_val_ber:.6e}")


if __name__ == "__main__":
    main()