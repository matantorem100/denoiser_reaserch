import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMSNRConditioner(nn.Module):
    """
    Turns scalar SNR [B] into per-channel FiLM parameters:
        gamma, beta in R^{B x C}
    """
    def __init__(self, channels: int, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * channels),
        )

    def forward(self, snr_db: torch.Tensor):
        # snr_db: [B]
        snr_db = snr_db.view(-1, 1)
        params = self.mlp(snr_db)  # [B, 2C]
        gamma, beta = torch.chunk(params, 2, dim=1)
        return gamma, beta


class DilatedResidualBlock1D(nn.Module):
    """
    Residual block with:
    - dilation for larger receptive field
    - GroupNorm for stability
    - FiLM conditioning from SNR
    """
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1, num_groups: int = 8):
        super().__init__()
        padding = dilation * (kernel_size // 2)

        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.norm1 = nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)

        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.norm2 = nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # gamma, beta: [B, C]
        g = gamma.unsqueeze(-1)  # [B, C, 1]
        b = beta.unsqueeze(-1)   # [B, C, 1]

        y = self.conv1(x)
        y = self.norm1(y)
        y = y * (1.0 + g) + b
        y = self.act(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = y * (1.0 + g) + b

        return self.act(x + y)


class FMDenoiserNet(nn.Module):
    """
    Input:
        x      : [B, 1, T]   noisy FM-demodulated waveform
        snr_db : [B]         scalar SNR per sample

    Output:
        denoised waveform: [B, 1, T]

    Main fixes:
    - Larger receptive field via dilated residual blocks
    - SNR conditioning via FiLM
    - Residual learning: output = input + predicted_correction
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        kernel_size: int = 7,
        dilations=(1, 2, 4, 8, 16, 32),
        snr_hidden_dim: int = 32,
        num_groups: int = 8,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(num_groups=min(num_groups, hidden_channels), num_channels=hidden_channels),
            nn.GELU(),
        )

        self.snr_conditioner = FiLMSNRConditioner(
            channels=hidden_channels,
            hidden_dim=snr_hidden_dim
        )

        self.blocks = nn.ModuleList([
            DilatedResidualBlock1D(
                channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=d,
                num_groups=num_groups,
            )
            for d in dilations
        ])

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, T]
        snr_db: [B]
        """
        y = self.input_proj(x)
        gamma, beta = self.snr_conditioner(snr_db)

        for block in self.blocks:
            y = block(y, gamma, beta)

        residual = self.output_proj(y)
        return x + residual