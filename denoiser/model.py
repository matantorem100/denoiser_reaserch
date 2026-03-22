import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class FMDenoiserNet(nn.Module):
    """
    Input:  [B, 1, T]  noisy FM-demodulated waveform
    Output: [B, 1, T]  denoised waveform

    Residual learning:
        output = input + predicted_residual
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 7,
    ):
        super().__init__()
        padding = kernel_size // 2

        layers = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_blocks):
            layers.append(ResidualBlock1D(hidden_channels, kernel_size=kernel_size))

        layers.append(nn.Conv1d(hidden_channels, 1, kernel_size=kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.net(x)
        return x + residual