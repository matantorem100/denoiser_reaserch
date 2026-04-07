import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedDenoiseLoss(nn.Module):
    """
    Reduces oversmoothing by combining:
    - MSE loss      : keeps global fidelity
    - L1 loss       : preserves sharper transitions better than pure MSE
    - Derivative L1 : penalizes slope mismatch and helps preserve waveform edges

    Total:
        loss = w_mse * MSE + w_l1 * L1 + w_diff * diff_L1
    """
    def __init__(self, w_mse: float = 1.0, w_l1: float = 0.2, w_diff: float = 0.1):
        super().__init__()
        self.w_mse = w_mse
        self.w_l1 = w_l1
        self.w_diff = w_diff

    @staticmethod
    def derivative_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_d = pred[..., 1:] - pred[..., :-1]
        target_d = target[..., 1:] - target[..., :-1]
        return F.l1_loss(pred_d, target_d)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        l1 = F.l1_loss(pred, target)
        diff = self.derivative_l1(pred, target)

        return self.w_mse * mse + self.w_l1 * l1 + self.w_diff * diff