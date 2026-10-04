"""候选动态加权差值 MSE；不是某篇论文的精确实现。"""
import torch
from torch import nn
from torch.nn import functional as F

class DeltaLoss(nn.Module):
    def __init__(self, mode="mse", scale=1.0, alpha_max=1.0, warmup_epochs=10, cap=3.0):
        super().__init__()
        if mode not in ("mse", "static", "dynamic"):
            raise ValueError("未知 loss mode")
        if not all(torch.isfinite(torch.tensor(v)) for v in [scale, alpha_max, cap]) or scale <= 0 or alpha_max < 0 or cap < 0 or warmup_epochs < 1:
            raise ValueError("权重参数非法")
        self.mode, self.scale, self.alpha_max = mode, scale, alpha_max
        self.warmup_epochs, self.cap = warmup_epochs, cap

    def forward(self, prediction, target, epoch=0):
        if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 1 or not prediction.numel():
            raise ValueError("预测和目标必须为相同非空 [B,1]")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("预测或目标非有限")
        error = F.mse_loss(prediction, target, reduction="none")
        alpha = 0 if self.mode == "mse" else self.alpha_max
        if self.mode == "dynamic":
            alpha *= min(max(epoch, 0) / self.warmup_epochs, 1)
        weights = 1 + alpha * (target.detach().abs() / self.scale).clamp(max=self.cap)
        weights = weights / weights.mean()
        return (weights * error).mean(), weights.detach()
