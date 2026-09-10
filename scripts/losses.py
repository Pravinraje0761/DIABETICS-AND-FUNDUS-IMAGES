"""
Shared loss functions for the training scripts.

Only one non-default loss is used in the study: multi-class focal loss, for the
Phase 17 class-weighting / loss-function robustness check
(scripts/phase17_loss_sensitivity.py). The primary models use weighted
cross-entropy; this file exists so `train_baseline.py --loss focal` and
`train_retfound.py --loss focal` share exactly one implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017), CE special-case at gamma=0.

    FL = - alpha_y * (1 - p_y)^gamma * log p_y

    Parameters
    ----------
    gamma : float
        Focusing parameter. gamma=0 reduces to (optionally weighted) CE.
    weight : Tensor | None
        Per-class alpha weights (same semantics as ``nn.CrossEntropyLoss``'s
        ``weight``); pass the inverse-frequency vector to keep the class
        re-balancing while adding the focal term, or None for no re-weighting.
    label_smoothing : float
        Applied to the cross-entropy term before the focal modulation, matching
        ``nn.CrossEntropyLoss`` semantics.
    reduction : {"mean", "sum", "none"}
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.dtype) if self.weight is not None else None
        ce = F.cross_entropy(logits, target, weight=w,
                             label_smoothing=self.label_smoothing,
                             reduction="none")
        # p_y for the true class; detach so the focal weight is a modulation, not
        # a second gradient path (standard practice).
        logp = F.log_softmax(logits, dim=1)
        p_y = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp().clamp_(1e-6, 1.0)
        focal = (1.0 - p_y).pow(self.gamma).detach()
        loss = focal * ce
        if self.reduction == "mean":
            # normalise by summed alpha weight when weighting, else by count,
            # to match nn.CrossEntropyLoss(weight=...) reduction.
            if w is not None:
                return loss.sum() / w[target].sum().clamp_min(1e-8)
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(loss: str, class_weights: torch.Tensor | None,
                    label_smoothing: float = 0.0,
                    focal_gamma: float = 2.0) -> nn.Module:
    """`loss` is 'ce' or 'focal'. Keeps the two trainers' wiring identical."""
    if loss == "ce":
        return nn.CrossEntropyLoss(weight=class_weights,
                                   label_smoothing=label_smoothing)
    if loss == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights,
                         label_smoothing=label_smoothing)
    raise ValueError(f"unknown loss {loss!r} (expected 'ce' or 'focal')")
