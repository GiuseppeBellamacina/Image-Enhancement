"""
Loss functions for Image Enhancement.

Supports L1, L2, Charbonnier (used in the NAFNet paper), and combined variants
with optional SSIM structural component.  All loss classes return a tuple of
``(loss_tensor, metrics_dict)`` so the training loop can log individual terms.

Usage via factory:
    >>> from src.losses.combined_loss import get_criterion
    >>> criterion = get_criterion(config)    # reads config["loss_type"]
"""

import torch
import torch.nn as nn
from pytorch_msssim import SSIM


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _ssim_from_normalized(pred, target):
    """Compute SSIM value after converting from [-1,1] to [0,1] range."""
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2
    ssim_module = SSIM(data_range=1.0, size_average=True, channel=3)
    return ssim_module(pred_01, target_01)


# ---------------------------------------------------------------------------
#  Core loss modules
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Charbonnier penalty  √(x² + ε²).

    Used in the original NAFNet paper (ECCV 2022).  Unlike L1 it is
    differentiable everywhere, which stabilises training and produces
    visually sharper outputs at high noise levels.

    When ``alpha=1.0, beta=0.0`` this is a pure Charbonnier loss.
    When ``beta > 0`` it adds a weighted SSIM structural term.

    Args:
        eps: Smoothing constant (default 1e-3, as in the NAFNet paper).
        alpha: Weight for the Charbonnier term.
        beta:  Weight for the SSIM term (0 = disabled).
    """

    def __init__(self, eps: float = 1e-3, alpha: float = 1.0, beta: float = 0.0):
        super().__init__()
        self.eps2 = eps ** 2
        self.alpha = alpha
        self.beta = beta

        if beta > 0:
            self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=3)
        else:
            self.ssim_module = None

    def forward(self, pred, target):
        diff = pred - target
        charb = torch.sqrt(diff * diff + self.eps2).mean()

        # Always compute SSIM for monitoring, even when beta == 0
        pred_01 = (pred + 1) / 2
        target_01 = (target + 1) / 2

        if self.ssim_module is not None:
            ssim_val = self.ssim_module(pred_01, target_01)
        else:
            with torch.no_grad():
                ssim_val = _ssim_from_normalized(pred, target)

        ssim_loss = 1 - ssim_val

        if self.beta > 0:
            total = self.alpha * charb + self.beta * ssim_loss
        else:
            total = charb

        return total, {
            "l1": charb.item(),        # logged as "l1" for compatibility
            "ssim": ssim_val.item(),
            "total": total.item(),
        }


class CombinedLoss(nn.Module):
    """Combined L1 + SSIM Loss for image enhancement.

    Args:
        alpha: Weight for L1 loss (default: 0.84)
        beta: Weight for SSIM loss (default: 0.16)
    """

    def __init__(self, alpha=0.84, beta=0.16):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)

        pred_01 = (pred + 1) / 2
        target_01 = (target + 1) / 2
        ssim_val = self.ssim_loss(pred_01, target_01)
        ssim_loss = 1 - ssim_val

        total_loss = self.alpha * l1 + self.beta * ssim_loss

        return total_loss, {
            "l1": l1.item(),
            "ssim": ssim_val.item(),
            "total": total_loss.item(),
        }


class L1Loss(nn.Module):
    """Simple L1 Loss wrapper."""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        loss = self.loss(pred, target)

        # Compute SSIM for monitoring
        with torch.no_grad():
            ssim_val = _ssim_from_normalized(pred, target)

        return loss, {
            "l1": loss.item(),
            "ssim": ssim_val.item(),
            "total": loss.item(),
        }


class L2Loss(nn.Module):
    """Simple L2/MSE Loss wrapper."""

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        return loss, {"mse": loss.item(), "total": loss.item()}


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def get_criterion(config: dict) -> nn.Module:
    """Create a loss function from the training config dictionary.

    Reads ``config["loss_type"]`` to select the loss family.  Falls back to
    ``"combined"`` (L1+SSIM) when the key is missing, for backward
    compatibility with existing notebooks.

    Supported values for ``loss_type``:
        * ``"charbonnier"``     — Charbonnier (+ optional SSIM via beta)
        * ``"combined"``        — L1 + SSIM  (legacy default)
        * ``"l1"``              — Pure L1
        * ``"l2"``              — Pure L2 / MSE

    Args:
        config: Training configuration dict.  Relevant keys:
            - loss_type (str): Loss family selector.
            - loss_alpha (float): Weight for pixel-wise term.
            - loss_beta  (float): Weight for SSIM term (0 = off).

    Returns:
        A ``nn.Module`` whose ``forward(pred, target)`` returns
        ``(loss_tensor, metrics_dict)``.
    """
    loss_type = config.get("loss_type", "combined").lower()
    alpha = config.get("loss_alpha", 1.0)
    beta = config.get("loss_beta", 0.0)

    if loss_type == "charbonnier":
        print(f"📐 Loss: Charbonnier (α={alpha}, β={beta})")
        return CharbonnierLoss(eps=1e-3, alpha=alpha, beta=beta)

    elif loss_type == "combined":
        print(f"📐 Loss: Combined L1+SSIM (α={alpha}, β={beta})")
        return CombinedLoss(alpha=alpha, beta=beta)

    elif loss_type == "l1":
        print("📐 Loss: L1")
        return L1Loss()

    elif loss_type == "l2":
        print("📐 Loss: L2 / MSE")
        return L2Loss()

    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            f"Supported: 'charbonnier', 'combined', 'l1', 'l2'"
        )


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

def test_losses():
    """Test all loss functions."""
    batch_size = 4
    pred = torch.randn(batch_size, 3, 128, 128) * 0.5
    target = torch.randn(batch_size, 3, 128, 128) * 0.5

    # --- Charbonnier pure ---
    charb = CharbonnierLoss()
    loss, m = charb(pred, target)
    print(f"Charbonnier:       total={m['total']:.4f}  l1={m['l1']:.4f}  ssim={m['ssim']:.4f}")

    # --- Charbonnier + SSIM ---
    charb_ssim = CharbonnierLoss(alpha=0.95, beta=0.05)
    loss, m = charb_ssim(pred, target)
    print(f"Charb+SSIM(0.05):  total={m['total']:.4f}  l1={m['l1']:.4f}  ssim={m['ssim']:.4f}")

    # --- Combined L1+SSIM ---
    comb = CombinedLoss()
    loss, m = comb(pred, target)
    print(f"Combined L1+SSIM:  total={m['total']:.4f}  l1={m['l1']:.4f}  ssim={m['ssim']:.4f}")

    # --- L1 pure ---
    l1 = L1Loss()
    loss, m = l1(pred, target)
    print(f"L1:                total={m['total']:.4f}  l1={m['l1']:.4f}  ssim={m['ssim']:.4f}")

    # --- L2 ---
    l2 = L2Loss()
    loss, m = l2(pred, target)
    print(f"L2:                total={m['total']:.4f}")

    # --- Factory ---
    print("\nFactory tests:")
    for lt in ["charbonnier", "combined", "l1", "l2"]:
        cfg = {"loss_type": lt, "loss_alpha": 0.9, "loss_beta": 0.1}
        crit = get_criterion(cfg)
        loss, m = crit(pred, target)
        print(f"  {lt:15s} → total={m['total']:.4f}")

    print("\n✅ All loss functions test passed!")


if __name__ == "__main__":
    test_losses()
