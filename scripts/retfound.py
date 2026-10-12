"""
RETFound backbone (Phase 5).

RETFound (Zhou et al., Nature 2023) is a ViT-Large/16 masked-autoencoder
pretrained on ~1.6M retinal images. Downstream it is used exactly like a timm
ViT-L: load the MAE encoder weights, attach a fresh linear head, fine-tune.

We build the backbone with timm rather than cloning the RETFound training repo -
the architectures are identical (`vit_large_patch16_224`, global average pool).

    m = build_retfound_backbone(num_classes=5)
    report = load_retfound_weights(m, "weights/RETFound_mae_natureCFP.pth")

The weights are gated on Hugging Face - see docs/phase_reports/PHASE5_SETUP.md. This module does no
downloading; it only loads a local .pth.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

ARCH_NAME = "retfound_vit_large_patch16"
TIMM_MODEL = "vit_large_patch16_224"


def build_retfound_backbone(
    num_classes: int = 5,
    drop_path: float = 0.2,
    global_pool: bool = True,
    img_size: int | None = None,
) -> nn.Module:
    import timm  # local import: heavy, and only Phase 5 needs it

    kwargs = dict(
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path,
        global_pool="avg" if global_pool else "token",
    )
    if img_size is not None:
        # Non-native input size: timm resizes the model's own pos_embed grid on
        # build; loading a 224-trained checkpoint then needs _interpolate_pos_embed
        # (Phase 15 Variant C - input-resolution sensitivity, not a fair retrain).
        kwargs["img_size"] = img_size
    model = timm.create_model(TIMM_MODEL, **kwargs)
    return model


def _unwrap(state: dict) -> dict:
    for key in ("model", "state_dict", "module"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def _interpolate_pos_embed(model: nn.Module, state: dict) -> None:
    """Resize pretrained pos_embed if the token grid differs (rare at 224)."""
    if "pos_embed" not in state:
        return
    pe = state["pos_embed"]
    model_pe = model.pos_embed
    if pe.shape == model_pe.shape:
        return
    num_extra = model_pe.shape[1] - int((model_pe.shape[1] - 1) ** 0.5) ** 2  # cls tokens
    n_old = pe.shape[1] - num_extra
    n_new = model_pe.shape[1] - num_extra
    size_old, size_new = int(math.sqrt(n_old)), int(math.sqrt(n_new))
    extra = pe[:, :num_extra]
    grid = pe[:, num_extra:].reshape(1, size_old, size_old, -1).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(
        grid, size=(size_new, size_new), mode="bicubic", align_corners=False
    )
    grid = grid.permute(0, 2, 3, 1).reshape(1, n_new, -1)
    state["pos_embed"] = torch.cat([extra, grid], dim=1)


def load_retfound_weights(model: nn.Module, path: str | Path) -> dict:
    """Load MAE encoder weights into `model` (strict=False). Return a report."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"RETFound weights not found at {path}. See docs/phase_reports/PHASE5_SETUP.md - the "
            "checkpoint is gated on Hugging Face and must be downloaded manually."
        )
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = _unwrap(raw)

    # Drop MAE-decoder / masking / pretrained-head keys - not part of the encoder.
    drop_prefixes = ("decoder", "mask_token", "head.", "fc.")
    state = {k: v for k, v in state.items() if not k.startswith(drop_prefixes)}

    # The MAE checkpoint carries a pre-pool `norm` (used with the CLS token).
    # timm's global-average-pool ViT uses `fc_norm` for the same role. Reuse the
    # pretrained LayerNorm as `fc_norm` rather than training it from scratch -
    # matters for linear probing, where the head sees fc_norm's output directly.
    if "fc_norm.weight" not in state and "norm.weight" in state:
        state["fc_norm.weight"] = state.pop("norm.weight")
        state["fc_norm.bias"] = state.pop("norm.bias")

    _interpolate_pos_embed(model, state)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state)
    result = model.load_state_dict(state, strict=False)

    report = {
        "weights_file": str(path),
        "n_ckpt_tensors": len(ckpt_keys),
        "n_model_tensors": len(model_keys),
        "n_loaded": len(ckpt_keys & model_keys),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }
    # Sanity: the transformer blocks must have loaded. Head / fc_norm missing is fine.
    block_missing = [k for k in result.missing_keys if k.startswith("blocks.")]
    if block_missing:
        raise RuntimeError(
            f"{len(block_missing)} transformer-block tensors did not load from "
            f"{path} - wrong checkpoint? First few: {block_missing[:5]}"
        )
    return report
