"""
Shared colour-normalisation ops: CLAHE and per-image colour transfer to the
APTOS-train fundus channel statistics.

Extracted from scripts/phase15_ablation.py (Phase 15 Variants A/B, inference-time
only) so the same, tested logic can also be baked into TRAINING data by
scripts/phase18_retrain_variants_prep.py (Phase 18, training-time ablation).
Behaviour is unchanged from the original phase15_ablation.py definitions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FUNDUS_TOL = 7  # mean channel value > tol == fundus (matches preprocessing.crop_borders)


def fundus_mask(arr: np.ndarray) -> np.ndarray:
    return arr.mean(axis=2) > FUNDUS_TOL


def channel_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = fundus_mask(arr)
    if m.sum() < 16:
        m = np.ones(arr.shape[:2], dtype=bool)
    px = arr[m].astype(np.float64)
    return px.mean(axis=0), px.std(axis=0) + 1e-6


def aptos_reference_stats(cache: Path, project_dir: Path,
                          split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """APTOS-train fundus channel mean/std, cached to `cache` (JSON)."""
    if cache.is_file():
        d = json.loads(cache.read_text())
        return np.array(d["mean"]), np.array(d["std"])
    df = pd.read_csv(split_dir / "aptos_manifest.csv")
    df = df[(df["split"] == "train") & df["processed_path"].astype(bool)]
    s1 = np.zeros(3); s2 = np.zeros(3); n = 0
    for rel in df["processed_path"]:
        p = project_dir / rel
        if not p.is_file():
            continue
        arr = np.asarray(Image.open(p).convert("RGB"))
        px = arr[fundus_mask(arr)].astype(np.float64)
        s1 += px.sum(axis=0); s2 += (px ** 2).sum(axis=0); n += len(px)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-6))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(),
                                 "n_pixels": int(n), "n_images": int(len(df))}, indent=2))
    return mean, std


def clahe(arr: np.ndarray) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = c.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def colour_xfer(arr: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray) -> np.ndarray:
    mean, std = channel_stats(arr)
    out = (arr.astype(np.float64) - mean) / std * ref_std + ref_mean
    out[~fundus_mask(arr)] = 0.0  # keep the black frame black
    return np.clip(out, 0, 255).astype(np.uint8)


class ClaheOp:
    """Picklable (needed for Py3.14 spawn/forkserver DataLoader workers)."""

    def __call__(self, im: Image.Image) -> Image.Image:
        return Image.fromarray(clahe(np.asarray(im)))


class ColourXferOp:
    def __init__(self, ref_mean, ref_std):
        self.ref_mean = np.asarray(ref_mean)
        self.ref_std = np.asarray(ref_std)

    def __call__(self, im: Image.Image) -> Image.Image:
        return Image.fromarray(colour_xfer(np.asarray(im), self.ref_mean, self.ref_std))
