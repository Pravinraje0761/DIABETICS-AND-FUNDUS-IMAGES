"""
Single fundus preprocessing pipeline - used identically for APTOS 2019,
Messidor-2 and IDRiD (Phase 3).

Pipeline (deterministic, no randomness):
  1. Convert to RGB.
  2. Tolerance-based border crop - drop near-black rows/cols framing the fundus.
  3. Pad to a centred square (aspect ratio preserved, no distortion).
  4. Resize to 224x224 (RETFound's native input size), bilinear.
  5. Optional CLAHE contrast enhancement (default OFF; needs opencv-python).
  6. Optional circular mask that zeroes the square's corners (default OFF).

Saved preprocessed images stay uint8 RGB [0, 255] so they remain viewable and
reusable. Channel normalisation (ImageNet mean/std, RETFound's expected stats)
is applied later in the Dataset transform, not baked into the pixels here.

Pure numpy + Pillow for the default path - no OpenCV dependency unless CLAHE is
switched on.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# RETFound (rmaphoh/RETFound_MAE) downstream config: 224 px input, ImageNet
# channel statistics. Kept here as the single source of truth.
RETFOUND_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_BORDER_TOL = 7  # 0-255; pixels with mean channel value <= tol are "black frame"


def _to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def crop_borders(arr: np.ndarray, tol: int = DEFAULT_BORDER_TOL) -> np.ndarray:
    """Crop the tightest bounding box around non-black content.

    Robust to fully dark images (returns the input unchanged) and to a few
    stray bright pixels (uses row/col *any* over a brightness mask, which is the
    standard Ben-Graham-style fundus crop).
    """
    if arr.ndim != 3:
        raise ValueError(f"expected HxWx3, got shape {arr.shape}")
    gray = arr.mean(axis=2)
    mask = gray > tol
    if not mask.any():
        return arr
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def pad_to_square(arr: np.ndarray) -> np.ndarray:
    """Zero-pad a HxWx3 array to a centred square."""
    h, w = arr.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, arr.shape[2]), dtype=arr.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = arr
    return out


def apply_circle_mask(arr: np.ndarray) -> np.ndarray:
    """Zero the four corners outside the inscribed circle of a square image."""
    h, w = arr.shape[:2]
    if h != w:
        raise ValueError("circle mask expects a square image")
    r = h / 2.0
    yy, xx = np.ogrid[:h, :w]
    inside = (xx - r + 0.5) ** 2 + (yy - r + 0.5) ** 2 <= r * r
    out = arr.copy()
    out[~inside] = 0
    return out


def _apply_clahe(arr: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel of LAB. Requires opencv-python.

    Kept optional and off by default - if used, report it in Methods and run the
    no-CLAHE ablation (Phase 3 note in the methodology).
    """
    try:
        import cv2  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "CLAHE requested but opencv-python is not installed. "
            "Run: pip install opencv-python"
        ) from exc
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_fundus(
    img: Image.Image,
    size: int = RETFOUND_INPUT_SIZE,
    border_tol: int = DEFAULT_BORDER_TOL,
    circle_crop: bool = False,
    clahe: bool = False,
) -> Image.Image:
    """Run the full pipeline on one image and return a uint8 RGB PIL image."""
    arr = _to_rgb_array(img)
    arr = crop_borders(arr, border_tol)
    arr = pad_to_square(arr)
    out = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
    if clahe:
        out = Image.fromarray(_apply_clahe(np.asarray(out)))
    if circle_crop:
        out = Image.fromarray(apply_circle_mask(np.asarray(out)))
    return out


def preprocess_file(
    src_path,
    dst_path,
    size: int = RETFOUND_INPUT_SIZE,
    border_tol: int = DEFAULT_BORDER_TOL,
    circle_crop: bool = False,
    clahe: bool = False,
) -> None:
    """Preprocess one image file and write a PNG to dst_path."""
    with Image.open(src_path) as im:
        out = preprocess_fundus(
            im, size=size, border_tol=border_tol, circle_crop=circle_crop, clahe=clahe
        )
    out.save(dst_path, format="PNG")
