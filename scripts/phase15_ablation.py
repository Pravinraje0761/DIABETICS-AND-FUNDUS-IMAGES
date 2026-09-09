"""
Phase 15 -- controlled degradation ablation (Variants A + B).

Re-runs external inference only (no retraining) for the primary model pair
EfficientNet-B3 (baseline) and RETFound full fine-tune, on APTOS-internal test,
Messidor-2 and IDRiD, under three preprocessing variants:

  baseline   : the committed 224px pipeline, ImageNet normalisation (reproduces
               the Phase 6 numbers -- a control).
  clahe      : CLAHE (LAB L-channel, clip 2.0, 8x8) applied to the 224px image
               before normalisation. Caveat: applied post-resize, an
               approximation of a true preprocessing toggle.
  colour_xfer: per-channel mean/std of each image's fundus pixels mapped to the
               APTOS-train fundus channel statistics (a direct test of the
               Discussion's "RETFound fitted APTOS acquisition/colour statistics"
               hypothesis).

Variant C (input resolution 224 vs 384) needs re-preprocessing from originals and
ViT positional-embedding surgery -- see scripts/PHASE15_ABLATION_SPEC.md.

Outputs (outputs/phase15/):
  <model>__<dataset>__<variant>_preds.csv / _metrics.json   (Phase-6 schema)
  ablation_master.csv   model,dataset,variant,n,accuracy,qwk,macro_f1,referable_auc
  ablation_drop.csv     internal->external QWK drop and delta vs baseline variant
  ablation_summary.md
  aptos_channel_stats.json   cached APTOS-train fundus channel mean/std

Run: .venv/Scripts/python.exe scripts/phase15_ablation.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import FundusDataset  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from predict import infer, load_checkpoint, pick_device  # noqa: E402
from preprocessing import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_DIR / "data_splits"
OUT_DIR = PROJECT_DIR / "outputs" / "phase15"
FUNDUS_TOL = 7  # mean channel value > tol == fundus (matches preprocessing.crop_borders)

CKPTS = {
    "baseline_efficientnet_b3": PROJECT_DIR / "outputs/phase4_baseline/efficientnet_b3_best.pt",
    "retfound_full": PROJECT_DIR / "outputs/phase5_retfound/full_best.pt",
}
DATASETS = {
    "APTOS-internal": (SPLIT_DIR / "aptos_manifest.csv", "test"),
    "Messidor-2": (SPLIT_DIR / "messidor2_manifest.csv", "external"),
    "IDRiD": (SPLIT_DIR / "idrid_manifest.csv", "external"),
}
VARIANTS = ("baseline", "clahe", "colour_xfer")


# --------------------------------------------------------------------------- #
def _fundus_mask(arr: np.ndarray) -> np.ndarray:
    return arr.mean(axis=2) > FUNDUS_TOL


def _channel_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = _fundus_mask(arr)
    if m.sum() < 16:
        m = np.ones(arr.shape[:2], dtype=bool)
    px = arr[m].astype(np.float64)
    return px.mean(axis=0), px.std(axis=0) + 1e-6


def aptos_reference_stats(cache: Path) -> tuple[np.ndarray, np.ndarray]:
    if cache.is_file():
        d = json.loads(cache.read_text())
        return np.array(d["mean"]), np.array(d["std"])
    df = pd.read_csv(SPLIT_DIR / "aptos_manifest.csv")
    df = df[(df["split"] == "train") & df["processed_path"].astype(bool)]
    s1 = np.zeros(3); s2 = np.zeros(3); n = 0
    for rel in df["processed_path"]:
        p = PROJECT_DIR / rel
        if not p.is_file():
            continue
        arr = np.asarray(Image.open(p).convert("RGB"))
        px = arr[_fundus_mask(arr)].astype(np.float64)
        s1 += px.sum(axis=0); s2 += (px ** 2).sum(axis=0); n += len(px)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-6))
    cache.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(),
                                 "n_pixels": int(n), "n_images": int(len(df))}, indent=2))
    return mean, std


# --------------------------------------------------------------------------- #
def _clahe(arr: np.ndarray) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = c.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _colour_xfer(arr: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray) -> np.ndarray:
    mean, std = _channel_stats(arr)
    out = (arr.astype(np.float64) - mean) / std * ref_std + ref_mean
    out[~_fundus_mask(arr)] = 0.0                        # keep the black frame black
    return np.clip(out, 0, 255).astype(np.uint8)


class _ClaheOp:
    """Picklable (needed for Py3.14 spawn/forkserver DataLoader workers)."""

    def __call__(self, im: Image.Image) -> Image.Image:
        return Image.fromarray(_clahe(np.asarray(im)))


class _ColourXferOp:
    def __init__(self, ref_mean, ref_std):
        self.ref_mean = np.asarray(ref_mean)
        self.ref_std = np.asarray(ref_std)

    def __call__(self, im: Image.Image) -> Image.Image:
        return Image.fromarray(_colour_xfer(np.asarray(im), self.ref_mean, self.ref_std))


def make_transform(variant: str, ref):
    ops: list = []
    if variant == "clahe":
        ops.append(_ClaheOp())
    elif variant == "colour_xfer":
        ops.append(_ColourXferOp(ref[0], ref[1]))
    ops += [transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def _referable_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = (y_true >= 2).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, prob[:, 2:].sum(axis=1)))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--models", nargs="+", default=list(CKPTS), choices=list(CKPTS))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    ref = aptos_reference_stats(OUT_DIR / "aptos_channel_stats.json")
    print(f"APTOS-train fundus channel mean {np.round(ref[0], 1)} std {np.round(ref[1], 1)}")

    rows: list[dict] = []
    for mkey in args.models:
        ckpt = CKPTS[mkey]
        if not ckpt.is_file():
            print(f"skip {mkey}: {ckpt} missing")
            continue
        model, _ = load_checkpoint(ckpt, device)
        for variant in args.variants:
            tfm = make_transform(variant, ref)
            for dkey, (manifest, split) in DATASETS.items():
                if not manifest.is_file():
                    continue
                ds = FundusDataset(manifest, split=split, train=False, transform=tfm)
                loader = torch.utils.data.DataLoader(
                    ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
                res = infer(model, loader, device)
                prefix = OUT_DIR / f"{mkey}__{dkey}__{variant}"
                pd.DataFrame({
                    "id_code": res["id_code"], "y_true": res["y_true"],
                    "y_pred": res["y_pred"],
                    **{f"prob_{c}": res["prob"][:, c] for c in range(5)},
                    "entropy": res["entropy"],
                }).to_csv(prefix.with_name(prefix.name + "_preds.csv"), index=False)
                m = compute_metrics(res["y_true"], res["y_pred"], res["prob"])
                m["referable_auc"] = _referable_auc(res["y_true"], res["prob"])
                prefix.with_name(prefix.name + "_metrics.json").write_text(
                    json.dumps(m, indent=2), encoding="utf-8")
                rows.append({"model": mkey, "dataset": dkey, "variant": variant,
                             "n": m["n"], "accuracy": round(m["accuracy"], 4),
                             "qwk": round(m["quadratic_weighted_kappa"], 4),
                             "macro_f1": round(m["macro_f1"], 4),
                             "referable_auc": round(m["referable_auc"], 4)})
                print(f"{mkey:26s} {dkey:15s} {variant:11s} "
                      f"QWK={m['quadratic_weighted_kappa']:.4f} acc={m['accuracy']:.4f}")

    master = pd.DataFrame(rows)
    master.to_csv(OUT_DIR / "ablation_master.csv", index=False)

    drop_rows: list[dict] = []
    for mkey in master["model"].unique():
        sub = master[master["model"] == mkey]
        base_int = {v: sub[(sub.variant == v) & (sub.dataset == "APTOS-internal")]["qwk"].iloc[0]
                    for v in sub.variant.unique()}
        base_drop: dict[str, float] = {}
        for dk in ("Messidor-2", "IDRiD"):
            for v in sub.variant.unique():
                r = sub[(sub.variant == v) & (sub.dataset == dk)]
                if r.empty:
                    continue
                drop = base_int[v] - r["qwk"].iloc[0]
                if v == "baseline":
                    base_drop[dk] = drop
                drop_rows.append({"model": mkey, "external": dk, "variant": v,
                                  "qwk_external": r["qwk"].iloc[0], "qwk_drop": round(drop, 4),
                                  "delta_drop_vs_baseline_variant":
                                      round(drop - base_drop.get(dk, drop), 4)})
    pd.DataFrame(drop_rows).to_csv(OUT_DIR / "ablation_drop.csv", index=False)

    lines = ["# Phase 15 -- degradation ablation (Variants A + B)", "",
             "QWK by model x dataset x variant:", "",
             "| Model | Dataset | variant | QWK | acc | referable AUC |",
             "|---|---|---|--:|--:|--:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['dataset']} | {r['variant']} | "
                     f"{r['qwk']:.3f} | {r['accuracy']:.3f} | {r['referable_auc']:.3f} |")
    lines += ["", "Internal->external QWK drop, and change vs the baseline variant "
              "(negative = the variant *recovers* generalisation):", "",
              "| Model | External | variant | QWK drop | delta vs baseline |",
              "|---|---|---|--:|--:|"]
    for r in drop_rows:
        lines.append(f"| {r['model']} | {r['external']} | {r['variant']} | "
                     f"{r['qwk_drop']:.3f} | {r['delta_drop_vs_baseline_variant']:+.3f} |")
    (OUT_DIR / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_DIR}/ablation_master.csv, ablation_drop.csv, ablation_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
