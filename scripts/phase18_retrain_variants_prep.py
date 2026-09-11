"""
Phase 18 -- training-data prep for TRAINING-TIME (not inference-time) ablations.

Phase 15 (Variants A/B: CLAHE, colour transfer; Variant C: resolution) only
re-runs INFERENCE with the perturbation applied -- a measure of how sensitive an
already-trained model is to a distribution shift at test time. It does not tell
us whether training under the same transform would change what the model
learns. This script prepares APTOS manifests (all three splits: train/val/test)
so scripts/train_baseline.py and scripts/train_retfound.py can be pointed at
them directly via --manifest (+ --img-size for the resolution variant), giving
a genuine from-scratch retrain under each condition.

Variants produced (APTOS only -- Messidor-2/IDRiD stay untouched, evaluated
after training with the EXISTING scripts/phase15_ablation.py transform-hook
machinery, pointed at the new checkpoint, so training and evaluation transforms
match without duplicating any eval code):

  clahe224      preprocessing.preprocess_fundus(..., clahe=True) from SOURCE
                images at 224px -- true preprocessing-time CLAHE (Phase 15's
                inference-only CLAHE is applied post-resize as an
                approximation; this bakes it in before the resize, matching
                what a real CLAHE-preprocessing pipeline would do).
  colourxfer224 colour_ops.colour_xfer() applied to the EXISTING committed
                224px processed images (colour transfer to APTOS-train fundus
                channel stats operates in already-cropped/padded/resized pixel
                space, same convention as Phase 15's inference-time version).
  res384_full   preprocessing.preprocess_fundus(..., size=384) from SOURCE
                images, ALL THREE SPLITS (Phase 15c/Variant C only did this for
                the test/external split, for inference). RETFound trained at
                this size needs no pos_embed interpolation vs a 224 checkpoint,
                because it is trained from scratch at 384 (train_retfound.py
                --img-size 384 builds the backbone with img_size=384 directly
                and the *pretrained* RETFound weights, native 224/14x14 grid,
                are bicubically interpolated ONCE at load time -- see
                retfound.py:load_retfound_weights -- then fine-tuned normally).

Outputs:
  data_processed_clahe224/aptos/*.png          + data_splits/aptos_manifest_clahe224.csv
  data_processed_colourxfer224/aptos/*.png     + data_splits/aptos_manifest_colourxfer224.csv
  data_processed_res384/aptos/*.png (extended) + data_splits/aptos_manifest_res384_full.csv

Run:
  .venv/Scripts/python.exe scripts/phase18_retrain_variants_prep.py
  .venv/Scripts/python.exe scripts/phase18_retrain_variants_prep.py --variants clahe224
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colour_ops import aptos_reference_stats, colour_xfer  # noqa: E402
from preprocessing import preprocess_file  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_DIR / "data_splits"
APTOS_MANIFEST = SPLIT_DIR / "aptos_manifest.csv"
PHASE15_DIR = PROJECT_DIR / "outputs" / "phase15"
VARIANTS = ("clahe224", "colourxfer224", "res384_full")


def _write_manifest(df: pd.DataFrame, rel_paths: list[str], out_manifest: Path) -> None:
    df = df.copy()
    df["processed_path"] = rel_paths
    df.to_csv(out_manifest, index=False)


def prep_clahe224(force: bool) -> None:
    out_manifest = SPLIT_DIR / "aptos_manifest_clahe224.csv"
    if out_manifest.is_file() and not force:
        print(f"clahe224: {out_manifest.name} exists, skip (--force)")
        return
    df = pd.read_csv(APTOS_MANIFEST).reset_index(drop=True)
    dst_dir = PROJECT_DIR / "data_processed_clahe224" / "aptos"
    dst_dir.mkdir(parents=True, exist_ok=True)
    rel_paths, done, missing = [], 0, 0
    for _, row in df.iterrows():
        src = (PROJECT_DIR / str(row["source_path"])).resolve()
        stem = Path(str(row["processed_path"])).stem or str(row["id_code"])
        dst = dst_dir / f"{stem}.png"
        rel_paths.append(str(dst.relative_to(PROJECT_DIR)).replace("\\", "/"))
        if dst.is_file() and not force:
            done += 1
            continue
        if not src.is_file():
            missing += 1
            continue
        preprocess_file(src, dst, size=224, clahe=True)
        done += 1
    _write_manifest(df, rel_paths, out_manifest)
    print(f"clahe224: {done}/{len(df)} images (missing sources: {missing}) "
          f"-> {dst_dir}; wrote {out_manifest.name}")


def prep_colourxfer224(force: bool) -> None:
    out_manifest = SPLIT_DIR / "aptos_manifest_colourxfer224.csv"
    if out_manifest.is_file() and not force:
        print(f"colourxfer224: {out_manifest.name} exists, skip (--force)")
        return
    df = pd.read_csv(APTOS_MANIFEST).reset_index(drop=True)
    dst_dir = PROJECT_DIR / "data_processed_colourxfer224" / "aptos"
    dst_dir.mkdir(parents=True, exist_ok=True)
    ref_mean, ref_std = aptos_reference_stats(
        PHASE15_DIR / "aptos_channel_stats.json", PROJECT_DIR, SPLIT_DIR)
    rel_paths, done, missing = [], 0, 0
    for _, row in df.iterrows():
        # Operates on the EXISTING committed 224px processed image (already
        # cropped/padded/resized), same convention as phase15_ablation.py.
        src = (PROJECT_DIR / str(row["processed_path"])).resolve()
        stem = Path(str(row["processed_path"])).stem or str(row["id_code"])
        dst = dst_dir / f"{stem}.png"
        rel_paths.append(str(dst.relative_to(PROJECT_DIR)).replace("\\", "/"))
        if dst.is_file() and not force:
            done += 1
            continue
        if not src.is_file():
            missing += 1
            continue
        arr = np.asarray(Image.open(src).convert("RGB"))
        Image.fromarray(colour_xfer(arr, ref_mean, ref_std)).save(dst, format="PNG")
        done += 1
    _write_manifest(df, rel_paths, out_manifest)
    print(f"colourxfer224: {done}/{len(df)} images (missing sources: {missing}) "
          f"-> {dst_dir}; wrote {out_manifest.name}")


def prep_res384_full(force: bool) -> None:
    out_manifest = SPLIT_DIR / "aptos_manifest_res384_full.csv"
    if out_manifest.is_file() and not force:
        print(f"res384_full: {out_manifest.name} exists, skip (--force)")
        return
    df = pd.read_csv(APTOS_MANIFEST).reset_index(drop=True)
    dst_dir = PROJECT_DIR / "data_processed_res384" / "aptos"  # shared with Phase 15c
    dst_dir.mkdir(parents=True, exist_ok=True)
    rel_paths, done, missing = [], 0, 0
    for _, row in df.iterrows():
        src = (PROJECT_DIR / str(row["source_path"])).resolve()
        stem = Path(str(row["processed_path"])).stem or str(row["id_code"])
        dst = dst_dir / f"{stem}.png"
        rel_paths.append(str(dst.relative_to(PROJECT_DIR)).replace("\\", "/"))
        if dst.is_file() and not force:
            done += 1
            continue
        if not src.is_file():
            missing += 1
            continue
        preprocess_file(src, dst, size=384)
        done += 1
    _write_manifest(df, rel_paths, out_manifest)
    print(f"res384_full: {done}/{len(df)} images (missing sources: {missing}) "
          f"-> {dst_dir}; wrote {out_manifest.name}  "
          "(all 3 splits; shares data_processed_res384/aptos/ with Phase 15c's "
          "test-only aptos_manifest_res384.csv, filenames match so no collision)")


PREPPERS = {"clahe224": prep_clahe224, "colourxfer224": prep_colourxfer224,
           "res384_full": prep_res384_full}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--force", action="store_true", help="reprocess even if the "
                    "manifest already exists")
    args = ap.parse_args()

    if not APTOS_MANIFEST.is_file():
        print(f"missing {APTOS_MANIFEST} - run phase3 scripts first")
        return 1

    for v in args.variants:
        PREPPERS[v](args.force)

    print("\nNext: point train_baseline.py / train_retfound.py at the new "
          "manifest (+ --img-size 384 for res384_full). See "
          "COLAB_RETRAIN_GUIDE.md for exact commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
