"""
Phase 3 - run the single preprocessing pipeline over all three datasets and
write per-dataset manifests.

Same pipeline for APTOS 2019, Messidor-2 and IDRiD (see preprocessing.py). This
consistency is itself a methods point.

Run:
  .venv/Scripts/python.exe scripts/phase3_preprocess.py
  .venv/Scripts/python.exe scripts/phase3_preprocess.py --datasets aptos --force
  .venv/Scripts/python.exe scripts/phase3_preprocess.py --clahe          # ablation

Prereq: scripts/phase3_split_aptos.py has been run (needs aptos_split_all.csv).

Outputs:
  data_processed/{aptos,messidor2,idrid}/<id>.png   224x224 uint8 RGB
  data_splits/{aptos,messidor2,idrid}_manifest.csv
  logs/phase3_preprocess_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import (  # noqa: E402
    DEFAULT_BORDER_TOL,
    RETFOUND_INPUT_SIZE,
    preprocess_file,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_DIR.parent

APTOS_IMG = PAPER_DIR / "aptos2019-blindness-detection" / "train_images"
MESSIDOR_CSV = PAPER_DIR / "archive" / "messidor_data.csv"
MESSIDOR_IMG = PAPER_DIR / "archive" / "messidor-2" / "messidor-2" / "preprocess"
IDRID_CSV = PAPER_DIR / "archive (1)" / "idrid_labels.csv"
IDRID_IMG = PAPER_DIR / "archive (1)" / "Imagenes" / "Imagenes"

SPLIT_DIR = PROJECT_DIR / "data_splits"
PROC_DIR = PROJECT_DIR / "data_processed"
LOG_DIR = PROJECT_DIR / "logs"
for d in (SPLIT_DIR, PROC_DIR, LOG_DIR):
    d.mkdir(exist_ok=True)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _rel(p: Path) -> str:
    """POSIX path relative to PROJECT_DIR (uses '..' for the sibling datasets)."""
    import os

    return Path(os.path.relpath(p, PROJECT_DIR)).as_posix()


def _find_image(folder: Path, stem_or_name: str) -> Path | None:
    """Resolve an id (with or without extension) to an actual file on disk."""
    cand = folder / stem_or_name
    if cand.is_file():
        return cand
    for ext in IMG_EXTS:
        c = folder / f"{Path(stem_or_name).stem}{ext}"
        if c.is_file():
            return c
    return None


# --------------------------------------------------------------------------- #
# Per-dataset row tables (id_code, diagnosis, split, + dataset-specific extras) #
# --------------------------------------------------------------------------- #
def rows_aptos() -> pd.DataFrame:
    split_csv = SPLIT_DIR / "aptos_split_all.csv"
    if not split_csv.is_file():
        raise SystemExit(
            f"missing {split_csv} - run scripts/phase3_split_aptos.py first"
        )
    df = pd.read_csv(split_csv)
    df["id_code"] = df["id_code"].astype(str)
    df["diagnosis"] = df["diagnosis"].astype(int)
    df["src_folder"] = str(APTOS_IMG)
    return df[["id_code", "diagnosis", "split", "src_folder"]]


def rows_messidor() -> pd.DataFrame:
    df = pd.read_csv(MESSIDOR_CSV)
    df.columns = [c.strip() for c in df.columns]
    df["id_code"] = df["id_code"].astype(str)
    df["diagnosis"] = df["diagnosis"].astype(int)
    df["split"] = "external"
    df["src_folder"] = str(MESSIDOR_IMG)
    keep = ["id_code", "diagnosis", "split", "src_folder"]
    for extra in ("adjudicated_dme", "adjudicated_gradable"):
        if extra in df.columns:
            keep.append(extra)
    return df[keep]


def rows_idrid() -> pd.DataFrame:
    # Malformed header: read positionally. col0 id, col1 DR grade, col2 dme risk.
    raw = pd.read_csv(
        IDRID_CSV, header=0, usecols=[0, 1, 2],
        names=["id_code", "diagnosis", "dme_risk"],
    )
    raw = raw.dropna(subset=["id_code"])
    raw["id_code"] = raw["id_code"].astype(str).str.strip()
    raw = raw[raw["id_code"].str.startswith("IDRiD_")].copy()
    raw["diagnosis"] = pd.to_numeric(raw["diagnosis"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["diagnosis"])
    raw["diagnosis"] = raw["diagnosis"].astype(int)
    raw["orig_split"] = raw["id_code"].apply(
        lambda s: "idrid_test" if s.endswith("test") else "idrid_train"
    )
    raw["split"] = "external"
    raw["src_folder"] = str(IDRID_IMG)
    return raw[["id_code", "diagnosis", "split", "src_folder", "dme_risk", "orig_split"]]


DATASETS = {
    "aptos": ("aptos", rows_aptos),
    "messidor": ("messidor2", rows_messidor),
    "idrid": ("idrid", rows_idrid),
}


def process_dataset(key: str, params: dict, force: bool) -> dict:
    out_name, row_fn = DATASETS[key]
    df = row_fn().reset_index(drop=True)
    dst_dir = PROC_DIR / out_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    processed_paths: list[str] = []
    source_paths: list[str] = []
    failures: list[dict] = []
    n_skipped = 0
    t0 = time.time()

    for i, r in df.iterrows():
        folder = Path(r["src_folder"])
        src = _find_image(folder, r["id_code"])
        stem = Path(r["id_code"]).stem
        dst = dst_dir / f"{stem}.png"

        if src is None:
            failures.append({"id_code": r["id_code"], "error": "source image not found"})
            source_paths.append("")
            processed_paths.append("")
            continue

        # Store portable POSIX paths relative to the project dir so manifests
        # are machine-independent and committable. dataset.py resolves them.
        source_paths.append(_rel(src))
        processed_paths.append(_rel(dst))

        if dst.is_file() and not force:
            n_skipped += 1
        else:
            try:
                preprocess_file(
                    src, dst,
                    size=params["size"],
                    border_tol=params["border_tol"],
                    circle_crop=params["circle_crop"],
                    clahe=params["clahe"],
                )
            except Exception as exc:  # noqa: BLE001 - record and continue
                failures.append({"id_code": r["id_code"], "error": repr(exc)})
                processed_paths[-1] = ""

        if (i + 1) % 500 == 0:
            print(f"  [{out_name}] {i + 1}/{len(df)}  ({time.time() - t0:.0f}s)")

    df = df.drop(columns=["src_folder"])
    df["source_path"] = source_paths
    df["processed_path"] = processed_paths
    manifest_path = SPLIT_DIR / f"{out_name}_manifest.csv"
    df.to_csv(manifest_path, index=False)

    n_ok = int((df["processed_path"] != "").sum())
    print(f"  [{out_name}] done: {n_ok} images, {n_skipped} already present, "
          f"{len(failures)} failures -> {manifest_path.relative_to(PROJECT_DIR)}")

    return {
        "output_folder": str(dst_dir.relative_to(PROJECT_DIR)),
        "manifest": str(manifest_path.relative_to(PROJECT_DIR)),
        "n_rows": int(len(df)),
        "n_processed_or_present": n_ok,
        "n_skipped_existing": n_skipped,
        "n_failures": len(failures),
        "failures": failures[:50],
        "split_breakdown": {
            k: int(v) for k, v in df["split"].value_counts().items()
        },
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--size", type=int, default=RETFOUND_INPUT_SIZE)
    ap.add_argument("--border-tol", type=int, default=DEFAULT_BORDER_TOL)
    ap.add_argument("--circle-crop", action="store_true")
    ap.add_argument("--clahe", action="store_true",
                    help="enable CLAHE (ablation); needs opencv-python")
    ap.add_argument("--force", action="store_true",
                    help="reprocess even if the output PNG already exists")
    args = ap.parse_args()

    params = {
        "size": args.size,
        "border_tol": args.border_tol,
        "circle_crop": args.circle_crop,
        "clahe": args.clahe,
    }
    print(f"Preprocessing params: {params}")

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "datasets": {},
    }
    for key in args.datasets:
        print(f"\n=== {key} ===")
        result["datasets"][key] = process_dataset(key, params, args.force)

    cfg_path = LOG_DIR / "phase3_preprocess_config.json"
    cfg_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {cfg_path.relative_to(PROJECT_DIR)}")

    total_fail = sum(d["n_failures"] for d in result["datasets"].values())
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
