"""
Phase 3 - visual sanity check.

For each dataset, draw a stratified random sample (up to 4 images per DR grade)
of the *preprocessed* images with their label overlaid, and save a contact
sheet. Look at these before trusting the pipeline: a shifted / misaligned label
column is the most common silent bug in medical-imaging pipelines, and it is
obvious to the eye (grade-4 panels showing a clean retina, grade-0 panels full
of haemorrhages).

Run:
  .venv/Scripts/python.exe scripts/phase3_sanity_check.py [--per-grade 4] [--seed 42]

Outputs:
  outputs/phase3_sanity_{aptos,messidor2,idrid}.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_DIR / "data_splits"
OUT_DIR = PROJECT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

GRADE_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Prolif."}
MANIFESTS = {
    "aptos": "aptos_manifest.csv",
    "messidor2": "messidor2_manifest.csv",
    "idrid": "idrid_manifest.csv",
}


def make_sheet(name: str, manifest: Path, per_grade: int, seed: int) -> Path | None:
    df = pd.read_csv(manifest)
    df = df[df["processed_path"].notna() & (df["processed_path"].astype(str) != "")]
    df["diagnosis"] = df["diagnosis"].astype(int)

    picks = []
    for g in sorted(df["diagnosis"].unique()):
        sub = df[df["diagnosis"] == g]
        picks.append(sub.sample(n=min(per_grade, len(sub)), random_state=seed))
    sample = pd.concat(picks).reset_index(drop=True)

    ncols = per_grade
    nrows = len(sample) // ncols + int(len(sample) % ncols > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax in axes:
        ax.axis("off")
    for i, r in sample.iterrows():
        ax = axes[i]
        p = Path(r["processed_path"])
        if p.is_file():
            with Image.open(p) as im:
                ax.imshow(im.convert("RGB"))
        g = int(r["diagnosis"])
        ax.set_title(f"{r['id_code']}\ngrade {g} - {GRADE_NAMES.get(g, '?')}",
                     fontsize=8)

    dist = df["diagnosis"].value_counts().sort_index().to_dict()
    fig.suptitle(
        f"{name} - preprocessed sanity check "
        f"(n={len(df)}, grades " + ", ".join(f"{k}:{v}" for k, v in dist.items()) + ")",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / f"phase3_sanity_{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-grade", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    any_done = False
    for name, fname in MANIFESTS.items():
        manifest = SPLIT_DIR / fname
        if not manifest.is_file():
            print(f"skip {name}: {manifest} not found (run phase3_preprocess.py)")
            continue
        out = make_sheet(name, manifest, args.per_grade, args.seed)
        print(f"wrote {out.relative_to(PROJECT_DIR)}")
        any_done = True

    if not any_done:
        print("nothing to do - no manifests found")
        return 1
    print("\nEyeball each sheet: labels must match the visible pathology. "
          "Grade 0 = clean; grade 3-4 = obvious haemorrhages / neovascularisation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
