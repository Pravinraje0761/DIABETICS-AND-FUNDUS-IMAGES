"""
Phase 28a - official IDRiD Disease-Grading release: manifest, preprocessing,
and an exact audit of the third-party mirror used so far.

Input: the official IEEE DataPort folder "B. Disease Grading" (516 images:
413 train + 103 test; labels in "2. Groundtruths/*Training Labels.csv" and
"*Testing Labels.csv"). Test-set file names restart at IDRiD_001, so test IDs
are suffixed "test" (IDRiD_001test), the same convention as the mirror.

Outputs:
  data_splits/idrid_official_manifest.csv      516 rows, split=external
  data_processed/idrid_official/<id>.png       224 px, same Phase-3 pipeline
  outputs/phase28/mirror_vs_official_labels.csv one row per official image
  outputs/phase28/mirror_audit.md              exact discrepancy counts

Run: python scripts/phase28_idrid_official_prep.py --root "B. Disease Grading"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import preprocess_file  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT = PROJECT_DIR / "outputs" / "phase28"
PROC = PROJECT_DIR / "data_processed" / "idrid_official"


def find_labels(root: Path):
    csvs = [p for p in root.rglob("*.csv")]
    tr = [p for p in csvs if "train" in p.name.lower()]
    te = [p for p in csvs if "test" in p.name.lower()]
    if not tr or not te:
        raise SystemExit(f"label CSVs not found under {root}: {[p.name for p in csvs]}")
    return tr[0], te[0]


def read_labels(p: Path, split: str) -> pd.DataFrame:
    d = pd.read_csv(p)
    d.columns = [c.strip() for c in d.columns]
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")].dropna(subset=["Image name"])
    d["image"] = d["Image name"].astype(str).str.strip()
    d["id_code"] = d["image"] + ("test" if split == "test" else "")
    d["diagnosis"] = d["Retinopathy grade"].astype(int)
    d["dme_risk"] = d["Risk of macular edema"].astype(int)
    d["orig_split"] = f"idrid_{split}"
    return d[["id_code", "image", "diagnosis", "dme_risk", "orig_split"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help='the "B. Disease Grading" folder')
    a = ap.parse_args()
    root = Path(a.root)
    if not root.is_absolute():
        root = PROJECT_DIR / root
    OUT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    trp, tep = find_labels(root)
    lab = pd.concat([read_labels(trp, "train"), read_labels(tep, "test")], ignore_index=True)

    imgs = {}
    for p in root.rglob("*.jpg"):
        split = "test" if "test" in str(p.parent).lower() else "train"
        imgs[p.stem + ("test" if split == "test" else "")] = p
    lab["source_path"] = lab["id_code"].map(lambda i: str(imgs.get(i, "")))
    missing = lab[lab.source_path == ""]
    if len(missing):
        print(f"WARNING: {len(missing)} labelled images have no file: {missing.id_code.tolist()[:10]}")
    lab = lab[lab.source_path != ""].copy()

    rows = []
    for _, r in lab.iterrows():
        dst = PROC / f"{r.id_code}.png"
        if not dst.exists():
            preprocess_file(r.source_path, dst)
        rows.append(str(dst.relative_to(PROJECT_DIR)).replace("\\", "/"))
    lab["processed_path"] = rows
    lab["split"] = "external"
    lab["source_path"] = lab["source_path"].map(
        lambda s: str(Path(s).relative_to(PROJECT_DIR)).replace("\\", "/")
        if Path(s).is_relative_to(PROJECT_DIR) else s)
    man = lab[["id_code", "diagnosis", "split", "dme_risk", "orig_split", "source_path", "processed_path"]]
    man.to_csv(PROJECT_DIR / "data_splits" / "idrid_official_manifest.csv", index=False)

    # ---- exact mirror audit
    mir = pd.read_csv(PROJECT_DIR / "data_splits" / "idrid_manifest.csv")[["id_code", "diagnosis"]]
    j = man[["id_code", "diagnosis", "orig_split"]].merge(
        mir.rename(columns={"diagnosis": "mirror_diagnosis"}), on="id_code", how="left")
    j["in_mirror"] = j.mirror_diagnosis.notna()
    j["label_differs"] = j.in_mirror & (j.mirror_diagnosis != j.diagnosis)
    j.to_csv(OUT / "mirror_vs_official_labels.csv", index=False)
    extra = sorted(set(mir.id_code) - set(man.id_code))
    off_counts = man.diagnosis.value_counts().sort_index().tolist()
    d = j[j.label_differs]
    L = ["# Phase 28a - IDRiD mirror vs official Disease-Grading release\n",
         f"- Official images with labels: {len(man)} (train {int((man.orig_split=='idrid_train').sum())}, "
         f"test {int((man.orig_split=='idrid_test').sum())}); grade counts 0-4: {off_counts}; "
         f"referable {100*(man.diagnosis>=2).mean():.1f}%",
         f"- Present in the mirror: {int(j.in_mirror.sum())}; missing from mirror: {int((~j.in_mirror).sum())}",
         f"- Mirror IDs not in the official release: {len(extra)} {extra[:10]}",
         f"- **Mirror labels that differ from the official grade: {len(d)}** "
         f"({100*len(d)/max(1,j.in_mirror.sum()):.1f}% of mirror images)", "",
         "| id | official | mirror |", "|---|--:|--:|"]
    L += [f"| {r.id_code} | {int(r.diagnosis)} | {int(r.mirror_diagnosis)} |" for _, r in d.iterrows()]
    (OUT / "mirror_audit.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
