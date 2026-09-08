"""
Phase 2 - Data acquisition audit.

Verifies that the three raw datasets are present and internally consistent,
then reports the DR-grade class distribution for each. Nothing here modifies
the datasets; it only reads and summarises.

Datasets (see DATA_PROVENANCE.md for sources / licenses / citations):
  APTOS 2019    - internal only (train / val / test are carved from this)
  Messidor-2    - EXTERNAL TEST ONLY
  IDRiD         - EXTERNAL TEST ONLY

Outputs (written to outputs/ and logs/):
  outputs/phase2_class_distribution.csv   long-format grade counts per dataset
  outputs/phase2_audit.json               machine-readable full audit result
  outputs/phase2_class_distribution.png   grouped bar chart of grade share
  logs/phase2_audit_report.md             human-readable report

Run:
  .venv/Scripts/python.exe scripts/phase2_data_audit.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_DIR.parent

APTOS_DIR = PAPER_DIR / "aptos2019-blindness-detection"
APTOS_TRAIN_CSV = APTOS_DIR / "train.csv"
APTOS_TRAIN_IMG = APTOS_DIR / "train_images"
APTOS_TEST_CSV = APTOS_DIR / "test.csv"
APTOS_TEST_IMG = APTOS_DIR / "test_images"

MESSIDOR_CSV = PAPER_DIR / "archive" / "messidor_data.csv"
MESSIDOR_IMG = PAPER_DIR / "archive" / "messidor-2" / "messidor-2" / "preprocess"

IDRID_CSV = PAPER_DIR / "archive (1)" / "idrid_labels.csv"
IDRID_IMG = PAPER_DIR / "archive (1)" / "Imagenes" / "Imagenes"

OUT_DIR = PROJECT_DIR / "outputs"
LOG_DIR = PROJECT_DIR / "logs"
OUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

GRADES = [0, 1, 2, 3, 4]
GRADE_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def list_images(folder: Path) -> set[str]:
    """Return the set of image *stems* (filename without extension) in a folder."""
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS}


def list_image_names(folder: Path) -> set[str]:
    """Return the set of full image filenames in a folder."""
    if not folder.is_dir():
        return set()
    return {p.name for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS}


def grade_counts(series: pd.Series) -> dict:
    c = Counter(int(v) for v in series.dropna())
    total = sum(c.values())
    return {
        "n_labeled": total,
        "counts": {int(g): int(c.get(g, 0)) for g in GRADES},
        "pct": {
            int(g): round(100.0 * c.get(g, 0) / total, 2) if total else 0.0
            for g in GRADES
        },
        "n_unexpected_labels": int(sum(v for k, v in c.items() if k not in GRADES)),
        "majority_grade": int(max(c, key=c.get)) if total else None,
    }


def imbalance_ratio(counts: dict) -> float | None:
    vals = [counts[g] for g in GRADES]
    if min(vals) == 0:
        return None
    return round(max(vals) / min(vals), 2)


# --------------------------------------------------------------------------- #
# APTOS 2019                                                                   #
# --------------------------------------------------------------------------- #
def audit_aptos() -> dict:
    res: dict = {"name": "APTOS 2019", "role": "internal (train/val/test)"}

    train_df = pd.read_csv(APTOS_TRAIN_CSV)
    train_df.columns = [c.strip() for c in train_df.columns]
    id_col, dx_col = "id_code", "diagnosis"

    label_ids = set(train_df[id_col].astype(str))
    img_stems = list_images(APTOS_TRAIN_IMG)

    res["train"] = {
        "label_csv": str(APTOS_TRAIN_CSV),
        "image_dir": str(APTOS_TRAIN_IMG),
        "n_label_rows": int(len(train_df)),
        "n_unique_label_ids": int(len(label_ids)),
        "n_image_files": int(len(img_stems)),
        "labels_without_image": sorted(label_ids - img_stems)[:20],
        "images_without_label": sorted(img_stems - label_ids)[:20],
        "n_labels_without_image": int(len(label_ids - img_stems)),
        "n_images_without_label": int(len(img_stems - label_ids)),
        "n_duplicate_label_ids": int(len(train_df) - len(label_ids)),
        "class_distribution": grade_counts(train_df[dx_col]),
    }
    res["train"]["class_distribution"]["imbalance_ratio"] = imbalance_ratio(
        res["train"]["class_distribution"]["counts"]
    )

    test_df = pd.read_csv(APTOS_TEST_CSV)
    test_df.columns = [c.strip() for c in test_df.columns]
    res["kaggle_holdout_test"] = {
        "label_csv": str(APTOS_TEST_CSV),
        "image_dir": str(APTOS_TEST_IMG),
        "n_rows": int(len(test_df)),
        "n_image_files": int(len(list_images(APTOS_TEST_IMG))),
        "has_labels": "diagnosis" in test_df.columns,
        "note": (
            "Kaggle competition holdout - NO public labels. Not usable as an "
            "internal test set. Internal test is carved from train.csv in Phase 3."
        ),
    }
    return res


# --------------------------------------------------------------------------- #
# Messidor-2                                                                   #
# --------------------------------------------------------------------------- #
def audit_messidor() -> dict:
    res: dict = {"name": "Messidor-2", "role": "EXTERNAL TEST ONLY"}

    df = pd.read_csv(MESSIDOR_CSV)
    df.columns = [c.strip() for c in df.columns]
    id_col = "id_code"
    dx_col = "diagnosis"
    grad_col = "adjudicated_gradable" if "adjudicated_gradable" in df.columns else None

    label_names = set(df[id_col].astype(str))
    img_names = list_image_names(MESSIDOR_IMG)

    res["n_label_rows"] = int(len(df))
    res["n_image_files"] = int(len(img_names))
    res["n_labels_without_image"] = int(len(label_names - img_names))
    res["n_images_without_label"] = int(len(img_names - label_names))
    res["labels_without_image"] = sorted(label_names - img_names)[:20]
    res["images_without_label"] = sorted(img_names - label_names)[:20]
    res["n_duplicate_label_ids"] = int(len(df) - len(label_names))

    res["class_distribution_all"] = grade_counts(df[dx_col])
    res["class_distribution_all"]["imbalance_ratio"] = imbalance_ratio(
        res["class_distribution_all"]["counts"]
    )

    if grad_col:
        gradable = df[df[grad_col] == 1]
        ungradable = df[df[grad_col] != 1]
        res["n_ungradable"] = int(len(ungradable))
        res["class_distribution_gradable_only"] = grade_counts(gradable[dx_col])
        res["class_distribution_gradable_only"]["imbalance_ratio"] = imbalance_ratio(
            res["class_distribution_gradable_only"]["counts"]
        )
        res["note_gradable"] = (
            "Recommend restricting the external test to adjudicated_gradable == 1."
        )
    return res


# --------------------------------------------------------------------------- #
# IDRiD                                                                        #
# --------------------------------------------------------------------------- #
def audit_idrid() -> dict:
    res: dict = {"name": "IDRiD (Disease Grading)", "role": "EXTERNAL TEST ONLY"}

    # Header is malformed (trailing commas, a stray label column, and the
    # test-split rows carry only 3 fields). Read positionally: col0 = id,
    # col1 = DR grade, col2 = risk of macular edema.
    raw = pd.read_csv(
        IDRID_CSV, header=0, usecols=[0, 1, 2],
        names=["id_code", "diagnosis", "dme_risk"],
    )
    raw = raw.dropna(subset=["id_code"])
    raw["id_code"] = raw["id_code"].astype(str).str.strip()
    raw = raw[raw["id_code"].str.startswith("IDRiD_")]
    raw["diagnosis"] = pd.to_numeric(raw["diagnosis"], errors="coerce")

    raw["orig_split"] = raw["id_code"].apply(
        lambda s: "test" if s.endswith("test") else "train"
    )

    label_ids = set(raw["id_code"])
    img_stems = list_images(IDRID_IMG)

    res["n_label_rows"] = int(len(raw))
    res["n_unique_label_ids"] = int(len(label_ids))
    res["n_image_files"] = int(len(img_stems))
    res["n_labels_without_image"] = int(len(label_ids - img_stems))
    res["n_images_without_label"] = int(len(img_stems - label_ids))
    res["labels_without_image"] = sorted(label_ids - img_stems)[:20]
    res["images_without_label"] = sorted(img_stems - label_ids)[:20]
    res["orig_split_breakdown"] = {
        k: int(v) for k, v in raw["orig_split"].value_counts().items()
    }

    res["class_distribution"] = grade_counts(raw["diagnosis"])
    res["class_distribution"]["imbalance_ratio"] = imbalance_ratio(
        res["class_distribution"]["counts"]
    )
    res["note"] = (
        "Official IDRiD Disease Grading subset is 516 images (413 train + 103 "
        f"test). This mirror provides {len(img_stems)} images / {len(raw)} "
        "labels - confirm completeness against the source before the Data "
        "Availability statement."
    )
    return res


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def build_long_table(audit: dict) -> pd.DataFrame:
    rows = []

    def add(dataset, subset, dist):
        for g in GRADES:
            rows.append(
                {
                    "dataset": dataset,
                    "subset": subset,
                    "grade": g,
                    "grade_name": GRADE_NAMES[g],
                    "count": dist["counts"][g],
                    "pct": dist["pct"][g],
                }
            )

    add("APTOS 2019", "train.csv (all labeled)", audit["aptos"]["train"]["class_distribution"])
    add("Messidor-2", "all", audit["messidor"]["class_distribution_all"])
    if "class_distribution_gradable_only" in audit["messidor"]:
        add("Messidor-2", "gradable only", audit["messidor"]["class_distribution_gradable_only"])
    add("IDRiD", "all", audit["idrid"]["class_distribution"])
    return pd.DataFrame(rows)


def plot_distribution(df_long: pd.DataFrame, path: Path) -> None:
    pivot_keys = [
        ("APTOS 2019", "train.csv (all labeled)"),
        ("Messidor-2", "gradable only"),
        ("IDRiD", "all"),
    ]
    have = {(d, s) for d, s in zip(df_long["dataset"], df_long["subset"])}
    pivot_keys = [k for k in pivot_keys if k in have]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(len(pivot_keys), 1)
    for i, (dset, subset) in enumerate(pivot_keys):
        sub = df_long[(df_long["dataset"] == dset) & (df_long["subset"] == subset)]
        sub = sub.sort_values("grade")
        xs = [g + i * width for g in GRADES]
        ax.bar(xs, sub["pct"].values, width=width, label=f"{dset} ({subset})")
    ax.set_xticks([g + width * (len(pivot_keys) - 1) / 2 for g in GRADES])
    ax.set_xticklabels([f"{g}\n{GRADE_NAMES[g]}" for g in GRADES])
    ax.set_ylabel("Share of images (%)")
    ax.set_title("DR grade distribution by dataset (Phase 2 audit)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown(audit: dict, df_long: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    w = lines.append
    w("# Phase 2 - Data Acquisition Audit")
    w("")
    w(f"_Generated: {audit['generated_utc']}_")
    w("")
    w("## 1. File-level consistency")
    w("")
    w("| Dataset | Label rows | Image files | Labels w/o image | Images w/o label | Dup label IDs |")
    w("|---|---|---|---|---|---|")
    a = audit["aptos"]["train"]
    w(f"| APTOS 2019 (train.csv) | {a['n_label_rows']} | {a['n_image_files']} | "
      f"{a['n_labels_without_image']} | {a['n_images_without_label']} | {a['n_duplicate_label_ids']} |")
    m = audit["messidor"]
    w(f"| Messidor-2 | {m['n_label_rows']} | {m['n_image_files']} | "
      f"{m['n_labels_without_image']} | {m['n_images_without_label']} | {m['n_duplicate_label_ids']} |")
    i = audit["idrid"]
    w(f"| IDRiD (Disease Grading) | {i['n_label_rows']} | {i['n_image_files']} | "
      f"{i['n_labels_without_image']} | {i['n_images_without_label']} | "
      f"{i['n_label_rows'] - i['n_unique_label_ids']} |")
    w("")
    kh = audit["aptos"]["kaggle_holdout_test"]
    w(f"> APTOS Kaggle holdout `test_images/`: {kh['n_image_files']} images, "
      f"has_labels={kh['has_labels']}. {kh['note']}")
    w("")

    w("## 2. DR-grade class distribution")
    w("")
    w("| Dataset | Subset | 0 No DR | 1 Mild | 2 Moderate | 3 Severe | 4 Prolif. | n | Majority | Max/Min ratio |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for (dset, subset), grp in df_long.groupby(["dataset", "subset"], sort=False):
        grp = grp.sort_values("grade")
        cells = [f"{r['count']} ({r['pct']}%)" for _, r in grp.iterrows()]
        n = int(grp["count"].sum())
        maj = int(grp.loc[grp["count"].idxmax(), "grade"])
        counts = {int(r["grade"]): int(r["count"]) for _, r in grp.iterrows()}
        ratio = imbalance_ratio(counts)
        w(f"| {dset} | {subset} | " + " | ".join(cells) + f" | {n} | {maj} ({GRADE_NAMES[maj]}) | "
          f"{ratio if ratio is not None else 'inf (empty class)'} |")
    w("")

    w("## 3. Notes carried into later phases")
    w("")
    w("- **Metric choice (Phase 4+):** all three datasets are class-imbalanced "
      "(APTOS/Messidor-2 dominated by grade 0; IDRiD's modal class is grade 2), "
      "and imbalanced differently. Use quadratic-weighted kappa and per-class "
      "sensitivity/specificity as primary metrics; raw accuracy is misleading here.")
    if "note_gradable" in m:
        w(f"- **Messidor-2:** {m['note_gradable']} "
          f"({m.get('n_ungradable', 0)} ungradable rows present.)")
    w(f"- **IDRiD:** {i['note']}")
    w("- **Split isolation:** APTOS 2019 -> train/val/test only. "
      "Messidor-2 and IDRiD -> external test only, never seen in training or "
      "hyperparameter tuning.")
    w("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    missing = [
        str(p)
        for p in [
            APTOS_TRAIN_CSV, APTOS_TRAIN_IMG, APTOS_TEST_CSV,
            MESSIDOR_CSV, MESSIDOR_IMG, IDRID_CSV, IDRID_IMG,
        ]
        if not p.exists()
    ]
    if missing:
        print("ERROR - expected paths not found:")
        for m in missing:
            print("  ", m)
        return 1

    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aptos": audit_aptos(),
        "messidor": audit_messidor(),
        "idrid": audit_idrid(),
    }

    df_long = build_long_table(audit)
    df_long.to_csv(OUT_DIR / "phase2_class_distribution.csv", index=False)
    (OUT_DIR / "phase2_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    plot_distribution(df_long, OUT_DIR / "phase2_class_distribution.png")
    write_markdown(audit, df_long, LOG_DIR / "phase2_audit_report.md")

    print("=" * 70)
    print("PHASE 2 DATA AUDIT - SUMMARY")
    print("=" * 70)
    for key in ("aptos", "messidor", "idrid"):
        d = audit[key]
        name = d.get("name", key)
        print(f"\n{name}  [{d.get('role', '')}]")
        if key == "aptos":
            t = d["train"]
            print(f"  labels={t['n_label_rows']}  images={t['n_image_files']}  "
                  f"mismatch(l/i)={t['n_labels_without_image']}/{t['n_images_without_label']}")
            cd = t["class_distribution"]
        elif key == "messidor":
            print(f"  labels={d['n_label_rows']}  images={d['n_image_files']}  "
                  f"mismatch(l/i)={d['n_labels_without_image']}/{d['n_images_without_label']}")
            cd = d.get("class_distribution_gradable_only", d["class_distribution_all"])
        else:
            print(f"  labels={d['n_label_rows']}  images={d['n_image_files']}  "
                  f"mismatch(l/i)={d['n_labels_without_image']}/{d['n_images_without_label']}")
            cd = d["class_distribution"]
        print("  grades: " + "  ".join(
            f"{g}:{cd['counts'][g]}({cd['pct'][g]}%)" for g in GRADES
        ))
        print(f"  majority grade={cd['majority_grade']}  max/min ratio={cd['imbalance_ratio']}")
    print("\nWrote:")
    for p in [
        OUT_DIR / "phase2_class_distribution.csv",
        OUT_DIR / "phase2_audit.json",
        OUT_DIR / "phase2_class_distribution.png",
        LOG_DIR / "phase2_audit_report.md",
    ]:
        print("  ", p.relative_to(PROJECT_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
