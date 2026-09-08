"""
Phase 3 - stratified APTOS 2019 split: 70% train / 10% internal-val / 20%
internal-test, stratified by DR grade, fixed seed, written to CSV so it is
exactly reproducible.

APTOS is the ONLY dataset that is ever split like this. Messidor-2 and IDRiD
are external test sets and are never partitioned.

Run:
  .venv/Scripts/python.exe scripts/phase3_split_aptos.py [--seed 42]

Outputs (data_splits/):
  aptos_train.csv        id_code, diagnosis
  aptos_val.csv          id_code, diagnosis
  aptos_test.csv         id_code, diagnosis
  aptos_split_all.csv    id_code, diagnosis, split      (the three concatenated)
logs/:
  phase3_split_config.json   seed, ratios, per-grade counts, timestamp
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_DIR.parent
APTOS_TRAIN_CSV = PAPER_DIR / "aptos2019-blindness-detection" / "train.csv"

SPLIT_DIR = PROJECT_DIR / "data_splits"
LOG_DIR = PROJECT_DIR / "logs"
SPLIT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
TEST_FRAC = 0.20
GRADES = [0, 1, 2, 3, 4]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

    df = pd.read_csv(APTOS_TRAIN_CSV)
    df.columns = [c.strip() for c in df.columns]
    df = df[["id_code", "diagnosis"]].copy()
    df["id_code"] = df["id_code"].astype(str)
    df["diagnosis"] = df["diagnosis"].astype(int)

    n_before = len(df)
    df = df.drop_duplicates(subset="id_code").reset_index(drop=True)
    if len(df) != n_before:
        print(f"WARNING: dropped {n_before - len(df)} duplicate id_code rows")

    # First carve off the test set (20%), then split the remaining 80% into
    # train / val so that val is 10% of the whole (0.125 * 0.8 = 0.10).
    trainval_df, test_df = train_test_split(
        df,
        test_size=TEST_FRAC,
        stratify=df["diagnosis"],
        random_state=args.seed,
        shuffle=True,
    )
    val_ratio_within_trainval = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_ratio_within_trainval,
        stratify=trainval_df["diagnosis"],
        random_state=args.seed,
        shuffle=True,
    )

    train_df = train_df.sort_values("id_code").reset_index(drop=True)
    val_df = val_df.sort_values("id_code").reset_index(drop=True)
    test_df = test_df.sort_values("id_code").reset_index(drop=True)

    # Integrity: partitions must be disjoint and cover everything exactly once.
    ids_train, ids_val, ids_test = (
        set(train_df["id_code"]),
        set(val_df["id_code"]),
        set(test_df["id_code"]),
    )
    assert ids_train.isdisjoint(ids_val)
    assert ids_train.isdisjoint(ids_test)
    assert ids_val.isdisjoint(ids_test)
    assert ids_train | ids_val | ids_test == set(df["id_code"])

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(SPLIT_DIR / f"aptos_{name}.csv", index=False)

    all_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    all_df.to_csv(SPLIT_DIR / "aptos_split_all.csv", index=False)

    def dist(part: pd.DataFrame) -> dict:
        vc = part["diagnosis"].value_counts().to_dict()
        return {int(g): int(vc.get(g, 0)) for g in GRADES}

    config = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_csv": str(APTOS_TRAIN_CSV),
        "seed": args.seed,
        "ratios": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "stratified_by": "diagnosis",
        "sklearn_train_test_split": "two-stage (test first, then val from trainval)",
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "class_distribution": {
            "total": dist(df),
            "train": dist(train_df),
            "val": dist(val_df),
            "test": dist(test_df),
        },
    }
    (LOG_DIR / "phase3_split_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("APTOS 2019 stratified split")
    print(f"  seed={args.seed}  total={len(df)}")
    print(f"  train={len(train_df)} ({len(train_df)/len(df):.1%})  "
          f"val={len(val_df)} ({len(val_df)/len(df):.1%})  "
          f"test={len(test_df)} ({len(test_df)/len(df):.1%})")
    print("  per-grade (grade: total -> train/val/test):")
    for g in GRADES:
        t = config["class_distribution"]
        print(f"    {g}: {t['total'][g]:4d} -> {t['train'][g]:4d}/{t['val'][g]:3d}/{t['test'][g]:4d}")
    print(f"\nWrote split CSVs to {SPLIT_DIR.relative_to(PROJECT_DIR)}/ and "
          f"config to {(LOG_DIR / 'phase3_split_config.json').relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
