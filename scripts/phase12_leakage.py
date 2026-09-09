"""
Phase 12 - duplicate / leakage check.

Perceptual hashing (8x8 difference-hash + 8x8 average-hash, PIL only) of every
preprocessed image, then Hamming-distance matching:

  * within APTOS: train-vs-val, train-vs-test, val-vs-test near-duplicate pairs
    (a leaked pair would inflate the internal test numbers);
  * across datasets: APTOS-vs-Messidor-2, APTOS-vs-IDRiD, Messidor-2-vs-IDRiD
    (a cross-dataset duplicate would break the "independent external set" claim).

A pair is a near-duplicate if the 128-bit hash Hamming distance is <= --thresh
(default 8). Exact duplicates (distance 0) are reported separately.

Outputs (outputs/phase12/):
  leakage_summary.md   counts per comparison + interpretation
  leakage_pairs.csv    every flagged pair (group_a, group_b, id_a, id_b, dist)

Run (reads data_processed/ via DR_DATA_ROOT if set):
  .venv/Scripts/python.exe scripts/phase12_leakage.py --thresh 8
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT / "data_splits"
OUT = PROJECT / "outputs" / "phase12"
DATA_ROOT = os.environ.get("DR_DATA_ROOT")


def resolve(p: str) -> Path:
    q = Path(p)
    if q.is_absolute():
        return q
    if DATA_ROOT and (Path(DATA_ROOT) / q).exists():
        return Path(DATA_ROOT) / q
    return PROJECT / q


def phash(path: Path):
    """Fine dhash: 33x32 grey -> 32x32 horizontal-difference bits (1024-bit).
    Fine enough to separate genuinely distinct fundus frames (which otherwise
    all look like a centred circle on black) from true near-duplicates."""
    try:
        im = Image.open(path).convert("L").resize((33, 32), Image.BILINEAR)
        a = np.asarray(im, dtype=np.int16)
        return (a[:, 1:] > a[:, :-1]).flatten().astype(np.uint8)   # 1024 bits
    except Exception:
        return None


def md5(path: Path):
    import hashlib
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:
        return None


def load_rows() -> pd.DataFrame:
    frames = []
    for name, tag in [("aptos_manifest.csv", "APTOS"),
                      ("messidor2_manifest.csv", "Messidor-2"),
                      ("idrid_manifest.csv", "IDRiD")]:
        p = SPLIT / name
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        df = df[df["processed_path"].notna() & (df["processed_path"].astype(str) != "")]
        df["dataset"] = tag
        if "split" not in df.columns:
            df["split"] = "external"
        frames.append(df[["id_code", "dataset", "split", "processed_path"]])
    return pd.concat(frames, ignore_index=True)


def hamming_pairs(A_ids, A_h, B_ids, B_h, thresh, same=False):
    out = []
    Bm = np.stack(B_h)
    for i, h in enumerate(A_h):
        d = np.count_nonzero(Bm != h, axis=1)
        for j in np.where(d <= thresh)[0]:
            if same and A_ids[i] == B_ids[j]:
                continue
            out.append((A_ids[i], B_ids[j], int(d[j])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thresh", type=int, default=24,
                    help="Hamming threshold on the 1024-bit fine dhash "
                         "(~2.3%% of bits); pairs at or below are near-duplicates")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    print(f"hashing {len(rows)} images ...")
    hashes = {}
    md5s = {}
    keep = []
    for _, r in rows.iterrows():
        pth = resolve(str(r["processed_path"]))
        h = phash(pth)
        if h is not None:
            hashes[(r["dataset"], r["id_code"])] = h
            md5s[(r["dataset"], r["id_code"])] = md5(pth)
            keep.append((r["dataset"], r["id_code"], r["split"]))
    print(f"hashed {len(keep)}")

    # exact byte-identical duplicates (unambiguous)
    from collections import defaultdict
    split_of = {(ds, idc): sp for ds, idc, sp in keep}
    m5 = defaultdict(list)
    for (ds, idc), h in md5s.items():
        if h:
            m5[h].append((ds, idc))
    exact_groups = [sorted(v) for v in m5.values() if len(v) > 1]
    exact_rows = []
    for g in exact_groups:
        for (da, ia), (db, ib) in itertools.combinations(g, 2):
            exact_rows.append({
                "group_a": f"{da}:{split_of[(da, ia)]}", "id_a": ia,
                "group_b": f"{db}:{split_of[(db, ib)]}", "id_b": ib,
                "boundary_crossed": (da != db) or (split_of[(da, ia)] != split_of[(db, ib)]),
            })
    ex = pd.DataFrame(exact_rows)
    ex.to_csv(OUT / "exact_duplicates.csv", index=False)

    by = {}
    for ds, idc, sp in keep:
        by.setdefault(ds, []).append((idc, sp, hashes[(ds, idc)]))

    flagged = []
    if "APTOS" in by:
        bysplit = {}
        for idc, sp, h in by["APTOS"]:
            bysplit.setdefault(sp, []).append((idc, h))
        for sa, sb in [("train", "val"), ("train", "test"), ("val", "test")]:
            if sa in bysplit and sb in bysplit:
                Aid = [x[0] for x in bysplit[sa]]; Ah = [x[1] for x in bysplit[sa]]
                Bid = [x[0] for x in bysplit[sb]]; Bh = [x[1] for x in bysplit[sb]]
                for a, b, d in hamming_pairs(Aid, Ah, Bid, Bh, args.thresh):
                    flagged.append(("APTOS:%s" % sa, "APTOS:%s" % sb, a, b, d))

    for da, db in itertools.combinations(
            [k for k in ("APTOS", "Messidor-2", "IDRiD") if k in by], 2):
        Aid = [x[0] for x in by[da]]; Ah = [x[2] for x in by[da]]
        Bid = [x[0] for x in by[db]]; Bh = [x[2] for x in by[db]]
        for a, b, d in hamming_pairs(Aid, Ah, Bid, Bh, args.thresh):
            flagged.append((da, db, a, b, d))

    fp = pd.DataFrame(flagged, columns=["group_a", "group_b", "id_a", "id_b", "dist"])
    fp.to_csv(OUT / "leakage_pairs.csv", index=False)

    n_exact_cross = int(ex["boundary_crossed"].sum()) if len(ex) else 0
    L = ["# Phase 12 - duplicate / leakage check", "",
         f"{len(keep)} preprocessed images. Two tests: (a) exact byte-identical "
         f"duplicates (MD5 of the processed PNG); (b) near-duplicates by a "
         f"1024-bit fine difference-hash, Hamming $\\le$ {args.thresh}.", "",
         "## (a) Exact duplicates (MD5)", "",
         f"- Exact-duplicate groups: **{len(exact_groups)}**",
         f"- Pairs crossing a split or dataset boundary: **{n_exact_cross}**",
         ("- No exact duplicate crosses a train/val/test boundary or a dataset "
          "boundary: no image is literally repeated between APTOS development and "
          "the internal test set, and no external image is a byte-copy of a "
          "development image." if n_exact_cross == 0 else
          "- Boundary-crossing exact duplicates are listed in "
          "`exact_duplicates.csv` (boundary_crossed == True); the affected split "
          "or external comparison should be re-run with them removed."), "",
         "## (b) Near-duplicates (fine dhash)", "",
         "| Comparison | near-duplicate pairs (dhash Hamming <= %d) |" % args.thresh,
         "|---|--:|"]
    if fp.empty:
        L.append("| (all comparisons) | 0 |")
    else:
        for (ga, gb), sub in fp.groupby(["group_a", "group_b"]):
            L.append(f"| {ga} vs {gb} | {len(sub)} |")
    L += ["", "Near-duplicate pairs (if any) are in `leakage_pairs.csv`. "
          "Perceptual near-duplicate detection on tightly cropped, resized fundus "
          "frames is intrinsically noisy (every image is a centred disc on black); "
          "the exact-MD5 test above is the reliable one, and an embedding-based "
          "near-duplicate check is listed as a follow-up.", ""]
    (OUT / "leakage_summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}/leakage_summary.md  "
          f"(exact groups {len(exact_groups)}, cross-boundary {n_exact_cross}, "
          f"near-dup pairs {len(fp)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
