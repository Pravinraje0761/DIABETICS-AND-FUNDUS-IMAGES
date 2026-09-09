"""
Phase 14 -- embedding-based near-duplicate check across the three datasets.

Upgrades the Phase 12 leakage check (MD5 byte-identity + a noisy perceptual
dhash) with a semantic check: an ImageNet-pretrained EfficientNet-B3 feature
extractor (1536-d penultimate activations, no fine-tuning on any DR data, so it
favours neither the internal nor the external sets), L2-normalised, cosine
similarity for every cross-dataset image pair.

Reports, per dataset pair (APTOS<->Messidor-2, APTOS<->IDRiD, Messidor-2<->IDRiD):
  * count of cross-dataset pairs with cosine >= 0.90 / 0.95 / 0.98
  * for the internal APTOS test split specifically, how many of its images have
    a >= 0.95 cosine match anywhere in APTOS train/val (context for the known
    within-APTOS duplication) and in either external set (should be ~0)
  * the 40 most-similar cross-dataset pairs, for manual inspection

Outputs (outputs/phase14/):
  embedding_neardup_summary.md
  cross_dataset_top_pairs.csv        (dataset_a,id_a,dataset_b,id_b,cosine)
  aptos_test_vs_external_matches.csv (any APTOS-test image with a cos>=0.9 hit
                                      in an external set -- expected empty)

Needs the processed images (data_processed/, via the committed manifests) and a
GPU is used if present. ~1-2 min on GPU for 5861 images.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import FundusDataset  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_DIR / "data_splits"
OUT_DIR = PROJECT_DIR / "outputs" / "phase14"

MANIFESTS = {
    "APTOS": (SPLIT_DIR / "aptos_manifest.csv", None),          # all splits
    "Messidor-2": (SPLIT_DIR / "messidor2_manifest.csv", "external"),
    "IDRiD": (SPLIT_DIR / "idrid_manifest.csv", "external"),
}
THRESHOLDS = (0.90, 0.95, 0.98)


def _extractor(device: torch.device) -> nn.Module:
    from torchvision import models

    m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    m.classifier = nn.Identity()  # -> 1536-d pooled features
    return m.to(device).eval()


@torch.no_grad()
def _embed(manifest: Path, split, model, device, batch_size, num_workers):
    ds = FundusDataset(manifest, split=split, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    feats, ids, splits = [], [], []
    df = ds.df.set_index("id_code")
    has_split = "split" in ds.df.columns
    for images, _labels, id_codes in loader:
        f = model(images.to(device, non_blocking=True))
        f = torch.nn.functional.normalize(f, dim=1)
        feats.append(f.cpu())
        ids.extend(id_codes)
        if has_split:
            splits.extend(str(df.loc[i, "split"]) for i in id_codes)
    return (torch.cat(feats).numpy().astype(np.float32),
            np.asarray(ids), np.asarray(splits))


def _cross_pairs(fa, ida, fb, idb, name_a, name_b, thr_counts, top_heap, k, chunk=512):
    """Accumulate threshold counts and keep the global top-k pairs (by cosine)."""
    import heapq

    for i0 in range(0, len(fa), chunk):
        block = fa[i0:i0 + chunk] @ fb.T                      # (c, Nb) cosine
        for t in THRESHOLDS:
            thr_counts[t] += int((block >= t).sum())
        flat_idx = np.argsort(block, axis=None)[-k:]
        for fi in flat_idx:
            r, c = divmod(int(fi), block.shape[1])
            cos = float(block[r, c])
            item = (cos, name_a, str(ida[i0 + r]), name_b, str(idb[c]))
            if len(top_heap) < k:
                heapq.heappush(top_heap, item)
            elif cos > top_heap[0][0]:
                heapq.heapreplace(top_heap, item)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = (torch.device("cuda") if (args.device != "cpu" and torch.cuda.is_available())
              else torch.device("cpu"))
    model = _extractor(device)

    emb: dict[str, tuple] = {}
    for name, (manifest, split) in MANIFESTS.items():
        if not manifest.is_file():
            print(f"skip {name}: {manifest} missing")
            continue
        f, ids, spl = _embed(manifest, split, model, device, args.batch_size, args.num_workers)
        emb[name] = (f, ids, spl)
        print(f"{name}: {len(ids)} embeddings")

    rows_md = ["# Phase 14 -- embedding near-duplicate check", "",
               "Feature extractor: ImageNet-pretrained EfficientNet-B3, 1536-d pooled, "
               "L2-normalised. Cosine similarity, all cross-dataset pairs.", "",
               "| Dataset pair | pairs cos>=0.90 | >=0.95 | >=0.98 |",
               "|---|--:|--:|--:|"]
    top_csv_rows: list[dict] = []
    for a, b in [("APTOS", "Messidor-2"), ("APTOS", "IDRiD"), ("Messidor-2", "IDRiD")]:
        if a not in emb or b not in emb:
            continue
        thr_counts = {t: 0 for t in THRESHOLDS}
        top_heap: list = []
        _cross_pairs(emb[a][0], emb[a][1], emb[b][0], emb[b][1], a, b,
                     thr_counts, top_heap, args.top_k)
        rows_md.append(f"| {a} <-> {b} | {thr_counts[0.90]} | {thr_counts[0.95]} "
                       f"| {thr_counts[0.98]} |")
        for cos, na, ia, nb, ib in sorted(top_heap, reverse=True):
            top_csv_rows.append({"dataset_a": na, "id_a": ia, "dataset_b": nb,
                                 "id_b": ib, "cosine": round(cos, 4)})

    pd.DataFrame(top_csv_rows).to_csv(OUT_DIR / "cross_dataset_top_pairs.csv", index=False)

    if "APTOS" in emb:
        fA, idA, splA = emb["APTOS"]
        is_test = splA == "test"
        is_trainval = np.isin(splA, ["train", "val"])
        ft, idt = fA[is_test], idA[is_test]
        rows_md += ["", "## APTOS internal-test images with a near-duplicate elsewhere",
                    "", "| target pool | test images cos>=0.95 | cos>=0.90 |",
                    "|---|--:|--:|"]
        sim_tv = ft @ fA[is_trainval].T
        rows_md.append(f"| APTOS train/val | {int((sim_tv.max(1) >= 0.95).sum())} / {len(idt)} "
                       f"| {int((sim_tv.max(1) >= 0.90).sum())} / {len(idt)} |")
        ext_hit_rows: list[dict] = []
        for ext in ("Messidor-2", "IDRiD"):
            if ext not in emb:
                continue
            sim_e = ft @ emb[ext][0].T
            mx, am = sim_e.max(1), sim_e.argmax(1)
            rows_md.append(f"| {ext} (external) | {int((mx >= 0.95).sum())} / {len(idt)} "
                           f"| {int((mx >= 0.90).sum())} / {len(idt)} |")
            for j in np.where(mx >= 0.90)[0]:
                ext_hit_rows.append({"aptos_test_id": str(idt[j]), "external_set": ext,
                                     "external_id": str(emb[ext][1][am[j]]),
                                     "cosine": round(float(mx[j]), 4)})
        pd.DataFrame(ext_hit_rows).to_csv(
            OUT_DIR / "aptos_test_vs_external_matches.csv", index=False)
        rows_md += ["",
                    f"Cross-dataset near-duplicates into an external set: "
                    f"**{len(ext_hit_rows)}** APTOS-test images with any cos>=0.90 match "
                    f"(detail in `aptos_test_vs_external_matches.csv`; expected ~0)."]

    rows_md += ["", "Top cross-dataset pairs are in `cross_dataset_top_pairs.csv`; "
                "inspect the highest-cosine rows manually -- fundus frames are all "
                "centred discs on black, so even unrelated images sit around cos 0.6-0.85."]
    (OUT_DIR / "embedding_neardup_summary.md").write_text("\n".join(rows_md) + "\n",
                                                          encoding="utf-8")
    print("\n".join(rows_md))
    print(f"\nwrote {OUT_DIR}/embedding_neardup_summary.md (+2 csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
