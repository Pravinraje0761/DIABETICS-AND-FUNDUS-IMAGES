"""
Phase 20 -- domain-specific OOD detection vs. predictive-entropy error detection.

Phase 8 showed predictive entropy is a mediocre-to-poor error detector on the
external sets, and (per the extension in that same phase) RETFound's
label-smoothed entropy barely separates its own errors at all on IDRiD. This
script asks the reviewer's question directly: can a domain-shift-specific OOD
score do better than entropy at flagging which external predictions are
wrong?

OOD score: Mahalanobis distance-to-nearest-class-centroid in an ImageNet
EfficientNet-B3 feature space (the SAME frozen, DR-naive extractor Phase 14
already uses for the near-duplicate check -- reused here so the OOD detector
cannot have learned anything about the DR task or about which dataset an
image comes from). Class-conditional means + a Ledoit-Wolf-shrunk SHARED
covariance are fit once on APTOS-train features (Lee et al. 2018 "Mahalanobis
detector" recipe); every eval image gets score = min over classes of the
squared Mahalanobis distance. This score is model-agnostic (same 1536-d
extractor for every row) -- only the "is this prediction wrong" label varies
by grading model, so it is scored once per dataset and compared against every
model's own errors and against that model's Phase 8 entropy-based error AUROC.

Outputs (outputs/phase20/):
  ood_master.csv     model x dataset: AUROC(OOD) vs AUROC(entropy), paired
                     bootstrap difference with 95% CI
  ood_summary.md      manuscript digest

Run:
  .venv/Scripts/python.exe scripts/phase20_ood_detection.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase14_embedding_dedup import _embed, _extractor  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_DIR / "data_splits"
PHASE6_DIR = PROJECT_DIR / "outputs" / "phase6"
PHASE8_SUMMARY = PROJECT_DIR / "outputs" / "phase8" / "uncertainty_singlepass.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "phase20"

MODELS = ["baseline_efficientnet_b3", "baseline_resnet50",
         "retfound_linprobe", "retfound_full"]
DATASETS = {
    "APTOS-internal": (SPLIT_DIR / "aptos_manifest.csv", "test"),
    "Messidor-2": (SPLIT_DIR / "messidor2_manifest.csv", "external"),
    "IDRiD": (SPLIT_DIR / "idrid_manifest.csv", "external"),
}
NUM_CLASSES = 5


def mahalanobis_fit(feats: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Class-conditional means + shared Ledoit-Wolf-shrunk covariance (inverse)."""
    lw = LedoitWolf().fit(feats)
    cov_inv = np.linalg.inv(lw.covariance_)
    means = np.stack([feats[labels == c].mean(axis=0) for c in range(NUM_CLASSES)])
    return means, cov_inv


def mahalanobis_score(feats: np.ndarray, means: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    """min-over-class squared Mahalanobis distance -> (N,) OOD score (higher = more OOD)."""
    d = np.stack([np.einsum("ij,jk,ik->i", feats - m, cov_inv, feats - m) for m in means],
                axis=1)  # (N, num_classes)
    return d.min(axis=1)


def boot_auroc_diff(y_err: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                    B: int, seed: int) -> tuple[float, float, float]:
    """Paired bootstrap CI for AUROC(score_a) - AUROC(score_b) as error detectors."""
    rng = np.random.default_rng(seed)
    n = len(y_err)
    point = roc_auc_score(y_err, score_a) - roc_auc_score(y_err, score_b)
    draws = np.full(B, np.nan)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        ye = y_err[idx]
        if len(np.unique(ye)) < 2:
            continue
        draws[b] = roc_auc_score(ye, score_a[idx]) - roc_auc_score(ye, score_b[idx])
    d = draws[~np.isnan(draws)]
    lo, hi = (np.percentile(d, [2.5, 97.5]) if d.size else (np.nan, np.nan))
    return point, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = (torch.device("cuda") if (args.device != "cpu" and torch.cuda.is_available())
              else torch.device("cpu"))
    extractor = _extractor(device)

    train_feats, train_ids, _ = _embed(SPLIT_DIR / "aptos_manifest.csv", "train",
                                       extractor, device, args.batch_size, args.num_workers)
    aptos_df = pd.read_csv(SPLIT_DIR / "aptos_manifest.csv")
    aptos_df["id_code"] = aptos_df["id_code"].astype(str)
    aptos_df = aptos_df.set_index("id_code")
    train_labels = aptos_df.loc[[str(i) for i in train_ids], "diagnosis"].to_numpy(int)
    means, cov_inv = mahalanobis_fit(train_feats, train_labels)
    print(f"fit Mahalanobis OOD detector: {train_feats.shape[0]} APTOS-train images, "
         f"{train_feats.shape[1]}-d ImageNet EfficientNet-B3 features, "
         f"Ledoit-Wolf shrinkage covariance")

    ood_scores: dict[str, dict] = {}
    for dkey, (manifest, split) in DATASETS.items():
        feats, ids, _ = _embed(manifest, split, extractor, device,
                               args.batch_size, args.num_workers)
        scores = mahalanobis_score(feats, means, cov_inv)
        ood_scores[dkey] = {"id_code": np.asarray([str(i) for i in ids]), "score": scores}
        print(f"{dkey}: n={len(ids)}  OOD score mean={scores.mean():.1f} sd={scores.std():.1f}")

    entropy_ref = None
    if PHASE8_SUMMARY.is_file():
        entropy_ref = (pd.read_csv(PHASE8_SUMMARY)
                       .set_index(["model", "dataset"])["auroc_entropy_error_detection"])

    rows: list[dict] = []
    for model in MODELS:
        for dkey in DATASETS:
            p = PHASE6_DIR / f"{model}__{dkey}_preds.csv"
            if not p.is_file():
                continue
            preds = pd.read_csv(p)
            preds["id_code"] = preds["id_code"].astype(str)
            preds = preds.set_index("id_code")
            os_ = ood_scores[dkey]
            common = [i for i in os_["id_code"] if i in preds.index]
            if not common:
                continue
            pr = preds.loc[common]
            sc = pd.Series(os_["score"], index=os_["id_code"]).loc[common].to_numpy()
            ent = pr["entropy"].to_numpy(float)
            err = (pr["y_true"].to_numpy(int) != pr["y_pred"].to_numpy(int)).astype(int)
            if len(np.unique(err)) < 2:
                print(f"  {model} / {dkey}: no errors or no correct predictions, skip")
                continue
            auroc_ood = float(roc_auc_score(err, sc))
            auroc_ent = float(roc_auc_score(err, ent))
            diff, dlo, dhi = boot_auroc_diff(err, sc, ent, args.boot, args.seed)
            ent_ref = (float(entropy_ref.loc[(model, dkey)])
                      if entropy_ref is not None and (model, dkey) in entropy_ref.index
                      else float("nan"))
            rows.append({
                "model": model, "dataset": dkey, "n": len(common),
                "n_errors": int(err.sum()),
                "auroc_ood_mahalanobis": round(auroc_ood, 4),
                "auroc_entropy_recomputed": round(auroc_ent, 4),
                "auroc_entropy_phase8": round(ent_ref, 4) if np.isfinite(ent_ref) else None,
                "auroc_diff_ood_minus_entropy": round(diff, 4),
                "diff_ci_lo": round(dlo, 4), "diff_ci_hi": round(dhi, 4),
                "ood_significantly_better": bool(dlo > 0),
                "entropy_significantly_better": bool(dhi < 0),
            })
            print(f"{model:26s} {dkey:15s} AUROC(OOD)={auroc_ood:.3f} "
                 f"AUROC(entropy)={auroc_ent:.3f} diff={diff:+.3f} "
                 f"CI[{dlo:+.3f},{dhi:+.3f}]")

    master = pd.DataFrame(rows)
    master.to_csv(OUT_DIR / "ood_master.csv", index=False)

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L = ["# Phase 20 -- domain-shift OOD detection vs. entropy error detection", "",
        f"_Generated {ts}. OOD score = min-over-class squared Mahalanobis distance "
        f"in a frozen ImageNet EfficientNet-B3 feature space (class-conditional "
        f"means + Ledoit-Wolf shared covariance fit on APTOS-train only; same "
        f"extractor as Phase 14). Paired bootstrap on the difference in "
        f"error-detection AUROC, B={args.boot}, seed={args.seed}._", "",
        "| Model | Dataset | n | errors | AUROC(OOD) | AUROC(entropy) | "
        "diff (OOD-entropy), 95% CI | verdict |",
        "|---|---|--:|--:|--:|--:|---|---|"]
    for r in rows:
        verdict = ("OOD better" if r["ood_significantly_better"]
                  else "entropy better" if r["entropy_significantly_better"]
                  else "no sig. difference")
        L.append(f"| {r['model']} | {r['dataset']} | {r['n']} | {r['n_errors']} | "
                f"{r['auroc_ood_mahalanobis']:.3f} | {r['auroc_entropy_recomputed']:.3f} | "
                f"{r['auroc_diff_ood_minus_entropy']:+.3f} "
                f"({r['diff_ci_lo']:+.3f},{r['diff_ci_hi']:+.3f}) | {verdict} |")
    L += ["", "APTOS-internal rows are a control: the OOD detector is fit on this "
         "same distribution, so a large advantage there (if any) reflects in-domain "
         "behaviour, not domain-shift detection.", ""]
    (OUT_DIR / "ood_summary.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\nwrote {OUT_DIR}/ood_master.csv, ood_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
