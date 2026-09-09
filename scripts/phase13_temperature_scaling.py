"""
Phase 13 -- post-hoc temperature scaling and recalibrated reliability metrics.

Closes the calibration follow-up named in the manuscript Limitations / Methods
(Section "Calibration"): the primary analysis reports ECE/MCE/Brier/NLL on the
raw softmax outputs but applies no post-hoc temperature scaling, because a
probability-calibration split held out *separately* from the internal test set
was never retained.

This script does the temperature-scaling analysis anyway, honestly:

  * It fits a single scalar temperature T > 0 per model by minimising the
    negative log-likelihood on a *calibration* set, then applies that T to the
    internal-test and both external prediction files and recomputes the
    calibration metrics.
  * T is a monotonic rescaling of the logits, so it never changes argmax:
    accuracy and quadratic-weighted kappa are recomputed only as a sanity check
    (they must match the raw values to ~1e-9).
  * The prediction files store probabilities, not logits. We reconstruct
    logit_c = log(prob_c); softmax(log(p) / T) is the standard tempered
    distribution and is exactly p at T = 1, so this is lossless for T-scaling.

CALIBRATION SET -- read this before quoting the numbers
-------------------------------------------------------
Ideally --calib-preds points at predictions on an APTOS split that no model
selection or reporting ever touched. That file does not exist yet; generate it
with (RETFound example; repeat per checkpoint):

  set DR_DATA_ROOT=C:\\dr_run          # PowerShell: $env:DR_DATA_ROOT="C:\\dr_run"
  .venv\\Scripts\\python.exe scripts\\predict.py ^
      --checkpoint outputs\\phase5_retfound\\full_best.pt ^
      --manifest data_splits\\aptos_manifest.csv --split val ^
      --out-prefix outputs\\phase13\\calib\\retfound_full__APTOS-val

  then: python scripts/phase13_temperature_scaling.py \
            --calib-preds outputs/phase13/calib --calib-tag APTOS-val

Without --calib-preds the script falls back to fitting T on the internal APTOS
*test* predictions (outputs/phase6/*__APTOS-internal_preds.csv). That is an
in-sample fit and will understate post-scaling ECE on the internal set; the
external post-scaling ECE is still informative because T is fitted without
seeing external data. The output CSV records which mode was used per row.

Outputs (outputs/phase13/):
  temperature_scaling.csv    one row per model x dataset, raw vs scaled metrics
  temperature_scaling.md     short human-readable summary + fitted T per model

Pure numpy + scipy; no torch, no GPU. Runs in a couple of seconds.
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROB_COLS = [f"prob_{c}" for c in range(5)]
EPS = 1e-12

# model key -> pretty label, matching the manuscript tables
MODEL_LABELS = {
    "baseline_efficientnet_b3": "EfficientNet-B3 (baseline)",
    "baseline_resnet50": "ResNet-50 (baseline)",
    "retfound_linprobe": "RETFound (linear probe)",
    "retfound_full": "RETFound (full fine-tune)",
}
DATASET_ORDER = {"APTOS-internal": 0, "Messidor-2": 1, "IDRiD": 2}

# outputs/phase6 uses "<model>__<dataset>_preds.csv"
_PREDS_RE = re.compile(r"^(?P<model>.+?)__(?P<dataset>.+?)_preds\.csv$")


# --------------------------------------------------------------------------- #
# calibration metrics (15 equal-width confidence bins, top-1 confidence)
# mirrors scripts/phase11_review_response.py so numbers are comparable
# --------------------------------------------------------------------------- #
def _reliability(prob: np.ndarray, y_true: np.ndarray, n_bins: int = 15):
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = mce = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        gap = abs(correct[m].mean() - conf[m].mean())
        w = m.sum() / n
        ece += w * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def _brier(prob: np.ndarray, y_true: np.ndarray) -> float:
    onehot = np.eye(prob.shape[1])[y_true]
    return float(((prob - onehot) ** 2).sum(axis=1).mean())


def _nll(prob: np.ndarray, y_true: np.ndarray) -> float:
    p_true = prob[np.arange(len(y_true)), y_true]
    return float(-np.log(np.clip(p_true, EPS, 1.0)).mean())


def _qwk(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    o = np.zeros((k, k), dtype=float)
    for t, p in zip(y_true, y_pred):
        o[t, p] += 1
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2
    act = o.sum(axis=1)
    prd = o.sum(axis=0)
    e = np.outer(act, prd) / max(o.sum(), 1.0)
    denom = (w * e).sum()
    return float(1.0 - (w * o).sum() / denom) if denom > 0 else float("nan")


def _temper(prob: np.ndarray, T: float) -> np.ndarray:
    """softmax(log(prob) / T), row-wise, numerically stable."""
    z = np.log(np.clip(prob, EPS, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_temperature(prob: np.ndarray, y_true: np.ndarray) -> float:
    def obj(logT: float) -> float:
        return _nll(_temper(prob, float(np.exp(logT))), y_true)

    res = minimize_scalar(obj, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def _bootstrap_temperature(prob: np.ndarray, y_true: np.ndarray, b: int, seed: int = 42):
    """Refit T on `b` bootstrap resamples of the calibration set -> (median, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    ts = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        ts[i] = _fit_temperature(prob[idx], y_true[idx])
    return float(np.median(ts)), float(np.percentile(ts, 2.5)), float(np.percentile(ts, 97.5))


# --------------------------------------------------------------------------- #
def _load(preds_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(preds_csv)
    prob = df[PROB_COLS].to_numpy(dtype=float)
    prob = prob / prob.sum(axis=1, keepdims=True)  # guard against tiny drift
    return prob, df["y_true"].to_numpy(dtype=int)


def _index_preds_dir(d: Path) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for p in sorted(glob.glob(str(d / "*_preds.csv"))):
        m = _PREDS_RE.match(Path(p).name)
        if m:
            out[(m["model"], m["dataset"])] = Path(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-dir", default="outputs/phase6",
                    help="dir with <model>__<dataset>_preds.csv (default outputs/phase6)")
    ap.add_argument("--calib-preds", default=None,
                    help="dir with held-out calibration preds "
                         "(<model>__<calib-tag>_preds.csv). If omitted, T is fitted "
                         "in-sample on the APTOS-internal test preds.")
    ap.add_argument("--calib-tag", default="APTOS-val",
                    help="dataset tag inside --calib-preds filenames")
    ap.add_argument("--out-dir", default="outputs/phase13")
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--bootstrap-T", type=int, default=0,
                    help="if >0, also refit T on this many bootstrap resamples of "
                         "the calibration set and report the 95%% interval")
    args = ap.parse_args()

    preds_dir = (PROJECT_DIR / args.preds_dir).resolve()
    out_dir = (PROJECT_DIR / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_index = _index_preds_dir(preds_dir)
    if not eval_index:
        raise SystemExit(f"no *_preds.csv under {preds_dir}")

    calib_index: dict[str, Path] = {}
    if args.calib_preds:
        cdir = (PROJECT_DIR / args.calib_preds).resolve()
        for (model, ds), p in _index_preds_dir(cdir).items():
            if ds == args.calib_tag:
                calib_index[model] = p
        if not calib_index:
            raise SystemExit(
                f"no *__{args.calib_tag}_preds.csv under {cdir} -- see the "
                f"docstring for how to generate them")
        calib_mode = f"held-out:{args.calib_tag}"
    else:
        for (model, ds), p in eval_index.items():
            if ds == "APTOS-internal":
                calib_index[model] = p
        calib_mode = "in-sample:APTOS-internal-test"
        print("!! no --calib-preds: fitting T in-sample on APTOS-internal test "
              "predictions. Internal post-scaling ECE will be optimistic; see "
              "the script docstring.\n")

    models = sorted({m for (m, _) in eval_index},
                    key=lambda k: list(MODEL_LABELS).index(k) if k in MODEL_LABELS else 99)

    rows: list[dict] = []
    fitted_T: dict[str, float] = {}
    boot_T: dict[str, tuple] = {}
    for model in models:
        if model not in calib_index:
            print(f"skip {model}: no calibration preds")
            continue
        cprob, cy = _load(calib_index[model])
        T = _fit_temperature(cprob, cy)
        fitted_T[model] = T
        if args.bootstrap_T > 0:
            boot_T[model] = _bootstrap_temperature(cprob, cy, args.bootstrap_T)
        for (m, ds), p in sorted(eval_index.items(),
                                 key=lambda kv: (kv[0][0], DATASET_ORDER.get(kv[0][1], 9))):
            if m != model:
                continue
            prob, y = _load(p)
            scaled = _temper(prob, T)
            raw_ece, raw_mce = _reliability(prob, y, args.n_bins)
            sc_ece, sc_mce = _reliability(scaled, y, args.n_bins)
            rows.append({
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "dataset": ds,
                "n": int(len(y)),
                "calib_mode": calib_mode,
                "temperature": round(T, 4),
                "T_boot_lo": round(boot_T[model][1], 4) if model in boot_T else "",
                "T_boot_hi": round(boot_T[model][2], 4) if model in boot_T else "",
                "ECE_raw": round(raw_ece, 4), "ECE_scaled": round(sc_ece, 4),
                "MCE_raw": round(raw_mce, 4), "MCE_scaled": round(sc_mce, 4),
                "Brier_raw": round(_brier(prob, y), 4),
                "Brier_scaled": round(_brier(scaled, y), 4),
                "NLL_raw": round(_nll(prob, y), 4),
                "NLL_scaled": round(_nll(scaled, y), 4),
                "QWK_raw": round(_qwk(y, prob.argmax(1)), 4),
                "QWK_scaled": round(_qwk(y, scaled.argmax(1)), 4),
            })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "temperature_scaling.csv"
    df.to_csv(csv_path, index=False)

    # sanity: T-scaling must not move argmax
    moved = df[(df["QWK_raw"] - df["QWK_scaled"]).abs() > 1e-6]
    if len(moved):
        print("WARNING: QWK changed after temperature scaling (argmax moved?):")
        print(moved[["model", "dataset", "QWK_raw", "QWK_scaled"]].to_string(index=False))

    lines = [
        "# Phase 13 -- post-hoc temperature scaling",
        "",
        f"Calibration mode: **{calib_mode}**  ",
        f"Bins: {args.n_bins} equal-width, top-1 confidence  ",
        f"Eval predictions: `{args.preds_dir}`",
        "",
        "## Fitted temperature (per model)",
        "",
        "| Model | T | reading |",
        "|---|--:|---|",
    ]
    for model, T in fitted_T.items():
        rd = ("well-scaled already" if abs(T - 1) < 0.05
              else "over-confident -> softened (T>1)" if T > 1
              else "under-confident -> sharpened (T<1)")
        bt = (f" (bootstrap 95% CI {boot_T[model][1]:.2f}-{boot_T[model][2]:.2f})"
              if model in boot_T else "")
        lines.append(f"| {MODEL_LABELS.get(model, model)} | {T:.3f}{bt} | {rd} |")
    lines += [
        "",
        "## ECE before -> after, by dataset",
        "",
        "| Model | Dataset | n | ECE raw | ECE scaled | Brier raw | Brier scaled | NLL raw | NLL scaled |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model_label']} | {r['dataset']} | {r['n']} | "
            f"{r['ECE_raw']:.3f} | {r['ECE_scaled']:.3f} | "
            f"{r['Brier_raw']:.3f} | {r['Brier_scaled']:.3f} | "
            f"{r['NLL_raw']:.3f} | {r['NLL_scaled']:.3f} |"
        )
    lines += [
        "",
        "_Temperature scaling is monotonic in the logits, so accuracy and QWK are",
        "unchanged (verified: `QWK_raw == QWK_scaled`). Only the probability",
        "calibration changes._",
    ]
    md_path = out_dir / "temperature_scaling.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nwrote {csv_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
