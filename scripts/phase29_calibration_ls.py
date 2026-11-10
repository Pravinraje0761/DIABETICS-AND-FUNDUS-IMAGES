"""
Phase 29b - calibration on the official IDRiD release and the label-smoothing
control (EfficientNet-B3 trained with label smoothing 0.1, the RETFound recipe's
value). Question: are RETFound's lower external ECE and flat entropy a property
of retinal pretraining, or of label smoothing?

Outputs: outputs/phase29/calibration_ls/{calibration.csv, ls_control.csv, summary.md}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase29" / "calibration_ls"
P29 = ROOT / "outputs" / "phase29" / "preds_official"
P28 = ROOT / "outputs" / "phase28"
K = 5
B = 2000
MODELS = {
    "baseline_efficientnet_b3": ("EfficientNet-B3", lambda d: P29 / f"baseline_efficientnet_b3__{d}_preds.csv"),
    "baseline_efficientnet_b3_ls01": ("EfficientNet-B3 + label smoothing 0.1",
                                      lambda d: P28 / f"baseline_efficientnet_b3_ls01__{'IDRiD-official' if d == 'IDRiD' else d}_preds.csv"),
    "baseline_resnet50": ("ResNet-50", lambda d: P29 / f"baseline_resnet50__{d}_preds.csv"),
    "retfound_linprobe": ("RETFound linear probe", lambda d: P29 / f"retfound_linprobe__{d}_preds.csv"),
    "retfound_full": ("RETFound full fine-tune", lambda d: P29 / f"retfound_full__{d}_preds.csv"),
}
DATASETS = ["APTOS-internal", "Messidor-2", "IDRiD"]


def qwk(y, yh):
    O = np.zeros((K, K)); np.add.at(O, (y, yh), 1)
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()


def ece(P, y, bins=15):
    conf = P.max(1); pred = P.argmax(1); acc = pred == y
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def auroc(s, pos):
    from scipy.stats import rankdata
    n1 = pos.sum(); n0 = len(pos) - n1
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    data = {}
    for m, (lab, f) in MODELS.items():
        for d in DATASETS:
            df = pd.read_csv(f(d))
            P = df[[f"prob_{k}" for k in range(K)]].to_numpy(float)
            y = df.y_true.to_numpy(int); yh = P.argmax(1)
            data[(m, d)] = (y, yh, P)
            onehot = np.eye(K)[y]
            err = yh != y
            H = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
            eb = [ece(P[b], y[b]) for b in rng.integers(0, len(y), (500, len(y)))]
            rows.append({"model": m, "label": lab, "dataset": d, "n": len(y),
                         "qwk": qwk(y, yh), "accuracy": float((~err).mean()),
                         "ece": ece(P, y), "ece_lo": np.percentile(eb, 2.5), "ece_hi": np.percentile(eb, 97.5),
                         "brier": float(((P - onehot) ** 2).sum(1).mean()),
                         "mean_entropy": float(H.mean()), "mean_max_prob": float(P.max(1).mean()),
                         "auroc_err_entropy": auroc(H, err), "auroc_err_msp": auroc(1 - P.max(1), err)})
    cal = pd.DataFrame(rows)
    cal.to_csv(OUT / "calibration.csv", index=False)

    # paired bootstrap: QWK drop of the label-smoothed CNN vs the original CNN and vs RETFound full
    ls = []
    for d in ["Messidor-2", "IDRiD"]:
        nI = len(data[("baseline_efficientnet_b3", "APTOS-internal")][0]); nE = len(data[("baseline_efficientnet_b3", d)][0])
        bI = rng.integers(0, nI, (B, nI)); bE = rng.integers(0, nE, (B, nE))
        def drop(m, i, e):
            yI, hI, _ = data[(m, "APTOS-internal")]; yE, hE, _ = data[(m, d)]
            return qwk(yI[i], hI[i]) - qwk(yE[e], hE[e])
        for comp in ["baseline_efficientnet_b3", "retfound_full"]:
            obs = drop("baseline_efficientnet_b3_ls01", slice(None), slice(None)) - drop(comp, slice(None), slice(None))
            bs = np.array([drop("baseline_efficientnet_b3_ls01", i, e) - drop(comp, i, e) for i, e in zip(bI, bE)])
            ls.append({"external": d, "comparison": f"EffB3+LS minus {MODELS[comp][0]}",
                       "ddrop": obs, "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                       "p_two_sided": min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))})
    lsd = pd.DataFrame(ls)
    lsd.to_csv(OUT / "ls_control.csv", index=False)

    L = ["# Phase 29b - calibration (official IDRiD) and label-smoothing control\n",
         "ECE: 15 equal-width bins on max-probability, 95% bootstrap CI (B=500). Error AUROC: grade error.\n",
         "| Model | Dataset | QWK | ECE [CI] | Brier | mean entropy | mean max-prob | AUROC(err) entropy | AUROC(err) MSP |",
         "|---|---|--:|---|--:|--:|--:|--:|--:|"]
    for _, r in cal.iterrows():
        L.append(f"| {r.label} | {r.dataset} | {r.qwk:.3f} | {r.ece:.3f} [{r.ece_lo:.3f}, {r.ece_hi:.3f}] | {r.brier:.3f} | "
                 f"{r.mean_entropy:.3f} | {r.mean_max_prob:.3f} | {r.auroc_err_entropy:.3f} | {r.auroc_err_msp:.3f} |")
    L += ["", "## Difference in internal-to-external QWK drop (paired bootstrap, B=2000)\n",
          "| External | Comparison | ΔDrop | 95% CI | p |", "|---|---|--:|---|--:|"]
    for _, r in lsd.iterrows():
        L.append(f"| {r.external} | {r.comparison} | {r.ddrop:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] | {r.p_two_sided:.3f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
