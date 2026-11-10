"""
Phase 26b - resolution-matched control for Phase 26.

CNN Grad-CAM++ maps come from a 7x7 feature grid (EfficientNet-B3 / ResNet-50
at 224 px); RETFound ViT Grad-CAM maps come from a 14x14 patch grid. A finer
map can score higher localisation simply because it is sharper. This script
re-scores the cached maps (outputs/phase26/cams.npz) after degrading every
map to a 7x7 grid (average-pool 224->7, bilinear back to 224), so all models
are compared at the CNN's native resolution.

Outputs: outputs/phase26/resolution_matched_summary.csv, ..._tests.csv,
         appended section in outputs/phase26/phase26_summary.md
Run:     python scripts/phase26b_resolution_matched.py --seg-root "<A. Segmentation>"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase26_lesion_localization as p26  # noqa: E402


def to7(cam: np.ndarray) -> np.ndarray:
    c = cam.reshape(7, 32, 7, 32).mean(axis=(1, 3)).astype(np.float32)
    return np.asarray(Image.fromarray(c).resize((224, 224), Image.BILINEAR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-root", required=True)
    ap.add_argument("--tol", type=float, default=4.0)
    a = ap.parse_args()
    out = p26.OUT
    z = np.load(out / "cams.npz", allow_pickle=True)
    cams = z["cams"].item()
    ids = list(z["ids"])
    items = p26.find_items(Path(a.seg_root))
    rows = []
    for i, iid in enumerate(ids):
        ip, mp = items[iid]
        with Image.open(ip) as im:
            arr = np.asarray(im.convert("RGB"))
        box = p26.bbox_of(arr)
        rgb = np.asarray(Image.fromarray(p26.to_square(arr, box)).resize((224, 224), Image.BILINEAR))
        fov = rgb.mean(2) > 15
        L = {k: p26.mask_to_224(p26.load_mask(mp[k], arr.shape[:2]), box) for k in p26.LESIONS if k in mp}
        for k in p26.LESIONS:
            L.setdefault(k, np.zeros((224, 224), bool))
        L["any"] = L["MA"] | L["HE"] | L["EX"] | L["SE"]
        if "SE" not in mp:
            L.pop("SE")
        od = p26.mask_to_224(p26.load_mask(mp["OD"], arr.shape[:2]), box) if "OD" in mp else None
        for m in cams:
            for res, cam in [("native", cams[m][i]), ("7x7", to7(cams[m][i]))]:
                rows.append({"image": iid, "model": m, "resolution": res,
                             **p26.loc_metrics(cam, fov, L, od, a.tol)})
    df = pd.DataFrame(rows)
    keys = ["lift_any", "auroc_lesion", "pointing_hit", "top10_precision", "energy_od"]
    summ = []
    for (m, r), g in df.groupby(["model", "resolution"]):
        d = {"model": m, "resolution": r}
        for k in keys:
            lo, hi = p26.boot_ci(g[k].astype(float))
            d[k] = float(g[k].astype(float).mean()); d[k + "_lo"] = lo; d[k + "_hi"] = hi
        summ.append(d)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "resolution_matched_summary.csv", index=False)

    from scipy.stats import wilcoxon
    base = df[(df.model == "baseline_efficientnet_b3") & (df.resolution == "native")].set_index("image")
    tests = []
    for m in ["retfound_full", "retfound_linprobe"]:
        g = df[(df.model == m) & (df.resolution == "7x7")].set_index("image")
        for k in ["lift_any", "auroc_lesion", "top10_precision"]:
            x, y = g[k].astype(float), base.loc[g.index, k].astype(float)
            ok = ~(x.isna() | y.isna())
            x, y = x[ok], y[ok]
            tests.append({"model": m, "metric": k, "retfound_7x7": x.mean(), "effb3_native": y.mean(),
                          "diff": (x - y).mean(), "frac_better": (x > y).mean(),
                          "wilcoxon_p": float(wilcoxon(x, y).pvalue)})
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "resolution_matched_tests.csv", index=False)

    lab = {k: v[0] for k, v in p26.MODELS.items()}
    L = ["", "## Resolution-matched control (Phase 26b)\n",
         "All maps degraded to the CNN's native 7x7 grid (average-pool, bilinear back to 224).\n",
         "| Model | resolution | lift (any) | pixel AUROC | pointing hit | top-10% precision |", "|---|---|---|---|---|---|"]
    for _, r in summ.sort_values(["model", "resolution"]).iterrows():
        f = lambda k: f"{r[k]:.2f} [{r[k+'_lo']:.2f}, {r[k+'_hi']:.2f}]"
        L.append(f"| {lab[r.model]} | {r.resolution} | {f('lift_any')} | {f('auroc_lesion')} | {f('pointing_hit')} | {f('top10_precision')} |")
    L += ["", "| RETFound (7x7) vs EfficientNet-B3 (native) | metric | RETFound | EffB3 | Δ | images better | p |", "|---|---|--:|--:|--:|--:|--:|"]
    for _, r in tests.iterrows():
        L.append(f"| {lab[r.model]} | {r.metric} | {r.retfound_7x7:.3f} | {r.effb3_native:.3f} | {r['diff']:+.3f} | {r.frac_better:.2f} | {r.wilcoxon_p:.2g} |")
    sp = out / "phase26_summary.md"
    txt = sp.read_text(encoding="utf-8").split("\n## Resolution-matched control")[0].rstrip("\n")
    sp.write_text(txt + "\n" + "\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
