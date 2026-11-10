"""
Phase 26 - quantitative saliency localisation against expert lesion masks.

Reviewer request (round 2): "A clinical-plausibility rubric is not enough by
itself. Add quantitative localization evaluation against expert lesion
annotations if available."

Data: IDRiD Segmentation subset (Porwal et al. 2018; IEEE DataPort, CC BY 4.0):
81 colour fundus photographs (Kowa VX-10a, 4288x2848; 54 train + 27 test), each
with pixel-level expert masks for microaneurysms (MA), haemorrhages (HE), hard
exudates (EX), soft exudates (SE, subset of images) and the optic disc (OD).
These images were not used for training or model selection. Download
"A. Segmentation" from https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
and pass its folder with --seg-root (any layout: images *.jpg and masks
*_MA.tif / *_HE.tif / *_EX.tif / *_SE.tif / *_OD.tif are found recursively).

Pipeline (identical to model input): crop black border (bbox from the image),
zero-pad to square, resize to 224. Every mask goes through the SAME crop/pad,
then is downsampled with max-pooling semantics (a 224-pixel is lesion if any
source pixel in it is), so microaneurysms are not lost.

Saliency: the method-matched maps already used in the paper (Grad-CAM++ on the
last conv block for CNNs; Grad-CAM over last-block patch tokens for RETFound),
for the model's predicted class, bilinear-upsampled to 224.

Per image x model metrics (all restricted to the fundus field of view):
  energy_lesion   share of saliency mass on any lesion (MA|HE|EX|SE)
  lift_lesion     energy_lesion / lesion area share (1 = uniform map)
  energy_<type>, lift_<type>  per lesion type
  energy_od       share of saliency mass on the expert optic-disc mask
  pointing_hit    saliency peak within `--tol` px of a lesion
  auroc_lesion    pixel-level AUROC of saliency for lesion vs non-lesion
  top10_precision fraction of the top-10% saliency pixels that are lesion
Controls, computed with the same metrics:
  uniform         (analytic) energy = area share, lift = 1, AUROC = 0.5
  centre          isotropic Gaussian centred on the fundus (sigma = 0.35 R)
  cross_image     the model's map for a DIFFERENT image (derangement, seed 42):
                  tests whether the map tracks this image's lesions or just a
                  generic location habit.

Statistics: per-model mean with bootstrap 95% CI over images (B=2000);
paired Wilcoxon signed-rank model vs each control and vs EfficientNet-B3.

Outputs (outputs/phase26/):
  localization_per_image.csv, localization_summary.csv, paired_tests.csv,
  fig_localization.png (example overlays), phase26_summary.md

Run (Windows, GPU):
  .venv\\Scripts\\python.exe scripts\\phase26_lesion_localization.py --seg-root "C:\\data\\IDRiD\\A. Segmentation" --device cuda
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty lesion types -> NaN by design
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT = PROJECT_DIR / "outputs" / "phase26"
LESIONS = ["MA", "HE", "EX", "SE"]
SIZE = 224
TOL = 7  # border tolerance, matches preprocessing.DEFAULT_BORDER_TOL
MODELS = {
    "baseline_efficientnet_b3": ("EfficientNet-B3", "outputs/phase4_baseline/efficientnet_b3_best.pt"),
    "baseline_resnet50": ("ResNet-50", "outputs/phase4_baseline/resnet50_best.pt"),
    "retfound_linprobe": ("RETFound linear probe", "outputs/phase5_retfound/linprobe_best.pt"),
    "retfound_full": ("RETFound full fine-tune", "outputs/phase5_retfound/full_best.pt"),
}


# ----------------------------------------------------------------- geometry
def bbox_of(arr: np.ndarray):
    m = arr.mean(2) > TOL
    if not m.any():
        return 0, arr.shape[0], 0, arr.shape[1]
    r = np.where(m.any(1))[0]
    c = np.where(m.any(0))[0]
    return r[0], r[-1] + 1, c[0], c[-1] + 1


def to_square(a: np.ndarray, box) -> np.ndarray:
    r0, r1, c0, c1 = box
    a = a[r0:r1, c0:c1]
    h, w = a.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s) + a.shape[2:], dtype=a.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = a
    return out


def mask_to_224(m: np.ndarray, box) -> np.ndarray:
    sq = to_square((m > 0).astype(np.float32), box)
    small = np.asarray(Image.fromarray(sq).resize((SIZE, SIZE), Image.BOX))
    return small > 0          # any lesion pixel inside the 224-cell


def load_mask(p: Path, shape) -> np.ndarray:
    with Image.open(p) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        a = a.max(2)
    if a.shape != shape:
        a = np.asarray(Image.fromarray((a > 0).astype(np.uint8) * 255).resize(
            (shape[1], shape[0]), Image.NEAREST))
    return a > 0


def find_items(root: Path):
    imgs = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and re.fullmatch(r"IDRiD_\d+", p.stem):
            imgs[p.stem] = p
    masks = {}
    for p in root.rglob("*.tif"):
        mm = re.fullmatch(r"(IDRiD_\d+)_(MA|HE|EX|SE|OD)", p.stem)
        if mm:
            masks.setdefault(mm.group(1), {})[mm.group(2)] = p
    return {k: (imgs[k], masks.get(k, {})) for k in sorted(imgs) if k in masks}


# ----------------------------------------------------------------- metrics
def auroc(score, pos):
    from scipy.stats import rankdata
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def loc_metrics(cam, fov, L, od, tol):
    """cam, fov, masks: 224x224. Lesion types in dict L, union under 'any'."""
    c = np.where(fov, np.clip(cam, 0, None), 0.0)
    tot = c.sum() + 1e-12
    out = {}
    area = fov.sum()
    for k, m in L.items():
        mm = m & fov
        share = mm.sum() / area
        e = c[mm].sum() / tot
        out[f"energy_{k}"] = float(e)
        out[f"area_{k}"] = float(share)
        out[f"lift_{k}"] = float(e / share) if share > 0 else np.nan
    out["energy_od"] = float(c[od & fov].sum() / tot) if od is not None else np.nan
    U = L["any"] & fov
    # pointing game
    py, px = np.unravel_index(int(np.argmax(c)), c.shape)
    if U.any():
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(~U)
        out["pointing_hit"] = bool(dist[py, px] <= tol)
    else:
        out["pointing_hit"] = np.nan
    v = cam[fov]
    out["auroc_lesion"] = auroc(v, U[fov])
    thr = np.quantile(v, 0.90)
    top = fov & (cam >= thr)
    out["top10_precision"] = float(U[top].mean()) if top.any() else np.nan
    return out


def centre_prior(fov):
    ys, xs = np.where(fov)
    cy, cx = ys.mean(), xs.mean()
    R = np.sqrt(fov.sum() / np.pi)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.35 * R) ** 2))


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def boot_ci(x, B=2000, seed=42):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(B, len(x)), replace=True).mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-root", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--tol", type=float, default=4.0,
                    help="pointing-game tolerance in 224-px units (default 4 = ~1/4 ViT patch)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from-cache", action="store_true",
                    help="skip inference; recompute metrics from outputs/phase26/cams.npz")
    ap.add_argument("--out", default=None, help="output folder (default outputs/phase26)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)

    items = find_items(Path(a.seg_root))
    if not items:
        print(f"No IDRiD_XX images with *_MA/HE/EX/SE/OD.tif masks found under {a.seg_root}")
        return 1
    print(f"{len(items)} images with masks")

    # -------- images + masks at model resolution
    ids, rgbs, fovs, Ls, ODs = [], [], [], [], []
    for iid, (ip, mp) in items.items():
        with Image.open(ip) as im:
            arr = np.asarray(im.convert("RGB"))
        box = bbox_of(arr)
        sq = to_square(arr, box)
        pil = Image.fromarray(sq).resize((SIZE, SIZE), Image.BILINEAR)
        rgb = np.asarray(pil)
        fov = rgb.mean(2) > 15          # fundus field of view at 224
        L = {}
        for k in LESIONS:
            L[k] = mask_to_224(load_mask(mp[k], arr.shape[:2]), box) if k in mp else np.zeros((SIZE, SIZE), bool)
        L["any"] = L["MA"] | L["HE"] | L["EX"] | L["SE"]
        od = mask_to_224(load_mask(mp["OD"], arr.shape[:2]), box) if "OD" in mp else None
        ids.append(iid); rgbs.append(rgb); fovs.append(fov); Ls.append(L); ODs.append(od)
    has_se = {iid: "SE" in items[iid][1] for iid in ids}

    # -------- saliency
    cache = OUT / "cams.npz"
    if a.from_cache:
        z = np.load(cache, allow_pickle=True)
        cams = z["cams"].item()
        preds = z["preds"].item()
    else:
        import torch
        import torch.nn.functional as F
        from predict import load_checkpoint, pick_device
        from phase7_explainability import cnn_target_layer, gradcampp, vit_gradcam, to_tensor
        device = pick_device(a.device)
        cams, preds = {}, {}
        for m in a.models:
            model, _ = load_checkpoint(PROJECT_DIR / MODELS[m][1], device)
            model.eval()
            vit = "retfound" in m
            tl = None if vit else cnn_target_layer(model)
            cams[m], preds[m] = [], []
            for rgb in rgbs:
                x = to_tensor(Image.fromarray(rgb)).to(device)
                with torch.no_grad():
                    cls = int(model(x).argmax(1))
                cam = vit_gradcam(model, x, cls)[0] if vit else gradcampp(model, x, tl, cls)
                cam = np.asarray(F.interpolate(torch.tensor(cam)[None, None].float(),
                                               size=(SIZE, SIZE), mode="bilinear",
                                               align_corners=False)[0, 0])
                cams[m].append(np.clip(cam, 0, 1).astype(np.float32))
                preds[m].append(cls)
            print(f"saliency done: {m}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        np.savez_compressed(cache, cams=np.array(cams, dtype=object),
                            preds=np.array(preds, dtype=object), ids=np.array(ids))

    # -------- metrics
    rng = np.random.default_rng(a.seed)
    perm = derangement(len(ids), rng)
    rows = []
    for i, iid in enumerate(ids):
        fov, L, od = fovs[i], Ls[i], ODs[i]
        Lx = {k: v for k, v in L.items() if k != "SE" or has_se[iid]}
        base = {"image": iid, "n_lesion_px": int((L["any"] & fov).sum())}
        rows.append({**base, "model": "centre", "saliency": "control",
                     **loc_metrics(centre_prior(fov), fov, Lx, od, a.tol)})
        for m in cams:
            rows.append({**base, "model": m, "saliency": "model", "pred_grade": preds[m][i],
                         **loc_metrics(cams[m][i], fov, Lx, od, a.tol)})
            rows.append({**base, "model": m, "saliency": "cross_image",
                         **loc_metrics(cams[m][perm[i]], fov, Lx, od, a.tol)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "localization_per_image.csv", index=False)

    metrics = ["energy_any", "lift_any", "pointing_hit", "auroc_lesion", "top10_precision",
               "lift_MA", "lift_HE", "lift_EX", "lift_SE", "energy_od", "area_any"]
    srows = []
    for (m, s), g in df.groupby(["model", "saliency"]):
        r = {"model": m, "saliency": s, "n_images": len(g)}
        for k in metrics:
            v = g[k].astype(float)
            lo, hi = boot_ci(v)
            r[k] = float(np.nanmean(v)); r[k + "_lo"] = lo; r[k + "_hi"] = hi
        srows.append(r)
    summ = pd.DataFrame(srows)
    summ.to_csv(OUT / "localization_summary.csv", index=False)

    from scipy.stats import wilcoxon
    trows = []
    for m in cams:
        g = df[(df.model == m) & (df.saliency == "model")].set_index("image")
        comps = {"centre": df[df.model == "centre"].set_index("image"),
                 "cross_image": df[(df.model == m) & (df.saliency == "cross_image")].set_index("image")}
        if m != "baseline_efficientnet_b3" and "baseline_efficientnet_b3" in cams:
            comps["EfficientNet-B3"] = df[(df.model == "baseline_efficientnet_b3") & (df.saliency == "model")].set_index("image")
        for cname, h in comps.items():
            for k in ["lift_any", "auroc_lesion", "energy_any", "top10_precision", "energy_od"]:
                x, y = g[k].astype(float), h.loc[g.index, k].astype(float)
                ok = ~(x.isna() | y.isna())
                try:
                    p = float(wilcoxon(x[ok], y[ok]).pvalue)
                except ValueError:
                    p = np.nan
                trows.append({"model": m, "vs": cname, "metric": k,
                              "mean_model": float(x[ok].mean()), "mean_comparator": float(y[ok].mean()),
                              "mean_diff": float((x[ok] - y[ok]).mean()),
                              "frac_images_better": float((x[ok] > y[ok]).mean()),
                              "wilcoxon_p": p})
    tests = pd.DataFrame(trows)
    tests.to_csv(OUT / "paired_tests.csv", index=False)
    make_figure(ids, rgbs, Ls, cams, fovs)
    write_summary(summ, tests, len(ids), a)
    return 0


def make_figure(ids, rgbs, Ls, cams, fovs, n=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort([-(L["any"] & f).sum() for L, f in zip(Ls, fovs)])
    pick = order[np.linspace(0, len(order) - 1, n).astype(int)]
    ms = list(cams)
    fig, ax = plt.subplots(n, len(ms) + 1, figsize=(2.6 * (len(ms) + 1), 2.6 * n))
    for r, i in enumerate(pick):
        ax[r, 0].imshow(rgbs[i]); ax[r, 0].contour(Ls[i]["any"], colors="yellow", linewidths=0.5)
        ax[r, 0].set_title(f"{ids[i]} lesions", fontsize=7)
        for c, m in enumerate(ms):
            ax[r, c + 1].imshow(rgbs[i]); ax[r, c + 1].imshow(cams[m][i], cmap="jet", alpha=0.45)
            ax[r, c + 1].contour(Ls[i]["any"], colors="white", linewidths=0.4)
            ax[r, c + 1].set_title(MODELS[m][0], fontsize=7)
    for a_ in ax.ravel():
        a_.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_localization.png", dpi=200)
    plt.close(fig)


def write_summary(summ, tests, n, a):
    lab = {k: v[0] for k, v in MODELS.items()} | {"centre": "Centre-Gaussian control"}
    L = ["# Phase 26 - saliency localisation vs. IDRiD expert lesion masks\n",
         f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}; n = {n} images "
         f"(IDRiD Segmentation subset), pointing tolerance {a.tol} px at 224, bootstrap B=2000._\n",
         "Lift = share of saliency mass on lesions / share of the field of view that is lesion "
         "(uniform map = 1). Cross-image = the model's map for a different image.\n",
         "| Map | energy on lesions | lift (any) | pointing hit | pixel AUROC | top-10% precision | lift MA | lift HE | lift EX | lift SE | energy on OD |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda r, k: f"{r[k]:.2f} [{r[k+'_lo']:.2f}, {r[k+'_hi']:.2f}]"
    g = lambda r, k: f"{r[k]:.2f}"
    for _, r in summ.sort_values(["saliency", "model"]).iterrows():
        name = lab.get(r.model, r.model) + ("" if r.saliency != "cross_image" else " (cross-image)")
        L.append(f"| {name} | {f(r,'energy_any')} | {f(r,'lift_any')} | {f(r,'pointing_hit')} | {f(r,'auroc_lesion')} | "
                 f"{f(r,'top10_precision')} | {g(r,'lift_MA')} | {g(r,'lift_HE')} | {g(r,'lift_EX')} | {g(r,'lift_SE')} | {g(r,'energy_od')} |")
    a_ = summ.iloc[0]
    L += ["", f"Mean lesion share of the field of view: {a_['area_any']:.3f}.", "",
          "## Paired comparisons (Wilcoxon signed-rank over images)\n",
          "| Model | vs | metric | model | comparator | Δ | images better | p |", "|---|---|---|--:|--:|--:|--:|--:|"]
    for _, r in tests.iterrows():
        L.append(f"| {lab[r.model]} | {r.vs} | {r.metric} | {r.mean_model:.3f} | {r.mean_comparator:.3f} | "
                 f"{r.mean_diff:+.3f} | {r.frac_images_better:.2f} | {r.wilcoxon_p:.2g} |")
    (OUT / "phase26_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
