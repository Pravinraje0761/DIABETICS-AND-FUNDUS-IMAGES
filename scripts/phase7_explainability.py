"""
Phase 7 - explainability.

For each model x external dataset we take a matched, grade-stratified sample of
correctly and incorrectly classified images (IDs read from the Phase 6 preds
CSVs, fixed seed) and produce a saliency map for the model's predicted class:

  * CNN  (EfficientNet-B3 / ResNet-50):  Grad-CAM++ on the last convolutional
    block (Chattopadhay et al. 2018).
  * ViT  (RETFound):                     attention rollout across all blocks
    (Abnar & Zuidema 2020); if the timm attention internals cannot be captured
    on this version, falls back to last-block gradient saliency and records
    which method was used per image.

Because this public IDRiD mirror ships no lesion segmentation masks and no second
human grader was available, clinical plausibility is scored with a transparent
*automated proxy* that operationalises the pre-registered rubric: a map is
"proxy-plausible" if the bulk of its top-decile saliency mass falls inside the
fundus, outside an estimated optic-disc disk, and outside the outer image border
annulus (i.e. not dominated by disc or border). This is a screen, not a
substitute for expert rating - see the manuscript Limitations.

Outputs (outputs/phase7/):
  saliency_per_image.csv     one row per sampled image (coords, proxy flags)
  plausibility_rates.csv     proxy-plausible % by model x dataset x correctness
  maps/<model>__<dataset>/*.png   overlay per image
  fig_explainability.png     6-panel selected figure for the manuscript
  phase7_summary.md          digest
  explainability_methods_paragraph.md   drop-in Methods text

Run (after Phase 4/5 checkpoints exist; uses GPU if free):
  .venv/Scripts/python.exe scripts/phase7_explainability.py
  .venv/Scripts/python.exe scripts/phase7_explainability.py --n-correct 12 --n-incorrect 8 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import GRADE_NAMES  # noqa: E402
from predict import load_checkpoint, pick_device  # noqa: E402
from preprocessing import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
PHASE6_DIR = PROJECT_DIR / "outputs" / "phase6"
SPLIT_DIR = PROJECT_DIR / "data_splits"
OUT_DIR = PROJECT_DIR / "outputs" / "phase7"

CKPTS = {
    "baseline_efficientnet_b3": PROJECT_DIR / "outputs/phase4_baseline/efficientnet_b3_best.pt",
    "baseline_resnet50": PROJECT_DIR / "outputs/phase4_baseline/resnet50_best.pt",
    "retfound_linprobe": PROJECT_DIR / "outputs/phase5_retfound/linprobe_best.pt",
    "retfound_full": PROJECT_DIR / "outputs/phase5_retfound/full_best.pt",
}
MANIFESTS = {
    "Messidor-2": SPLIT_DIR / "messidor2_manifest.csv",
    "IDRiD": SPLIT_DIR / "idrid_manifest.csv",
    "APTOS-internal": SPLIT_DIR / "aptos_manifest.csv",
}
MEAN = np.asarray(IMAGENET_MEAN)
STD = np.asarray(IMAGENET_STD)


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #
def load_processed(path: str):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB").resize((224, 224))


def to_tensor(pil) -> torch.Tensor:
    x = np.asarray(pil, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0)


def denorm(pil) -> np.ndarray:
    return np.asarray(pil, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Grad-CAM++  (CNN)
# --------------------------------------------------------------------------- #
def cnn_target_layer(model):
    name = model.__class__.__name__.lower()
    if "efficientnet" in name:
        return model.features[-1]
    if "resnet" in name:
        return model.layer4[-1]
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def gradcampp(model, x, target_layer, cls: int) -> np.ndarray:
    acts, grads = {}, {}

    def f_hook(_m, _i, o):
        acts["v"] = o.detach()

    def b_hook(_m, _gi, go):
        grads["v"] = go[0].detach()

    h1 = target_layer.register_forward_hook(f_hook)
    h2 = target_layer.register_full_backward_hook(b_hook)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits[0, cls].backward()
    h1.remove()
    h2.remove()

    A = acts["v"][0]
    G = grads["v"][0]
    g2, g3 = G.pow(2), G.pow(3)
    denom = 2 * g2 + (A * g3).sum(dim=(1, 2), keepdim=True)
    denom = torch.where(denom != 0, denom, torch.ones_like(denom))
    alpha = g2 / denom
    weights = (alpha * F.relu(G)).sum(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * A).sum(0)).cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam


# --------------------------------------------------------------------------- #
# Attention rollout  (ViT)
# --------------------------------------------------------------------------- #
def vit_attention_rollout(model, x) -> tuple[np.ndarray, str]:
    """Attention rollout (Abnar & Zuidema 2020).

    Robust capture: disable timm's fused-attention path and hook the input of
    each block's post-softmax ``attn_drop`` module, which is exactly the
    (B, heads, N, N) attention matrix. No re-implementation of timm internals.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return _vit_grad_saliency(model, x), "grad_saliency(no_blocks)"

    store: list[torch.Tensor] = []
    hooks = []
    restore_fused = []

    def hook(_m, inp, _out):
        a = inp[0]                       # (B, heads, N, N) post-softmax
        store.append(a.detach().mean(1).cpu())

    try:
        for blk in blocks:
            a = blk.attn
            if hasattr(a, "fused_attn"):
                restore_fused.append((a, a.fused_attn))
                a.fused_attn = False
            if hasattr(a, "attn_drop"):
                hooks.append(a.attn_drop.register_forward_hook(hook))
        model.eval()
        with torch.no_grad():
            model(x)
    except Exception:
        for h in hooks:
            h.remove()
        for a, f in restore_fused:
            a.fused_attn = f
        return _vit_grad_saliency(model, x), "grad_saliency(patch_failed)"
    finally:
        for h in hooks:
            h.remove()
        for a, f in restore_fused:
            a.fused_attn = f

    if not store:
        return _vit_grad_saliency(model, x), "grad_saliency(no_attn)"

    R = None
    for A in store:
        A = A[0]                          # (N, N)
        A = A + torch.eye(A.size(0))
        A = A / A.sum(dim=-1, keepdim=True)
        R = A if R is None else A @ R
    n_tok = R.size(0)
    grid = int(round((n_tok - 1) ** 0.5))
    m = R[0, 1:1 + grid * grid].reshape(grid, grid).numpy()
    if m.max() > m.min():
        m = (m - m.min()) / (m.max() - m.min())
    return m, "attention_rollout"


def vit_gradcam(model, x, cls: int) -> tuple[np.ndarray, str]:
    """Grad-CAM adapted to ViT patch tokens: gradient-weighted activation of the
    last transformer block's token outputs for class `cls`, CLS token dropped,
    ReLU, reshaped to the patch grid. Method-matched to the CNN Grad-CAM++ so the
    two model families are compared on the same kind of explanation."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return _vit_grad_saliency(model, x), "grad_saliency(no_blocks)"
    acts, grads = {}, {}
    tgt = blocks[-1]

    h1 = tgt.register_forward_hook(lambda _m, _i, o: acts.__setitem__("v", o))
    h2 = tgt.register_full_backward_hook(
        lambda _m, _gi, go: grads.__setitem__("v", go[0].detach()))
    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits[0, cls].backward()
    h1.remove()
    h2.remove()

    A = acts["v"][0].detach()             # (N, C)
    G = grads["v"][0]                     # (N, C)
    w = G.mean(dim=0)                     # (C,) grad-CAM channel weights
    cam = F.relu((A * w).sum(dim=1))     # (N,)
    cam = cam[1:]                         # drop CLS
    grid = int(round(cam.numel() ** 0.5))
    cam = cam[:grid * grid].reshape(grid, grid).cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam, "vit_gradcam"


def _vit_grad_saliency(model, x) -> np.ndarray:
    x = x.clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    cls = int(logits.argmax(1))
    logits[0, cls].backward()
    g = x.grad[0].abs().mean(0).cpu().numpy()
    if g.max() > g.min():
        g = (g - g.min()) / (g.max() - g.min())
    return g.reshape(14, 16, 14, 16).mean(axis=(1, 3))


# --------------------------------------------------------------------------- #
# plausibility proxy
# --------------------------------------------------------------------------- #
def fundus_mask(rgb01: np.ndarray) -> np.ndarray:
    return rgb01.mean(2) > 0.06


def estimate_disc(rgb01: np.ndarray, mask: np.ndarray) -> tuple[int, int, float]:
    from scipy.ndimage import gaussian_filter
    v = rgb01.mean(2).copy()
    v[~mask] = 0
    vb = gaussian_filter(v, sigma=6)
    cy, cx = np.unravel_index(int(np.argmax(vb)), vb.shape)
    r = 0.10 * np.sqrt(mask.sum() / np.pi) * 2
    return int(cy), int(cx), float(max(r, 14))


def plausibility(cam224: np.ndarray, rgb01: np.ndarray) -> dict:
    H = W = 224
    mask = fundus_mask(rgb01)
    cy, cx, r = estimate_disc(rgb01, mask)
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    if mask.any():
        ys, xs = np.where(mask)
        fcy, fcx = ys.mean(), xs.mean()
        frad = np.sqrt(mask.sum() / np.pi)
    else:
        fcy, fcx, frad = H / 2, W / 2, H / 2
    border = ((yy - fcy) ** 2 + (xx - fcx) ** 2) > (0.92 * frad) ** 2

    peak = np.unravel_index(int(np.argmax(cam224)), cam224.shape)
    thr = np.quantile(cam224[mask] if mask.any() else cam224, 0.90)
    top = cam224 >= thr
    top_mass = cam224[top].sum() + 1e-9
    frac_disc = cam224[top & disc].sum() / top_mass
    frac_border = cam224[top & border].sum() / top_mass
    frac_outside = cam224[top & ~mask].sum() / top_mass
    frac_lesionzone = cam224[top & mask & ~disc & ~border].sum() / top_mass

    proxy_plausible = bool(
        mask[peak] and not disc[peak] and not border[peak]
        and frac_lesionzone >= 0.5
    )
    return {
        "peak_y": int(peak[0]), "peak_x": int(peak[1]),
        "peak_in_disc": bool(disc[peak]),
        "peak_in_border": bool(border[peak]),
        "peak_outside_fundus": bool(not mask[peak]),
        "topmass_frac_disc": round(float(frac_disc), 3),
        "topmass_frac_border": round(float(frac_border), 3),
        "topmass_frac_outside": round(float(frac_outside), 3),
        "topmass_frac_lesionzone": round(float(frac_lesionzone), 3),
        "proxy_plausible": proxy_plausible,
        "disc_cy": cy, "disc_cx": cx, "disc_r": round(r, 1),
    }


# --------------------------------------------------------------------------- #
# sampling + overlay
# --------------------------------------------------------------------------- #
def sample_ids(preds: pd.DataFrame, n_correct: int, n_incorrect: int,
               seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    preds = preds.copy()
    preds["correct"] = preds["y_true"] == preds["y_pred"]
    picks = []
    for corr, n in [(True, n_correct), (False, n_incorrect)]:
        pool = preds[preds["correct"] == corr]
        if pool.empty:
            continue
        grades = sorted(pool["y_true"].unique())
        per = max(1, n // max(len(grades), 1))
        chosen = []
        for g in grades:
            sub = pool[pool["y_true"] == g]
            chosen.append(sub.sample(min(len(sub), per),
                                     random_state=int(rng.integers(1_000_000_000))))
        got = pd.concat(chosen) if chosen else pool.head(0)
        if len(got) < n:
            extra = pool.drop(got.index)
            if len(extra):
                got = pd.concat([got, extra.sample(
                    min(len(extra), n - len(got)),
                    random_state=int(rng.integers(1_000_000_000)))])
        picks.append(got.head(n))
    return pd.concat(picks).reset_index(drop=True) if picks else preds.head(0)


def overlay(rgb01, cam224, out_png: Path, title: str, proxy: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(1, 2, figsize=(6.2, 3.3))
    ax[0].imshow(rgb01)
    ax[0].set_title("image", fontsize=8)
    ax[0].axis("off")
    ax[1].imshow(rgb01)
    ax[1].imshow(cam224, cmap="jet", alpha=0.45)
    ax[1].add_patch(Circle((proxy["disc_cx"], proxy["disc_cy"]), proxy["disc_r"],
                           fill=False, ec="white", lw=1.0, ls="--"))
    ax[1].plot(proxy["peak_x"], proxy["peak_y"], "w+", ms=9, mew=2)
    ax[1].set_title("proxy-plausible" if proxy["proxy_plausible"]
                    else "proxy-implausible", fontsize=8)
    ax[1].axis("off")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run_model_dataset(mkey, dkey, device, n_correct, n_incorrect, seed,
                      vit_method="gradcam"):
    preds_csv = PHASE6_DIR / f"{mkey}__{dkey}_preds.csv"
    manifest = MANIFESTS[dkey]
    if not preds_csv.is_file() or not manifest.is_file() or not CKPTS[mkey].is_file():
        return []
    preds = pd.read_csv(preds_csv)
    preds["id_code"] = preds["id_code"].astype(str)
    man = pd.read_csv(manifest)
    man["id_code"] = man["id_code"].astype(str)
    if "split" in man.columns:
        want = "test" if dkey == "APTOS-internal" else "external"
        man = man[man["split"] == want]
    pathmap = dict(zip(man["id_code"], man["processed_path"].astype(str)))

    picks = sample_ids(preds, n_correct, n_incorrect, seed)
    model, _ = load_checkpoint(CKPTS[mkey], device)
    is_vit = "retfound" in mkey
    tl = None if is_vit else cnn_target_layer(model)

    rows = []
    for _, r in picks.iterrows():
        pid = r["id_code"]
        rel = pathmap.get(pid)
        if rel is None:
            continue
        p = Path(rel)
        p = p if p.is_absolute() else PROJECT_DIR / p
        if not p.is_file():
            continue
        pil = load_processed(str(p))
        rgb01 = denorm(pil)
        x = to_tensor(pil).to(device)

        with torch.no_grad():
            pred_cls = int(model(x).argmax(1))
        if is_vit:
            if vit_method == "rollout":
                cam, method = vit_attention_rollout(model, x)
            else:
                cam, method = vit_gradcam(model, x, pred_cls)
        else:
            cam, method = gradcampp(model, x, tl, pred_cls), "gradcampp"
        cam224 = np.asarray(F.interpolate(
            torch.tensor(cam)[None, None].float(), size=(224, 224),
            mode="bilinear", align_corners=False)[0, 0])
        cam224 = np.clip(cam224, 0, 1)
        pr = plausibility(cam224, rgb01)

        tag = (f"{pid}_g{int(r['y_true'])}_p{pred_cls}_"
               f"{'ok' if int(r['y_true']) == pred_cls else 'err'}")
        overlay(rgb01, cam224,
                OUT_DIR / "maps" / f"{mkey}__{dkey}" / f"{tag}.png",
                f"{mkey} | {dkey} | true {GRADE_NAMES[int(r['y_true'])]} | "
                f"pred {GRADE_NAMES[pred_cls]} | {method}", pr)

        rows.append({
            "model": mkey, "dataset": dkey, "id_code": pid,
            "true_grade": int(r["y_true"]), "pred_grade": pred_cls,
            "correct": bool(int(r["y_true"]) == pred_cls),
            "saliency_method": method, **pr,
        })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(CKPTS))
    ap.add_argument("--datasets", nargs="+", default=["Messidor-2", "IDRiD"])
    ap.add_argument("--n-correct", type=int, default=12)
    ap.add_argument("--n-incorrect", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--vit-method", default="gradcam", choices=["gradcam", "rollout"],
                    help="ViT saliency: 'gradcam' (method-matched to the CNN, "
                         "primary) or 'rollout' (attention rollout, supplementary)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"Phase 7: device={device}  models={args.models}  "
          f"datasets={args.datasets}  vit_method={args.vit_method}")

    all_rows = []
    for mkey in args.models:
        for dkey in args.datasets:
            print(f"  {mkey} x {dkey} ...")
            all_rows += run_model_dataset(mkey, dkey, device, args.n_correct,
                                          args.n_incorrect, args.seed,
                                          vit_method=args.vit_method)

    if not all_rows:
        print("nothing produced - check checkpoints / manifests")
        return 1
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "saliency_per_image.csv", index=False)

    rate = (df.groupby(["model", "dataset", "correct"])
              .agg(n=("proxy_plausible", "size"),
                   proxy_plausible_frac=("proxy_plausible", "mean"),
                   mean_topmass_lesionzone=("topmass_frac_lesionzone", "mean"),
                   mean_topmass_disc=("topmass_frac_disc", "mean"),
                   mean_topmass_border=("topmass_frac_border", "mean"))
              .reset_index())
    rate.to_csv(OUT_DIR / "plausibility_rates.csv", index=False)

    _write_summary(df, rate)
    _write_methods_paragraph()
    _selected_figure(df)

    print(f"\nwrote {OUT_DIR}/saliency_per_image.csv  ({len(df)} maps)")
    print(f"wrote {OUT_DIR}/plausibility_rates.csv")
    print(f"wrote {OUT_DIR}/phase7_summary.md")
    print(f"wrote {OUT_DIR}/fig_explainability.png")
    print(rate.to_string(index=False))
    return 0


def _write_summary(df, rate):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L = ["# Phase 7 - Explainability (automated plausibility proxy)", "",
         f"_Generated {ts}. {len(df)} saliency maps; "
         "CNN = Grad-CAM++, ViT = attention rollout (fallback noted per image)._",
         "",
         "Proxy-plausible = peak inside fundus, outside the estimated optic-disc "
         "disk and outside the outer border annulus, with >=50% of top-decile "
         "saliency mass in that 'lesion zone'. Automated screen, not expert "
         "rating (see Limitations).", "",
         "| Model | Dataset | Correct? | n | proxy-plausible % | top-mass lesion-zone | in disc | in border |",
         "|---|---|:--:|--:|--:|--:|--:|--:|"]
    for r in rate.itertuples():
        L.append(f"| {r.model} | {r.dataset} | {'yes' if r.correct else 'no'} | "
                 f"{r.n} | {100 * r.proxy_plausible_frac:.0f}% | "
                 f"{r.mean_topmass_lesionzone:.2f} | {r.mean_topmass_disc:.2f} | "
                 f"{r.mean_topmass_border:.2f} |")
    L += ["", f"Saliency method usage: {df['saliency_method'].value_counts().to_dict()}",
          "", "Per-image detail: `outputs/phase7/saliency_per_image.csv`; "
          "overlays: `outputs/phase7/maps/<model>__<dataset>/`.", ""]
    (OUT_DIR / "phase7_summary.md").write_text("\n".join(L), encoding="utf-8")


def _write_methods_paragraph():
    txt = """\
### Explainability (drop-in Methods text)

For a grade-stratified sample of correctly and incorrectly classified external
images (image identifiers drawn from the Phase 6 prediction files with a fixed
seed), a saliency map was generated for the model's predicted class. For the
convolutional networks this was a Grad-CAM++ map computed at the last
convolutional block; for the Vision Transformer it was a method-matched Grad-CAM
map computed at the last transformer block (gradient-weighted patch-token
activations for the predicted class, class token removed, reshaped to the patch
grid and bilinearly upsampled). Attention-rollout maps are additionally provided
in the Supplement but were more diffuse and border-biased. Because the public
IDRiD mirror used here
provides no lesion segmentation masks and a second expert grader was not
available, clinical plausibility was scored with a pre-registered automated proxy
rather than by hand: a map was counted "proxy-plausible" when its peak lay within
the fundus, outside an automatically estimated optic-disc disk, and outside the
outer border annulus, and when at least half of the top-decile saliency mass fell
in that same lesion-eligible region. The proxy cannot exclude vessel-centred
activation and is reported as a screen; the absence of blinded expert rating is
stated as a limitation. Code: scripts/phase7_explainability.py.
"""
    (OUT_DIR / "explainability_methods_paragraph.md").write_text(txt, encoding="utf-8")


def _selected_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # 6 panels: {EfficientNet-B3, RETFound-full} x {Messidor-2, IDRiD} plus one
    # extra correct/incorrect pair; within each cell prefer a higher-grade
    # (lesion-bearing) example and pair a correct with an incorrect case.
    combos = [("baseline_efficientnet_b3", "Messidor-2", True),
              ("baseline_efficientnet_b3", "Messidor-2", False),
              ("baseline_efficientnet_b3", "IDRiD", True),
              ("retfound_full", "Messidor-2", True),
              ("retfound_full", "Messidor-2", False),
              ("retfound_full", "IDRiD", False)]
    want = []
    for mkey, dkey, corr in combos:
        sub = df[(df.model == mkey) & (df.dataset == dkey) & (df.correct == corr)]
        if sub.empty:
            sub = df[(df.model == mkey) & (df.dataset == dkey)]
        if sub.empty:
            continue
        hi = sub[sub.true_grade >= 2]
        want.append((hi if len(hi) else sub).iloc[0])
    if not want:
        return
    n = len(want)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axgrid = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.6 * nrow),
                               squeeze=False)
    axes = [ax for r in axgrid for ax in r]
    for ax, row in zip(axes, want):
        png = (OUT_DIR / "maps" / f"{row.model}__{row.dataset}"
               / f"{row.id_code}_g{row.true_grade}_p{row.pred_grade}_"
                 f"{'ok' if row.correct else 'err'}.png")
        if png.is_file():
            ax.imshow(mpimg.imread(png))
        mname = ("EfficientNet-B3" if "efficientnet" in row.model
                 else "ResNet-50" if "resnet" in row.model
                 else "RETFound-full" if row.model.endswith("full")
                 else "RETFound-lin")
        ax.set_title(f"{mname} / {row.dataset}\n"
                     f"true {GRADE_NAMES[row.true_grade]} $\\rightarrow$ pred "
                     f"{GRADE_NAMES[row.pred_grade]}  "
                     f"({'correct' if row.correct else 'error'})\n"
                     f"{'proxy-plausible' if row.proxy_plausible else 'proxy-implausible'}"
                     f"  [{row.saliency_method}]",
                     fontsize=7)
        ax.axis("off")
    for ax in axes[len(want):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_explainability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
