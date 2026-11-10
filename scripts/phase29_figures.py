"""
Phase 29d - manuscript figures (International Ophthalmology spec):
174 mm wide (double column), Arial-metric sans (Liberation Sans), 8-9 pt text,
no titles inside figures (captions live in the manuscript), colour + marker +
line style so every figure reads in greyscale. Saved as vector PDF (for LaTeX)
and 600-dpi TIFF (for the submission system).

Data: outputs/phase29/* (official IDRiD), outputs/phase26 (lesion masks).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
P29 = ROOT / "outputs" / "phase29"
OUT = P29 / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
W = 174 * MM
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "lines.linewidth": 1.4, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": "#5f5e5a", "axes.labelcolor": "#262624", "xtick.color": "#5f5e5a",
    "ytick.color": "#5f5e5a", "text.color": "#262624", "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
})
MODELS = ["baseline_efficientnet_b3", "retfound_full", "baseline_resnet50", "retfound_linprobe"]
LAB = {"baseline_efficientnet_b3": "EfficientNet-B3", "retfound_full": "RETFound (full fine-tune)",
       "baseline_resnet50": "ResNet-50", "retfound_linprobe": "RETFound (linear probe)"}
COL = {"baseline_efficientnet_b3": "#2a78d6", "retfound_full": "#eb6834",
       "baseline_resnet50": "#1baf7a", "retfound_linprobe": "#eda100"}
MK = {"baseline_efficientnet_b3": "o", "retfound_full": "s", "baseline_resnet50": "^", "retfound_linprobe": "D"}
LS = {"baseline_efficientnet_b3": "-", "retfound_full": "--", "baseline_resnet50": "-.", "retfound_linprobe": ":"}
DS = ["APTOS-internal", "Messidor-2", "IDRiD"]
DSLAB = {"APTOS-internal": "APTOS (internal)", "Messidor-2": "Messidor-2", "IDRiD": "IDRiD"}
INK2 = "#5f5e5a"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def qwk(y, yh, k=5):
    O = np.zeros((k, k)); np.add.at(O, (y, yh), 1)
    Wt = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (Wt * O).sum() / (Wt * E).sum()


def preds(m, d):
    return pd.read_csv(P29 / "preds_official" / f"{m}__{d}_preds.csv")


# ------------------------------------------------------------------ Fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(W, 62 * MM))
    ax.set_xlim(0, 174); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#5f5e5a", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6,
                fontweight="bold" if bold else "normal", linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#5f5e5a", mutation_scale=7))
    box(2, 40, 40, 18, "APTOS 2019\nn = 3,662 (India)\ndevelopment only\n(internal test n = 692)", "#e8f1fb")
    box(2, 21, 40, 15, "Messidor-2\nn = 1,744 (France)\nexternal test only", "#fdeee7")
    box(2, 3, 40, 15, "IDRiD (official release)\nn = 516 (India)\nexternal test only", "#fdeee7")
    box(50, 40, 56, 18, "Train on APTOS; select on val QWK\nEfficientNet-B3 (4 runs), ResNet-50\nRETFound: linear probe (4 runs),\nfull fine-tune (3 runs); controls", "#f3f2ee")
    box(50, 13, 56, 20, "Every trained model scored\nunchanged on all external images\n(no re-training, re-calibration\nor threshold tuning)", "#f3f2ee")
    box(114, 44, 58, 14, "Primary: internal-to-external\nQWK drop, mean over runs,\nvs EfficientNet-B3 (Holm)", "#fff7e0")
    box(114, 27, 58, 14, "Referable DR: sensitivity, specificity,\nPPV, NPV, AUC · case mix and\nno-DR/mild boundary (descriptive)", "#fff7e0")
    box(114, 3, 58, 21, "Uncertainty: error detection and\nreferral vs simple confidence\nSaliency vs expert lesion masks\n(IDRiD Segmentation, n = 81)", "#fff7e0")
    arrow(42, 49, 50, 49); arrow(78, 40, 78, 33); arrow(42, 28, 50, 24); arrow(42, 10, 50, 20)
    arrow(106, 49, 114, 51); arrow(106, 25, 114, 34); arrow(106, 20, 114, 14)
    save(fig, "Fig1")


# ------------------------------------------------------------------ Fig 2
def fig2():
    rng = np.random.default_rng(42)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 70 * MM), gridspec_kw={"width_ratios": [1.15, 1]})
    for j, m in enumerate(MODELS):
        for i, d in enumerate(DS):
            p = preds(m, d); y = p.y_true.to_numpy(); yh = p.y_pred.to_numpy()
            est = qwk(y, yh)
            bs = [qwk(y[s], yh[s]) for s in rng.integers(0, len(y), (1000, len(y)))]
            x = i + (j - 1.5) * 0.17
            a.errorbar(x, est, yerr=[[est - np.percentile(bs, 2.5)], [np.percentile(bs, 97.5) - est]],
                       fmt=MK[m], color=COL[m], ms=4.5, capsize=0, elinewidth=1.0,
                       mec="white", mew=0.5, label=LAB[m] if i == 0 else None)
    a.set_xticks(range(3)); a.set_xticklabels(["APTOS\n(internal)", "Messidor-2", "IDRiD"])
    a.set_ylabel("Quadratic weighted kappa (95% CI)"); a.set_ylim(0.3, 0.95)
    a.yaxis.grid(True); a.set_axisbelow(True)
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.08), handletextpad=0.3, columnspacing=1.2)
    a.text(-0.12, 1.02, "a", transform=a.transAxes, fontweight="bold", fontsize=10)
    # b: difference in drop vs EfficientNet-B3
    cf = pd.read_csv(P29 / "primary" / "confirmatory_family_official.csv")
    sf = pd.read_csv(P29 / "primary" / "secondary_family_official.csv")
    rows = []
    for _, r in cf.iterrows():
        rows.append((r["model"], r["external"], r["delta_drop"], r["ci95_lo"], r["ci95_hi"], r.get("p_holm", np.nan)))
    for _, r in sf.iterrows():
        rows.append(("baseline_resnet50", r["external"], r["delta_drop"], r["ci95_lo"], r["ci95_hi"], np.nan))
    order = [("retfound_full", "Messidor-2"), ("retfound_full", "IDRiD"), ("retfound_linprobe", "Messidor-2"),
             ("retfound_linprobe", "IDRiD"), ("baseline_resnet50", "Messidor-2"), ("baseline_resnet50", "IDRiD")]
    look = {(r[0], r[1]): r for r in rows}
    for k, key in enumerate(order):
        m, d, est, lo, hi, ph = look[key]
        yk = len(order) - 1 - k
        b.errorbar(est, yk, xerr=[[est - lo], [hi - est]], fmt=MK[m], color=COL[m], ms=4.5,
                   elinewidth=1.0, capsize=0, mec="white", mew=0.5)
        lab = f"{LAB[m]}, {d}"
        b.text(-0.135, yk, lab, ha="right", va="center", fontsize=7)
        if not np.isnan(ph):
            b.text(0.235, yk, f"Holm p = {ph:.3f}" if ph >= 0.001 else "Holm p < 0.001", ha="left", va="center", fontsize=6.5, color=INK2)
    b.axvline(0, color="#5f5e5a", lw=0.7)
    b.set_xlim(-0.12, 0.23); b.set_yticks([]); b.spines["left"].set_visible(False)
    b.set_xlabel("Excess QWK drop vs EfficientNet-B3 (95% CI)\n← smaller drop        larger drop →")
    b.xaxis.grid(True); b.set_axisbelow(True)
    b.text(-0.95, 1.02, "b", transform=b.transAxes, fontweight="bold", fontsize=10)
    fig.subplots_adjust(wspace=1.05)
    save(fig, "Fig2")


# ------------------------------------------------------------------ Fig 3
def fig3():
    dec = pd.read_csv(P29 / "decomposition" / "shift_decomposition.csv")
    fig = plt.figure(figsize=(W, 78 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.32)
    a = fig.add_subplot(gs[0])
    metrics = [("qwk_raw", "Observed"), ("qwk_reweighted", "Case mix\nmatched"),
               ("qwk_4level", "Grades 0+1\nmerged"), ("qwk_4level_reweighted", "Both")]
    ms = ["baseline_efficientnet_b3", "retfound_full"]
    xs = np.arange(len(metrics))
    for e_i, ext in enumerate(["Messidor-2", "IDRiD"]):
        for j, m in enumerate(ms):
            sub = dec[(dec.model == m) & (dec.external == ext)].set_index("metric")
            v = [sub.loc[k, "drop"] for k, _ in metrics]
            lo = [sub.loc[k, "drop"] - sub.loc[k, "drop_lo"] for k, _ in metrics]
            hi = [sub.loc[k, "drop_hi"] - sub.loc[k, "drop"] for k, _ in metrics]
            off = (e_i * 2 + j - 1.5) * 0.19
            a.errorbar(xs + off, v, yerr=[lo, hi], fmt=MK[m], color=COL[m], ms=4.2,
                       mfc=COL[m] if ext == "Messidor-2" else "white", mec=COL[m], mew=1.0,
                       elinewidth=0.9, capsize=0, ls="none",
                       label=f"{LAB[m]}, {ext}")
    a.axhline(0, color="#5f5e5a", lw=0.7)
    a.set_xticks(xs); a.set_xticklabels([l for _, l in metrics])
    a.set_ylabel("Internal-to-external QWK drop (95% CI)")
    a.yaxis.grid(True); a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=6.5, loc="upper right", handletextpad=0.3)
    a.text(-0.13, 1.02, "a", transform=a.transAxes, fontweight="bold", fontsize=10)
    # b: fate of reference-standard grade 0 / 1 / 2 images (EffB3), stacked, Messidor-2 vs IDRiD
    b = fig.add_subplot(gs[1])
    shades = ["#e8f1fb", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
    hatches = ["", "....", "", "////", ""]
    bars = []
    for ext in ["Messidor-2", "IDRiD"]:
        p = preds("baseline_efficientnet_b3", ext)
        for g in [0, 1, 2]:
            c = np.bincount(p.y_pred[p.y_true == g], minlength=5)
            bars.append((f"{ext}\ngrade {g}", c / c.sum(), int(c.sum())))
    y = np.arange(len(bars))[::-1]
    for yi, (lab, frac, n) in zip(y, bars):
        left = 0
        for k in range(5):
            b.barh(yi, frac[k], left=left, color=shades[k], edgecolor="white", lw=0.8,
                   hatch=hatches[k], height=0.72, label=f"predicted {k}" if yi == y[0] else None)
            left += frac[k]
        b.text(1.02, yi, f"n={n}", va="center", fontsize=6.5, color=INK2)
    b.set_yticks(y); b.set_yticklabels([l for l, _, _ in bars], fontsize=7)
    b.set_xlim(0, 1); b.set_xlabel("Share of images (EfficientNet-B3)")
    b.legend(frameon=False, ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.2),
             handlelength=1.2, columnspacing=0.8)
    b.text(-0.42, 1.02, "b", transform=b.transAxes, fontweight="bold", fontsize=10)
    save(fig, "Fig3")


# ------------------------------------------------------------------ Fig 4
def fig4():
    sel = pd.read_csv(P29 / "uncertainty" / "selective_prediction.csv")
    det = pd.read_csv(P29 / "uncertainty" / "error_detection.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W, 62 * MM), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for k, ext in enumerate(["Messidor-2", "IDRiD"]):
        ax = axs[k]
        for m in MODELS:
            s = sel[(sel.model == m) & (sel.dataset == ext) & (sel.score == "msp")].sort_values("referral_rate")
            ax.plot(s.referral_rate * 100, s.missed_referable_frac * 100, LS[m], color=COL[m],
                    marker=MK[m], ms=3.2, mec="white", mew=0.4, label=LAB[m])
        r = sel[(sel.model == "baseline_efficientnet_b3") & (sel.dataset == ext) & (sel.score == "random")].sort_values("referral_rate")
        ax.plot(r.referral_rate * 100, r.missed_referable_frac * 100, color="#a8a79f", lw=1.0,
                ls=(0, (1, 1.5)), label="Random referral (EfficientNet-B3)")
        ax.set_xlabel("Cases referred to a grader (%)")
        if k == 0:
            ax.set_ylabel("Referable DR missed by the system (%)")
        ax.set_ylim(0, 70); ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.text(0.5, 0.97, ext, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK2)
        ax.text(-0.2, 1.02, "ab"[k], transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.2), fontsize=6.5, handlelength=2.6, columnspacing=1.0)
    c = axs[2]
    g = det[det.error_type == "grade"]
    for j, m in enumerate(MODELS):
        for i, d in enumerate(DS):
            for sc, fill in [("msp", True), ("entropy", False)]:
                r = g[(g.model == m) & (g.dataset == d) & (g.score == sc)].iloc[0]
                x = i + (j - 1.5) * 0.19 + (0.05 if sc == "entropy" else -0.05)
                c.errorbar(x, r.auroc, yerr=[[r.auroc - r.auroc_lo], [r.auroc_hi - r.auroc]], fmt=MK[m],
                           color=COL[m], mfc=COL[m] if fill else "white", mec=COL[m], ms=3.6, mew=0.9,
                           elinewidth=0.8, capsize=0)
    c.axhline(0.5, color="#5f5e5a", lw=0.7, ls=(0, (2, 2)))
    c.set_xticks(range(3)); c.set_xticklabels(["APTOS\n(internal)", "Messidor-2", "IDRiD"])
    c.set_ylabel("Error-detection AUROC (95% CI)"); c.set_ylim(0.35, 0.95)
    c.yaxis.grid(True); c.set_axisbelow(True)
    from matplotlib.lines import Line2D
    c.legend(handles=[Line2D([], [], marker="o", color="#5f5e5a", ls="none", ms=4, label="Max-probability (filled)"),
                      Line2D([], [], marker="o", color="#5f5e5a", mfc="white", ls="none", ms=4, label="Entropy (open)")],
             frameon=False, fontsize=6.3, loc="upper right")
    c.text(-0.2, 1.02, "c", transform=c.transAxes, fontweight="bold", fontsize=10)
    fig.subplots_adjust(wspace=0.38, bottom=0.2)
    save(fig, "Fig4")


# ------------------------------------------------------------------ Fig 5
def fig5():
    sys.path.insert(0, str(ROOT / "scripts"))
    import phase26_lesion_localization as p26
    z = np.load(ROOT / "outputs" / "phase26" / "cams.npz", allow_pickle=True)
    cams = z["cams"].item(); ids = list(z["ids"])
    items = p26.find_items(ROOT / "A. Segmentation")
    per = pd.read_csv(ROOT / "outputs" / "phase26" / "localization_per_image.csv")
    # examples: median-lesion-burden images, chosen by rule (25th/50th/75th percentile of lesion pixels)
    burden = per[per.model == "centre"].set_index("image").n_lesion_px.sort_values()
    picks = [burden.index[int(q * (len(burden) - 1))] for q in (0.25, 0.5, 0.75)]
    FH = 168 * MM
    fig = plt.figure(figsize=(W, FH))
    img = 36 * MM; gap = 1.2 * MM
    gx = (W - 3 * img - 2 * gap) / 2 + 4 * MM          # centred block (small offset for row labels)
    top_y = FH - 7 * MM                                  # room for column titles
    for r, iid in enumerate(picks):
        ip, mp = items[iid]
        with Image.open(ip) as im:
            arr = np.asarray(im.convert("RGB"))
        box = p26.bbox_of(arr)
        rgb = np.asarray(Image.fromarray(p26.to_square(arr, box)).resize((224, 224), Image.BILINEAR))
        L = np.zeros((224, 224), bool)
        for k in ["MA", "HE", "EX", "SE"]:
            if k in mp:
                L |= p26.mask_to_224(p26.load_mask(mp[k], arr.shape[:2]), box)
        i = ids.index(iid)
        panels = [("Expert lesion masks", None), ("EfficientNet-B3", cams["baseline_efficientnet_b3"][i]),
                  ("RETFound (full fine-tune)", cams["retfound_full"][i])]
        for c, (t, cam) in enumerate(panels):
            x0 = gx + c * (img + gap); y0 = top_y - (r + 1) * img - r * gap
            ax = fig.add_axes([x0 / W, y0 / FH, img / W, img / FH]); ax.imshow(rgb)
            if cam is not None:
                ax.imshow(cam, cmap="magma", alpha=0.5, vmin=0, vmax=1)
            ax.contour(L, levels=[0.5], colors="#00e5ff" if cam is None else "white", linewidths=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(t, fontsize=7.5, pad=3)
            if c == 0:
                ax.set_ylabel(iid.replace("_", " "), fontsize=7)
                if r == 0:
                    ax.text(-0.22, 1.06, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)
    s1 = pd.read_csv(ROOT / "outputs" / "phase26" / "localization_summary.csv")
    s2 = pd.read_csv(ROOT / "outputs" / "phase26" / "resolution_matched_summary.csv")
    rows = [("Centre prior (control)", s1[(s1.model == "centre")].iloc[0], "#a8a79f", "X", "native")]
    for m in MODELS:
        rows.append((LAB[m], s1[(s1.model == m) & (s1.saliency == "model")].iloc[0], COL[m], MK[m], "native"))
    for m in ["retfound_full", "retfound_linprobe"]:
        rows.append((LAB[m] + ", 7×7 grid", s2[(s2.model == m) & (s2.resolution == "7x7")].iloc[0], COL[m], MK[m], "7x7"))
    lab_w = 46 * MM; pw = (W - lab_w - 10 * MM) / 2; ph = 31 * MM; py = 10 * MM
    for k, (metric, xl, ref) in enumerate([("auroc_lesion", "Pixel-level lesion AUC (95% CI)", 0.5),
                                           ("pointing_hit", "Saliency peak on a lesion (95% CI)", None)]):
        ax = fig.add_axes([(lab_w + k * (pw + 8 * MM)) / W, py / FH, pw / W, ph / FH])
        for yi, (lab, r, col, mk, res) in enumerate(rows[::-1]):
            ax.errorbar(r[metric], yi, xerr=[[r[metric] - r[metric + "_lo"]], [r[metric + "_hi"] - r[metric]]],
                        fmt=mk, color=col, mfc=col if res == "native" else "white", mec=col, ms=4.2, mew=1.0,
                        elinewidth=0.9, capsize=0)
        if ref is not None:
            ax.axvline(ref, color="#5f5e5a", lw=0.7, ls=(0, (2, 2)))
            ax.text(ref, len(rows) - 0.4, " chance", fontsize=6, color=INK2, va="bottom", ha="left")
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([l for l, *_ in rows[::-1]] if k == 0 else [], fontsize=7)
        ax.set_xlabel(xl); ax.xaxis.grid(True); ax.set_axisbelow(True)
        ax.text(-0.04 if k else -0.86, 1.06, "bc"[k], transform=ax.transAxes, fontweight="bold", fontsize=10)
    save(fig, "Fig5")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5}[w]()
        print("Fig", w, "done", flush=True)
