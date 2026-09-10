"""
Phase 7 - blinded ophthalmologist saliency-rating packet (the "expert rating
hook" that Phase 7 leaves as future work).

Phase 7 scored saliency-map plausibility with an *automated proxy* because this
public IDRiD mirror ships no lesion masks and no second grader was available.
This script builds everything needed for a proper blinded human rating and then
ingests the result, so the only outstanding step is a clinician sitting down with
the packet:

  build   (default)  Sample a manageable, grade-stratified subset of the Phase 7
                      saliency maps, RE-RENDER each as a clean two-panel image
                      (fundus | fundus+heatmap) with NO optic-disc circle, NO
                      peak marker and NO identifying text, assign a shuffled
                      blind id, and write:
                        rating_packet/img/<blind_id>.png     one map per file
                        rating_packet/rating_sheet_BLANK.csv  form for the rater
                        rating_packet/INSTRUCTIONS.md          rubric
                        rating_packet/KEY.csv                  blind_id -> truth
                                                               (WITHHOLD from rater)
                        rating_packet/manifest.csv             blind_id -> img path

  score   --score sheet1.csv [sheet2.csv ...]
                      Join one or more completed rating sheets to KEY.csv and
                      report, per model x dataset x correctness: expert
                      focus-plausible rate, agreement with the automated proxy
                      (raw % and Cohen's kappa), and - with >=2 raters -
                      inter-rater Cohen's kappa. Writes:
                        outputs/phase7/rating_agreement.csv
                        outputs/phase7/phase7_rating_summary.md

  build --demo-score  After building, fabricate a SYNTHETIC rating sheet
                      (proxy verdict flipped with prob 0.15; clearly labelled
                      DEMO) and run the scorer on it, so the whole pipeline is
                      exercised end-to-end before a real rater is available.
                      DEMO outputs are suffixed _DEMO and are NOT the real
                      analysis.

Run:
  .venv/Scripts/python.exe scripts/phase7_rating_packet.py --per-cell 10
  .venv/Scripts/python.exe scripts/phase7_rating_packet.py --per-cell 10 --demo-score
  .venv/Scripts/python.exe scripts/phase7_rating_packet.py --score outputs/phase7/rating_packet/rater_AB.csv
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
from phase7_explainability import (  # noqa: E402
    CKPTS, MANIFESTS, PROJECT_DIR,
    cnn_target_layer, denorm, gradcampp, load_processed, to_tensor,
    vit_attention_rollout, vit_gradcam,
)
from predict import load_checkpoint, pick_device  # noqa: E402

OUT_DIR = PROJECT_DIR / "outputs" / "phase7"
PACKET_DIR = OUT_DIR / "rating_packet"
SALIENCY_CSV = OUT_DIR / "saliency_per_image.csv"

MODELS = ["baseline_efficientnet_b3", "baseline_resnet50",
          "retfound_linprobe", "retfound_full"]
DATASETS = ["Messidor-2", "IDRiD"]


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def sample_subset(sal: pd.DataFrame, per_cell: int, seed: int) -> pd.DataFrame:
    """`per_cell` maps per (model, dataset, correct), grade-stratified where the
    Phase 7 sample allows it. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    picks = []
    for model in MODELS:
        for ds in DATASETS:
            for corr in (True, False):
                pool = sal[(sal.model == model) & (sal.dataset == ds)
                           & (sal.correct == corr)]
                if pool.empty:
                    continue
                grades = sorted(pool.true_grade.unique())
                per_g = max(1, per_cell // max(len(grades), 1))
                chosen = []
                for g in grades:
                    sub = pool[pool.true_grade == g]
                    take = min(len(sub), per_g)
                    chosen.append(sub.sample(take, random_state=int(
                        rng.integers(1_000_000_000))))
                got = pd.concat(chosen) if chosen else pool.head(0)
                if len(got) < per_cell:
                    extra = pool.drop(got.index)
                    if len(extra):
                        got = pd.concat([got, extra.sample(
                            min(len(extra), per_cell - len(got)),
                            random_state=int(rng.integers(1_000_000_000)))])
                picks.append(got.head(per_cell))
    out = pd.concat(picks).reset_index(drop=True)
    # shuffle so blind ids carry no ordering information
    return out.sample(frac=1.0, random_state=int(rng.integers(1_000_000_000))
                      ).reset_index(drop=True)


def resolve_pathmap(dkey: str) -> dict:
    man = pd.read_csv(MANIFESTS[dkey])
    man["id_code"] = man["id_code"].astype(str)
    if "split" in man.columns:
        man = man[man["split"] == ("test" if dkey == "APTOS-internal" else "external")]
    return dict(zip(man["id_code"], man["processed_path"].astype(str)))


def clean_overlay(rgb01: np.ndarray, cam224: np.ndarray, out_png: Path) -> None:
    """Two bare panels, no annotations, no text - what the rater sees."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(6.0, 3.1))
    ax[0].imshow(rgb01)
    ax[1].imshow(rgb01)
    ax[1].imshow(cam224, cmap="jet", alpha=0.45)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.03)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def saliency_for(model, mkey: str, x: torch.Tensor, tl, vit_method: str):
    is_vit = "retfound" in mkey
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
    return np.clip(cam224, 0, 1), method, pred_cls


def build(per_cell: int, seed: int, device_pref: str, vit_method: str) -> pd.DataFrame:
    if not SALIENCY_CSV.is_file():
        raise SystemExit(f"missing {SALIENCY_CSV} - run scripts/phase7_explainability.py first")
    sal = pd.read_csv(SALIENCY_CSV)
    sal["id_code"] = sal["id_code"].astype(str)
    sal = sal[sal.dataset.isin(DATASETS)].reset_index(drop=True)
    picks = sample_subset(sal, per_cell, seed)
    picks["blind_id"] = [f"R{i:04d}" for i in range(1, len(picks) + 1)]

    device = pick_device(device_pref)
    print(f"Phase 7 rating packet: {len(picks)} maps  device={device}  "
          f"vit_method={vit_method}")

    (PACKET_DIR / "img").mkdir(parents=True, exist_ok=True)
    pathmaps = {d: resolve_pathmap(d) for d in DATASETS}

    key_rows = []
    n = 0
    for mkey in MODELS:
        rows_m = picks[picks.model == mkey]
        if rows_m.empty:
            continue
        model, _ = load_checkpoint(CKPTS[mkey], device)
        tl = None if "retfound" in mkey else cnn_target_layer(model)
        for r in rows_m.itertuples():
            rel = pathmaps[r.dataset].get(r.id_code)
            if rel is None:
                continue
            p = Path(rel)
            p = p if p.is_absolute() else PROJECT_DIR / p
            if not p.is_file():
                continue
            pil = load_processed(str(p))
            rgb01 = denorm(pil)
            x = to_tensor(pil).to(device)
            cam224, method, pred_cls = saliency_for(model, mkey, x, tl, vit_method)

            clean_overlay(rgb01, cam224, PACKET_DIR / "img" / f"{r.blind_id}.png")
            key_rows.append({
                "blind_id": r.blind_id, "model": mkey, "dataset": r.dataset,
                "id_code": r.id_code, "true_grade": int(r.true_grade),
                "pred_grade": int(pred_cls),
                "correct": bool(int(r.true_grade) == pred_cls),
                "saliency_method": method,
                "proxy_plausible": bool(r.proxy_plausible),
                "proxy_topmass_lesionzone": float(r.topmass_frac_lesionzone),
                "proxy_topmass_disc": float(r.topmass_frac_disc),
                "proxy_topmass_border": float(r.topmass_frac_border),
            })
            n += 1
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    key = pd.DataFrame(key_rows).sort_values("blind_id").reset_index(drop=True)
    key.to_csv(PACKET_DIR / "KEY.csv", index=False)

    pd.DataFrame({
        "blind_id": key["blind_id"],
        "focus_plausible": "",   # 1 = saliency on a plausible DR-relevant region; 0 = not
        "lesion_aligned": "",    # 1 = overlaps a visible haemorrhage/exudate/NV lesion; 0/NA
        "confidence": "",        # 1 (guess) .. 5 (certain)
        "notes": "",
    }).to_csv(PACKET_DIR / "rating_sheet_BLANK.csv", index=False)

    key[["blind_id"]].assign(
        image=lambda d: d["blind_id"].map(lambda b: f"img/{b}.png")
    ).to_csv(PACKET_DIR / "manifest.csv", index=False)

    _write_instructions(len(key))
    print(f"wrote {PACKET_DIR}/img/*.png  ({n} maps)")
    print(f"wrote {PACKET_DIR}/rating_sheet_BLANK.csv")
    print(f"wrote {PACKET_DIR}/KEY.csv  (withhold from rater)")
    print(f"wrote {PACKET_DIR}/INSTRUCTIONS.md")
    return key


def _write_instructions(n: int) -> None:
    txt = f"""\
# Blinded saliency-plausibility rating - instructions

You are shown {n} fundus images. Each file `img/R####.png` has two panels:

* **left**  - the pre-processed colour fundus photograph;
* **right** - the same photograph with the model's saliency map (the region the
  model used for its diabetic-retinopathy grade) overlaid as a red/blue heatmap
  (red = most important).

You are **blinded** to the model, the dataset, the reference grade and the
model's prediction. Rate each map in `rating_sheet_BLANK.csv` (save as
`rater_<yourinitials>.csv`):

| column | values | meaning |
|---|---|---|
| `focus_plausible` | 1 / 0 | 1 if the *bulk* of the red saliency mass sits on retinal tissue a grader could plausibly use for DR assessment (lesions, macula, arcades) and is **not** dominated by the optic disc, the image border, or areas outside the retina. 0 otherwise. |
| `lesion_aligned` | 1 / 0 / NA | 1 if the red saliency overlaps a visible DR lesion (haemorrhage, microaneurysm, hard/soft exudate, neovascularisation). 0 if no such lesion is under the saliency. NA if you cannot tell / image ungradable. |
| `confidence` | 1-5 | 1 = pure guess, 5 = certain. |
| `notes` | free text | optional (e.g. "on disc", "vessel only", "border artefact"). |

Please rate every row. Do not confer with other raters. Return the completed
CSV; keep `blind_id` unchanged.
"""
    (PACKET_DIR / "INSTRUCTIONS.md").write_text(txt, encoding="utf-8")


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def _cohen_kappa(a, b) -> float:
    from sklearn.metrics import cohen_kappa_score
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(pd.isna(a) | pd.isna(b))
    if m.sum() < 2 or len(np.unique(np.concatenate([a[m], b[m]]))) < 2:
        return float("nan")
    return float(cohen_kappa_score(a[m].astype(int), b[m].astype(int)))


def _load_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "blind_id" not in df.columns or "focus_plausible" not in df.columns:
        raise SystemExit(f"{path}: needs at least 'blind_id' and 'focus_plausible' columns")
    df["blind_id"] = df["blind_id"].astype(str)
    for c in ("focus_plausible", "lesion_aligned", "confidence"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["rater"] = path.stem
    return df[[c for c in ("rater", "blind_id", "focus_plausible",
                           "lesion_aligned", "confidence", "notes")
              if c in df.columns]]


def score(sheet_paths: list[Path], out_tag: str = "") -> None:
    key_path = PACKET_DIR / "KEY.csv"
    if not key_path.is_file():
        raise SystemExit(f"missing {key_path} - run `build` first")
    key = pd.read_csv(key_path)
    key["blind_id"] = key["blind_id"].astype(str)

    sheets = [_load_sheet(p) for p in sheet_paths]
    long = pd.concat(sheets, ignore_index=True)
    raters = sorted(long["rater"].unique())

    # majority vote across raters (ties -> plausible) for the headline rates
    piv = long.pivot_table(index="blind_id", columns="rater",
                           values="focus_plausible", aggfunc="first")
    consensus = (piv.mean(axis=1) >= 0.5).astype(int).rename("expert_plausible")
    cons = (key.set_index("blind_id").join(consensus, how="inner")
            .dropna(subset=["expert_plausible"]))
    cons["expert_plausible"] = cons["expert_plausible"].astype(int)
    cons["proxy_int"] = cons["proxy_plausible"].astype(int)

    rows = []
    for (model, ds, corr), g in cons.groupby(["model", "dataset", "correct"]):
        rows.append({
            "model": model, "dataset": ds, "correct": bool(corr), "n": len(g),
            "expert_plausible_frac": round(g["expert_plausible"].mean(), 3),
            "proxy_plausible_frac": round(g["proxy_int"].mean(), 3),
            "expert_proxy_agree_frac": round(
                (g["expert_plausible"] == g["proxy_int"]).mean(), 3),
            "expert_proxy_kappa": round(
                _cohen_kappa(g["expert_plausible"], g["proxy_int"]), 3),
        })
    agree = pd.DataFrame(rows).sort_values(["model", "dataset", "correct"])
    agree.to_csv(OUT_DIR / f"rating_agreement{out_tag}.csv", index=False)

    overall_kappa = _cohen_kappa(cons["expert_plausible"], cons["proxy_int"])
    overall_agree = (cons["expert_plausible"] == cons["proxy_int"]).mean()
    inter = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            inter.append((raters[i], raters[j],
                          _cohen_kappa(piv[raters[i]], piv[raters[j]])))

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    L = [f"# Phase 7 - blinded saliency rating"
         f"{' (DEMO / SYNTHETIC)' if out_tag else ''}", "",
         f"_Generated {ts}. Raters: {', '.join(raters)} "
         f"({len(cons)} maps with a usable rating)._", ""]
    if out_tag:
        L += ["> **These numbers come from a fabricated rating sheet** "
              "(proxy verdict flipped with p=0.15), used only to exercise the "
              "scoring pipeline. Replace with real clinician ratings.", ""]
    L += [f"Expert (rater consensus) vs. automated proxy - overall agreement "
          f"**{overall_agree:.1%}**, Cohen's kappa **{overall_kappa:.3f}** "
          f"(n={len(cons)}).", ""]
    if inter:
        L += ["Inter-rater Cohen's kappa:", ""]
        L += [f"* {a} vs {b}: {k:.3f}" for a, b, k in inter] + [""]
    L += ["| Model | Dataset | Correct? | n | expert-plausible % | "
          "proxy-plausible % | agree % | kappa |",
          "|---|---|:--:|--:|--:|--:|--:|--:|"]
    for r in agree.itertuples():
        L.append(f"| {r.model} | {r.dataset} | {'yes' if r.correct else 'no'} | "
                 f"{r.n} | {100 * r.expert_plausible_frac:.0f}% | "
                 f"{100 * r.proxy_plausible_frac:.0f}% | "
                 f"{100 * r.expert_proxy_agree_frac:.0f}% | "
                 f"{r.expert_proxy_kappa:.3f} |")
    L += ["", f"Per-cell table: `outputs/phase7/rating_agreement{out_tag}.csv`; "
          f"key: `outputs/phase7/rating_packet/KEY.csv`.", ""]
    (OUT_DIR / f"phase7_rating_summary{out_tag}.md").write_text(
        "\n".join(L), encoding="utf-8")
    print(f"wrote {OUT_DIR}/rating_agreement{out_tag}.csv")
    print(f"wrote {OUT_DIR}/phase7_rating_summary{out_tag}.md")
    print(f"\noverall expert-vs-proxy agreement {overall_agree:.1%}  "
          f"kappa {overall_kappa:.3f}")
    print(agree.to_string(index=False))


def _demo_sheet(key: pd.DataFrame, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    flip = (rng.random(len(key)) < 0.15).astype(int)
    fp = key["proxy_plausible"].astype(int).to_numpy() ^ flip
    pd.DataFrame({
        "blind_id": key["blind_id"], "focus_plausible": fp,
        "lesion_aligned": "", "confidence": rng.integers(2, 6, len(key)),
        "notes": "SYNTHETIC - demo only",
    }).to_csv(PACKET_DIR / "DEMO_rater_synthetic.csv", index=False)
    p = PACKET_DIR / "DEMO_rater_synthetic.csv"
    print(f"wrote {p}  (fabricated; scoring-pipeline check only)")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", nargs="+", metavar="SHEET.csv",
                    help="score completed rating sheet(s) instead of building")
    ap.add_argument("--per-cell", type=int, default=10,
                    help="maps per (model x dataset x correct/incorrect) cell")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--vit-method", default="gradcam", choices=["gradcam", "rollout"])
    ap.add_argument("--demo-score", action="store_true",
                    help="after building, fabricate a rating sheet and run the "
                         "scorer on it (labelled DEMO) to check the pipeline")
    args = ap.parse_args()

    if args.score:
        score([Path(p) for p in args.score])
        return 0

    key = build(args.per_cell, args.seed, args.device, args.vit_method)
    if args.demo_score:
        print()
        score([_demo_sheet(key, args.seed)], out_tag="_DEMO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
