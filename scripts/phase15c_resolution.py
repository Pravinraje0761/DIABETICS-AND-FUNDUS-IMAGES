"""
Phase 15 Variant C -- input-resolution sensitivity (224 vs 384).

The committed pipeline resizes every fundus image to 224 px (RETFound's native
size). This runner re-preprocesses APTOS-internal test + Messidor-2 + IDRiD
*from their source images* at 384 px (upscaling a 224 PNG would add no
information) and re-runs external inference only -- no retraining -- for the
primary model pair:

  baseline_efficientnet_b3 : conv net, size-agnostic; the committed
                             efficientnet_b3_best.pt is run unchanged at 384.
  retfound_full            : ViT-L/16 trained at 224. Built with img_size=384 so
                             timm enlarges the positional-embedding grid to
                             24x24, then the 224 checkpoint's 14x14 pos_embed is
                             bicubically interpolated onto it
                             (retfound._interpolate_pos_embed). The transformer
                             blocks load unchanged. This is an OFF-DISTRIBUTION
                             test of the trained model, not a fair 384-trained
                             comparison -- stated as such in the manuscript.

For each (model, dataset, resolution) it writes Phase-6-schema predictions +
metrics, then the internal->external QWK drop per resolution and its change
versus the 224 baseline (negative delta => 384 *recovers* generalisation).
The 224 pass is recomputed here and asserted to match outputs/phase6/ (<5e-4)
so the two resolutions share one code path.

Run (preprocess is automatic if data_splits/*_manifest_res384.csv is absent):
  .venv/Scripts/python.exe scripts/phase15c_resolution.py --device cuda
  .venv/Scripts/python.exe scripts/phase15c_resolution.py --preprocess-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import build_dataloader  # noqa: E402
from metrics import compute_metrics, format_metrics  # noqa: E402
from predict import infer, load_checkpoint, pick_device, save_predictions  # noqa: E402
from preprocessing import preprocess_file  # noqa: E402
from retfound import _interpolate_pos_embed, build_retfound_backbone  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_DIR / "data_splits"
OUT_DIR = PROJECT_DIR / "outputs" / "phase15c"
RES_IMG_ROOT = PROJECT_DIR / "data_processed_res384"
HI_RES = 384

CKPTS = {
    "baseline_efficientnet_b3": PROJECT_DIR / "outputs/phase4_baseline/efficientnet_b3_best.pt",
    "retfound_full": PROJECT_DIR / "outputs/phase5_retfound/full_best.pt",
}
# dataset key -> (committed manifest, split, res384 subdir)
DATASETS = {
    "APTOS-internal": (SPLIT_DIR / "aptos_manifest.csv", "test", "aptos"),
    "Messidor-2": (SPLIT_DIR / "messidor2_manifest.csv", "external", "messidor2"),
    "IDRiD": (SPLIT_DIR / "idrid_manifest.csv", "external", "idrid"),
}
PHASE6_DIR = PROJECT_DIR / "outputs" / "phase6"


def res384_manifest(dkey: str) -> Path:
    return SPLIT_DIR / f"{DATASETS[dkey][2]}_manifest_res384.csv"


# --------------------------------------------------------------------------- #
def preprocess_all(force: bool = False) -> None:
    RES_IMG_ROOT.mkdir(parents=True, exist_ok=True)
    for dkey, (manifest, split, sub) in DATASETS.items():
        out_manifest = res384_manifest(dkey)
        if out_manifest.is_file() and not force:
            print(f"{dkey}: {out_manifest.name} exists, skip (use --force-preprocess)")
            continue
        df = pd.read_csv(manifest)
        df = df[df["split"] == split].reset_index(drop=True)
        dst_dir = RES_IMG_ROOT / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        rel_paths, done, missing = [], 0, 0
        for _, row in df.iterrows():
            src = (PROJECT_DIR / str(row["source_path"])).resolve()
            stem = Path(str(row["processed_path"])).stem or str(row["id_code"])
            dst = dst_dir / f"{stem}.png"
            rel_paths.append(str(dst.relative_to(PROJECT_DIR)).replace("\\", "/"))
            if dst.is_file() and not force:
                done += 1
                continue
            if not src.is_file():
                missing += 1
                continue
            preprocess_file(src, dst, size=HI_RES)
            done += 1
        df = df.copy()
        df["processed_path"] = rel_paths
        df.to_csv(out_manifest, index=False)
        print(f"{dkey}: {done}/{len(df)} images at {HI_RES}px -> {dst_dir}  "
              f"(missing sources: {missing});  wrote {out_manifest.name}")


# --------------------------------------------------------------------------- #
def referable_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = (np.asarray(y_true) >= 2).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, prob[:, 2:].sum(axis=1)))


def load_retfound_at(img_size: int, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = dict(ckpt["model_state"])
    model = build_retfound_backbone(
        num_classes=ckpt.get("num_classes", 5), img_size=img_size)
    _interpolate_pos_embed(model, state)
    res = model.load_state_dict(state, strict=False)
    block_missing = [k for k in res.missing_keys if k.startswith("blocks.")]
    if block_missing:
        raise RuntimeError(f"blocks did not load at {img_size}px: {block_missing[:5]}")
    model.to(device).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, img_size, img_size, device=device))
    assert tuple(out.shape) == (1, 5), out.shape
    print(f"  retfound @ {img_size}px: pos_embed {tuple(model.pos_embed.shape)}, "
          f"forward OK, missing={sorted(set(res.missing_keys))}")
    return model


def phase6_qwk(mkey: str, dkey: str) -> float | None:
    p = PHASE6_DIR / f"{mkey}__{dkey}_metrics.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())["quadratic_weighted_kappa"]


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(CKPTS), choices=list(CKPTS))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--preprocess-only", action="store_true")
    ap.add_argument("--force-preprocess", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preprocess_all(force=args.force_preprocess)
    if args.preprocess_only:
        return 0

    device = pick_device(args.device)
    rows: list[dict] = []
    tol = 5e-4

    for mkey in args.models:
        ckpt = CKPTS[mkey]
        if not ckpt.is_file():
            print(f"skip {mkey}: {ckpt} missing")
            continue
        for res in (224, 384):
            if mkey == "retfound_full":
                model = load_retfound_at(res, ckpt, device)
            else:
                model, _ = load_checkpoint(ckpt, device)  # conv net, size-agnostic
            for dkey, (manifest, split, _sub) in DATASETS.items():
                if res == 224:
                    mf, sp, size = manifest, split, 224
                else:
                    mf, sp, size = res384_manifest(dkey), split, HI_RES
                if not Path(mf).is_file():
                    print(f"  {dkey} @ {res}: manifest missing, skip")
                    continue
                loader = build_dataloader(mf, split=sp, batch_size=args.batch_size,
                                          train=False, num_workers=args.num_workers,
                                          shuffle=False, size=size)
                r = infer(model, loader, device)
                prefix = OUT_DIR / f"{mkey}__{dkey}__res{res}"
                save_predictions(r, prefix)
                m = compute_metrics(r["y_true"], r["y_pred"], r["prob"])
                qwk = m["quadratic_weighted_kappa"]
                if res == 224:
                    ref = phase6_qwk(mkey, dkey)
                    if ref is not None and abs(ref - qwk) > tol:
                        raise AssertionError(
                            f"{mkey} x {dkey} @224 QWK {qwk:.5f} != phase6 {ref:.5f}")
                rows.append({
                    "model": mkey, "dataset": dkey, "resolution": res, "n": m["n"],
                    "accuracy": round(m["accuracy"], 4), "qwk": round(qwk, 4),
                    "macro_f1": round(m["macro_f1"], 4),
                    "referable_auc": round(referable_auc(r["y_true"], r["prob"]), 4),
                })
                print(format_metrics(m, title=f"{mkey} x {dkey} @ {res}px"))
                print()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    master = pd.DataFrame(rows)
    master.to_csv(OUT_DIR / "resolution_master.csv", index=False)

    drop_rows: list[dict] = []
    for mkey in master["model"].unique():
        sub = master[master["model"] == mkey]
        for res in sorted(sub["resolution"].unique()):
            s = sub[sub["resolution"] == res].set_index("dataset")["qwk"]
            if "APTOS-internal" not in s.index:
                continue
            for dkey in ("Messidor-2", "IDRiD"):
                if dkey in s.index:
                    drop_rows.append({
                        "model": mkey, "external": dkey, "resolution": int(res),
                        "qwk_internal": float(s["APTOS-internal"]),
                        "qwk_external": float(s[dkey]),
                        "qwk_drop": round(float(s["APTOS-internal"] - s[dkey]), 4),
                    })
    drop = pd.DataFrame(drop_rows)
    # delta vs the 224 baseline (same model, same external set)
    if not drop.empty:
        base = drop[drop["resolution"] == 224].set_index(["model", "external"])["qwk_drop"]
        drop["delta_drop_vs_224"] = drop.apply(
            lambda r: round(r["qwk_drop"] - base.get((r["model"], r["external"]), np.nan), 4),
            axis=1)
    drop.to_csv(OUT_DIR / "resolution_drop.csv", index=False)

    lines = ["# Phase 15 Variant C -- input resolution 224 vs 384", "",
             "Inference-only; RETFound was trained at 224 and is run at 384 with a",
             "bicubically interpolated positional embedding (off-distribution test).",
             "", "## QWK by model x dataset x resolution", "",
             "| Model | Dataset | res | QWK | acc | macro-F1 | referable AUC |",
             "|---|---|--:|--:|--:|--:|--:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['dataset']} | {r['resolution']} | "
                     f"{r['qwk']:.3f} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | "
                     f"{r['referable_auc']:.3f} |")
    lines += ["", "## Internal->external QWK drop, and change vs 224", "",
              "| Model | External | res | QWK drop | delta vs 224 |",
              "|---|---|--:|--:|--:|"]
    for _, r in drop.iterrows():
        dv = r.get("delta_drop_vs_224")
        dv_s = "--" if dv is None or (isinstance(dv, float) and np.isnan(dv)) else f"{dv:+.3f}"
        lines.append(f"| {r['model']} | {r['external']} | {int(r['resolution'])} | "
                     f"{r['qwk_drop']:.3f} | {dv_s} |")
    (OUT_DIR / "resolution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 70)
    print(master.to_string(index=False))
    print()
    print(drop.to_string(index=False))
    print(f"\nwrote {OUT_DIR}/resolution_master.csv, resolution_drop.csv, resolution_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
