"""
Run a trained checkpoint over a dataset split and dump raw predictions + metrics.

Used for:
  - Phase 4/5 internal-test evaluation (called from the training scripts)
  - Phase 6 external validation on Messidor-2 / IDRiD (CLI, unmodified model)
  - Phase 8 MC-dropout (--mc-dropout --passes 30)

CLI:
  .venv/Scripts/python.exe scripts/predict.py \
      --checkpoint outputs/phase4_baseline/resnet50_best.pt \
      --manifest data_splits/messidor2_manifest.csv --split external \
      --out-prefix outputs/phase6/resnet50_messidor2

Outputs <out-prefix>_preds.csv  (id_code, y_true, y_pred, prob_0..prob_4, entropy)
        <out-prefix>_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import NUM_CLASSES, build_dataloader  # noqa: E402
from metrics import compute_metrics, format_metrics  # noqa: E402
from models import build_model, set_dropout_p  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    model = build_model(arch, num_classes=ckpt.get("num_classes", NUM_CLASSES),
                        pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def infer(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    mc_dropout: bool = False,
    passes: int = 1,
) -> dict:
    """Return dict of numpy arrays: id_code, y_true, y_pred, prob (N,C), entropy.

    mc_dropout=True keeps dropout layers active and averages `passes` softmax
    outputs per image (predictive distribution for Phase 8).
    """
    model.eval()
    if mc_dropout:
        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    ids: list[str] = []
    ys: list[int] = []
    probs: list[np.ndarray] = []

    for images, labels, id_codes in loader:
        images = images.to(device, non_blocking=True)
        if mc_dropout and passes > 1:
            acc = torch.zeros(images.size(0), NUM_CLASSES, device=device)
            for _ in range(passes):
                acc += F.softmax(model(images), dim=1)
            p = (acc / passes).cpu().numpy()
        else:
            p = F.softmax(model(images), dim=1).cpu().numpy()
        probs.append(p)
        ids.extend(id_codes)
        ys.extend(int(v) for v in labels)

    prob = np.concatenate(probs, axis=0)
    y_true = np.asarray(ys, dtype=int)
    y_pred = prob.argmax(axis=1)
    eps = 1e-12
    entropy = -(prob * np.log(prob + eps)).sum(axis=1)
    return {
        "id_code": np.asarray(ids),
        "y_true": y_true,
        "y_pred": y_pred,
        "prob": prob,
        "entropy": entropy,
    }


def save_predictions(res: dict, out_prefix: str | Path) -> tuple[Path, Path]:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"id_code": res["id_code"], "y_true": res["y_true"],
                       "y_pred": res["y_pred"]})
    for c in range(res["prob"].shape[1]):
        df[f"prob_{c}"] = res["prob"][:, c]
    df["entropy"] = res["entropy"]
    preds_path = out_prefix.with_name(out_prefix.name + "_preds.csv")
    df.to_csv(preds_path, index=False)

    m = compute_metrics(res["y_true"], res["y_pred"], res["prob"],
                        num_classes=res["prob"].shape[1])
    metrics_path = out_prefix.with_name(out_prefix.name + "_metrics.json")
    metrics_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return preds_path, metrics_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--mc-dropout", action="store_true")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--dropout-p", type=float, default=None,
                    help="override dropout p before MC-dropout inference")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    if args.dropout_p is not None:
        set_dropout_p(model, args.dropout_p)

    loader = build_dataloader(
        args.manifest, split=args.split, batch_size=args.batch_size,
        train=False, num_workers=args.num_workers, shuffle=False,
    )
    res = infer(model, loader, device, mc_dropout=args.mc_dropout, passes=args.passes)
    preds_path, metrics_path = save_predictions(res, args.out_prefix)

    m = json.loads(metrics_path.read_text())
    print(format_metrics(m, title=f"{Path(args.checkpoint).name} on "
                                  f"{Path(args.manifest).stem}:{args.split}"))
    print(f"\nwrote {preds_path}\nwrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
