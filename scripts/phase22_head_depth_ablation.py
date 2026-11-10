"""
Phase 22 -- classifier-head depth x activation ablation (DR grading).

WHY THE HEAD AND NOT THE BACKBONE
---------------------------------
EfficientNet-B3, ResNet-50 and RETFound ViT-L/16 all arrive with their depth
fixed by the pretrained checkpoint. You cannot set "5 hidden layers" on a
ViT-L/16 and still be using RETFound -- the pretrained weights are the asset,
and changing the trunk throws them away. The depth that is genuinely yours to
choose, and that the paper never justified, is the depth of the classifier head
sitting on top of the frozen features.

So this sweep is honest about what it varies:

    frozen backbone  ->  embedding  ->  MLP head with d hidden layers  ->  5 classes
                                         d in {0 (linear probe), 3, 4, 5, 6, 7}

Depth 0 is the published linear-probe configuration and is the anchor every
other row is compared against. Report it as such.

TWO STAGES
----------
Stage 1 extracts embeddings ONCE per (backbone, dataset) and caches them. This
is the expensive part and it is GPU-bound. Stage 2 sweeps 6 depths x 4
activations x 5 seeds over the cached embeddings on CPU in minutes, because the
backbone never re-runs.

    python scripts/phase22_head_depth_ablation.py --stage embed \
        --checkpoint outputs/phase4_baseline/efficientnet_b3_best.pt \
        --manifest data_splits/aptos_manifest.csv --device cuda

    python scripts/phase22_head_depth_ablation.py --stage embed \
        --checkpoint outputs/phase5_retfound/retfound_linprobe_best.pt \
        --manifest data_splits/aptos_manifest.csv --device cuda

    python scripts/phase22_head_depth_ablation.py --stage sweep \
        --outdir outputs/phase22

Add --external to also embed Messidor-2 and IDRiD; the sweep then reports the
depth effect internally AND under dataset shift, which is the question this
paper actually cares about. A head depth that helps internally and hurts
externally is the single most useful thing this ablation can find, and it is
invisible if you only run the internal test set.

PLACEMENT
---------
Drop in `scripts/` of the DIABETICS-AND-FUNDUS-IMAGES repo. Imports the repo's
own dataset, preprocessing and metrics modules so the sweep sees the committed
pipeline unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from ablation_common import (  # noqa: E402
    DEPTHS, SEEDS, RunRecord, count_params, hidden_sizes, key,
    metrics_from_confusion, write_outputs,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
CACHE = PROJECT_DIR / "outputs" / "phase22" / "embeddings"
NUM_CLASSES = 5

# torch gives us the full set, including GELU -- the activation the ViT trunk
# itself uses, which makes it the most defensible choice for a head sitting on
# ViT features. 'identity' is the control: it collapses the stack to a linear
# map, so its row says how much non-linearity the head needed at all.
ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "identity": nn.Identity,
}


# --------------------------------------------------------------------------- #
# Stage 1: freeze the backbone, cache the embeddings
# --------------------------------------------------------------------------- #

def strip_head(model: nn.Module, arch_hint: str) -> nn.Module:
    """Replace the classifier with Identity so forward() returns the embedding."""
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        model.classifier = nn.Identity()          # efficientnet
    elif hasattr(model, "fc"):
        model.fc = nn.Identity()                  # resnet
    elif hasattr(model, "head"):
        model.head = nn.Identity()                # vit / retfound
    elif hasattr(model, "heads"):
        model.heads = nn.Identity()
    else:
        raise RuntimeError(
            f"could not locate the classifier on {arch_hint}; "
            "add the attribute name to strip_head()"
        )
    return model


@torch.no_grad()
def stage_embed(args) -> int:
    from dataset import build_dataloader          # repo's own loader
    from predict import load_checkpoint, pick_device

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    arch = ckpt.get("arch", Path(args.checkpoint).stem)
    model = strip_head(model, arch).to(device).eval()

    CACHE.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"] if not args.external else ["external"]

    for split in splits:
        loader = build_dataloader(
            manifest_csv=args.manifest, split=split, train=False,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        feats, labels = [], []
        t0 = time.time()
        for xb, yb, _ in loader:
            feats.append(model(xb.to(device)).float().cpu().numpy())
            labels.append(yb.numpy())
        X = np.concatenate(feats)
        y = np.concatenate(labels)
        tag = f"{arch}__{Path(args.manifest).stem}__{split}"
        np.savez_compressed(CACHE / f"{tag}.npz", X=X, y=y)
        print(f"  {tag}: X{X.shape} y{y.shape}  ({time.time()-t0:.0f}s)")

    print(f"\nembeddings cached under {CACHE}")
    print("backbone was frozen and in eval() throughout; no gradients were taken.")
    return 0


# --------------------------------------------------------------------------- #
# Stage 2: sweep the head
# --------------------------------------------------------------------------- #

class MLPHead(nn.Module):
    """d hidden layers, one activation, one dropout rate. d=0 is a linear probe."""

    def __init__(self, n_in: int, hidden: tuple, activation: str,
                 n_out: int = NUM_CLASSES, p_drop: float = 0.3):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), act(), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_head(Xtr, ytr, Xva, yva, hidden, activation, seed, epochs=60,
             lr=1e-3, batch=256, device="cpu", patience=10):
    """Train one head. Early stopping on validation QWK-proxy (balanced accuracy)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLPHead(Xtr.shape[1], hidden, activation).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Class-balanced loss: DR grades are heavily imbalanced and an unweighted
    # head will simply predict grade 0, which makes every depth look identical.
    counts = np.bincount(ytr, minlength=NUM_CLASSES).astype(float)
    w = torch.tensor((counts.sum() / np.maximum(counts, 1)), dtype=torch.float32)
    w = (w / w.mean()).to(device)
    crit = nn.CrossEntropyLoss(weight=w)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32).to(device)

    best, best_state, bad = -np.inf, None, 0
    n = len(Xtr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx].to(device)), ytr_t[idx].to(device))
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).argmax(1).cpu().numpy()
        from sklearn.metrics import balanced_accuracy_score
        score = balanced_accuracy_score(yva, pv)
        if score > best:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, ep + 1


def stage_sweep(args) -> int:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    files = sorted(CACHE.glob("*__aptos_manifest__train.npz"))
    if not files:
        print(f"no cached embeddings in {CACHE}; run --stage embed first")
        return 1

    depths = [0] + list(args.depths)        # 0 = published linear probe, the anchor
    records: list[RunRecord] = []
    matrices: dict = {}

    for ftr in files:
        stem = ftr.name.replace("__train.npz", "")
        tr = np.load(ftr)
        va = np.load(CACHE / f"{stem}__val.npz")
        te = np.load(CACHE / f"{stem}__test.npz")
        Xtr, ytr = tr["X"], tr["y"]
        Xva, yva = va["X"], va["y"]
        Xte, yte = te["X"], te["y"]

        # Standardise on train statistics only -- same discipline as the repo.
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
        arch = stem.split("__")[0]
        eval_sets = [("aptos_internal", Xte, yte)]
        for fext in sorted(CACHE.glob(f"{arch}__*__external.npz")):
            ex = np.load(fext)
            ename = fext.name.split("__")[1].replace("_manifest", "")
            eval_sets.append((ename, (ex["X"] - mu) / sd, ex["y"]))
        print("  eval sets:", [(n, len(y)) for n, _, y in eval_sets])

        print(f"\n{stem}: train{Xtr.shape} val{Xva.shape} test{Xte.shape}")

        for depth in depths:
            hidden = () if depth == 0 else hidden_sizes(depth, args.width, args.width_policy)
            acts = ["identity"] if depth == 0 else list(ACTIVATIONS)
            for act in acts:
                for seed in args.seeds:
                    t0 = time.time()
                    model, n_ep = fit_head(Xtr, ytr, Xva, yva, hidden, act, seed,
                                           epochs=args.epochs, device=args.device)
                    fit_s = time.time() - t0
                    model.eval()
                    for eval_name, Xe, ye in eval_sets:
                        with torch.no_grad():
                            logits = model(torch.tensor(Xe, dtype=torch.float32).to(args.device))
                            pred = logits.argmax(1).cpu().numpy()
                        cm = confusion_matrix(ye, pred, labels=list(range(NUM_CLASSES)))
                        m = metrics_from_confusion(cm)
                        qwk = float(cohen_kappa_score(ye, pred,
                                                      labels=list(range(NUM_CLASSES)),
                                                      weights="quadratic"))
                        model_id = f"{arch}@{eval_name}"
                        matrices[key(model_id, depth, act, seed)] = cm.tolist()
                        records.append(RunRecord(
                            model=model_id, depth=depth, activation=act, seed=seed,
                            width_policy=args.width_policy, hidden_sizes=str(hidden),
                            n_params=count_params(hidden, Xtr.shape[1], NUM_CLASSES),
                            n_train=len(ytr), n_test=len(ye),
                            fit_seconds=fit_s, converged=(n_ep < args.epochs),
                            note=f"qwk={qwk:.4f};epochs={n_ep}", **m,
                        ))
                        print(f"  {eval_name:<20} d={depth:<2} {act:<9} seed={seed} "
                              f"QWK={qwk:.4f} bacc={m['balanced_accuracy']:.4f}", flush=True)

    write_outputs(
        Path(args.outdir), records, matrices,
        class_names=["0 none", "1 mild", "2 moderate", "3 severe", "4 proliferative"],
        context={
            "paper": "Cross-dataset external validation of a retinal foundation model",
            "what_varies": "classifier-head depth and activation, on FROZEN backbone "
                           "embeddings. Backbone depth is fixed by the pretrained "
                           "checkpoint and is NOT varied.",
            "anchor": "depth 0 = the published linear probe",
            "qwk_note": "QWK is recorded per run in the note field; it is the "
                        "paper's primary metric and should lead the table.",
        },
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["embed", "sweep"], required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--manifest", default="data_splits/aptos_manifest.csv")
    ap.add_argument("--external", action="store_true",
                    help="embed an external manifest (test split only)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--width-policy", choices=["constant", "pyramid"], default="constant")
    ap.add_argument("--depths", type=int, nargs="+", default=DEPTHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--outdir", default="outputs/phase22")
    args = ap.parse_args()

    if args.stage == "embed":
        if not args.checkpoint:
            ap.error("--stage embed needs --checkpoint")
        return stage_embed(args)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return stage_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
