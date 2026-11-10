"""
Phase 31 - single source of truth for every number in the manuscript.

Design (pre-specified in Phases 23-29; revised per the round-3 review):
  * Internal test set = APTOS internal test with the 41 images that have an exact
    duplicate in APTOS train/val removed (n = 692; primary). Full 733 = sensitivity.
  * External sets = Messidor-2 (1,744) and the official IDRiD release (516).
  * Bootstrap B = 2000, seed 42, the SAME replicate indices for every model (paired).
    Messidor-2 is resampled by examination (reconstructed fellow-eye pairs,
    outputs/phase30/messidor2_pairs_official.csv, official ADCIS file); APTOS and IDRiD have no patient
    identifiers and are resampled by image.
  * Primary estimand: difference in internal-to-external QWK drop vs EfficientNet-B3;
    confirmatory family = RETFound linear probe and full fine-tune x 2 external sets,
    Holm-adjusted two-sided bootstrap p values.
Any model whose prediction files exist is included automatically (seeds, controls).

Run (one model per call keeps each step short):
  python scripts/phase31_paper_numbers.py --model baseline_efficientnet_b3
  ...
  python scripts/phase31_paper_numbers.py --merge
Outputs: outputs/phase31/{reps/<model>.npz, point.csv, ci.csv, contrasts.csv, numbers.json, summary.md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase31"
P6, P28 = ROOT / "outputs" / "phase6", ROOT / "outputs" / "phase28"
K, B, SEED = 5, 2000, 42
DS = ["APTOS-internal", "Messidor-2", "IDRiD"]
REF_RATES = [0.10, 0.20, 0.30]


def model_files(m: str) -> dict:
    """prediction CSV for each dataset, or None if not available."""
    prim = {"baseline_efficientnet_b3", "baseline_resnet50", "retfound_linprobe", "retfound_full"}
    if m in prim:
        f = {"APTOS-internal": P6 / f"{m}__APTOS-internal_preds.csv",
             "Messidor-2": P6 / f"{m}__Messidor-2_preds.csv",
             "IDRiD": P28 / f"{m}__IDRiD-official_preds.csv"}
    else:
        f = {"APTOS-internal": P28 / f"{m}__APTOS-internal_preds.csv",
             "Messidor-2": P28 / f"{m}__Messidor-2_preds.csv",
             "IDRiD": P28 / f"{m}__IDRiD-official_preds.csv"}
    return f if all(p.exists() for p in f.values()) else None


ALL_MODELS = ["baseline_efficientnet_b3", "baseline_resnet50", "retfound_linprobe", "retfound_full",
              "baseline_efficientnet_b3_ls01", "baseline_efficientnet_b3_rfrecipe",
              "retfound_full_seed43", "retfound_full_seed44", "retfound_full_cnnrecipe",
              "baseline_efficientnet_b3_seed43", "baseline_efficientnet_b3_seed44", "baseline_efficientnet_b3_seed45",
              "retfound_linprobe_seed43", "retfound_linprobe_seed44", "retfound_linprobe_seed45"]
SEED_GROUPS = {
    "effb3": ["baseline_efficientnet_b3", "baseline_efficientnet_b3_seed43",
              "baseline_efficientnet_b3_seed44", "baseline_efficientnet_b3_seed45"],
    "retfound_linprobe": ["retfound_linprobe", "retfound_linprobe_seed43",
                          "retfound_linprobe_seed44", "retfound_linprobe_seed45"],
    "retfound_full": ["retfound_full", "retfound_full_seed43", "retfound_full_seed44"],
}


# ------------------------------------------------------------------ data
def dedup_exclude() -> set:
    d = pd.read_csv(ROOT / "outputs" / "phase12" / "exact_duplicates.csv")
    a = d[(d.group_a == "APTOS:test") & d.group_b.isin(["APTOS:train", "APTOS:val"])].id_a
    b = d[(d.group_b == "APTOS:test") & d.group_a.isin(["APTOS:train", "APTOS:val"])].id_b
    return set(pd.concat([a, b]))


def reference_order() -> dict:
    """fixed image order per dataset (EfficientNet-B3 files) + dedup mask + clusters."""
    f = model_files("baseline_efficientnet_b3")
    order = {d: pd.read_csv(f[d]).id_code.astype(str).tolist() for d in DS}
    ex = dedup_exclude()
    keep = np.array([i not in ex for i in order["APTOS-internal"]])
    pairs = pd.read_csv(ROOT / "outputs" / "phase30" / "messidor2_pairs_official.csv").set_index("id_code").exam_id
    clus = np.array([pairs[i] for i in order["Messidor-2"]])
    return order, keep, clus


def load(m: str, order: dict) -> dict:
    out = {}
    for d, p in model_files(m).items():
        df = pd.read_csv(p); df["id_code"] = df.id_code.astype(str)
        df = df.set_index("id_code").reindex(order[d])
        assert df.y_true.notna().all(), (m, d)
        P = df[[f"prob_{k}" for k in range(K)]].to_numpy(float)
        out[d] = (df.y_true.to_numpy(int), P.argmax(1), P)
    return out


def boot_indices(order, keep, clus):
    rng = np.random.default_rng(SEED)
    idx = {}
    nI = int(keep.sum()); base_I = np.where(keep)[0]
    idx["APTOS-internal"] = base_I[rng.integers(0, nI, (B, nI))]
    idx["APTOS-full"] = rng.integers(0, len(keep), (B, len(keep)))
    u, inv = np.unique(clus, return_inverse=True)
    members = [np.where(inv == c)[0] for c in range(len(u))]
    mi = []
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        mi.append(np.concatenate([members[c] for c in pick]))
    idx["Messidor-2"] = mi
    nE = len(order["IDRiD"])
    idx["IDRiD"] = rng.integers(0, nE, (B, nE))
    return idx


# ------------------------------------------------------------------ metrics
W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
W4 = (np.subtract.outer(np.arange(4), np.arange(4)) ** 2) / 9


def wqwk(y, h, w=None, k=K):
    O = np.zeros((k, k)); np.add.at(O, (y, h), 1 if w is None else w)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    Wt = W if k == K else W4
    den = (Wt * E).sum()
    return 1 - (Wt * O).sum() / den if den > 0 else np.nan


def auc(s, pos):
    n1 = pos.sum(); n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ece(P, y, bins=15):
    c = P.max(1); a = P.argmax(1) == y
    b = np.minimum((c * bins).astype(int), bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.mean() * abs(a[m].mean() - c[m].mean())
    return tot


def metrics(y, h, P, ref_prev, rnd):
    """all per-sample metrics for one (resampled) set."""
    out = {}
    ext_prev = np.bincount(y, minlength=K) / len(y)
    w = ref_prev[y] / np.maximum(ext_prev[y], 1e-12)
    y4, h4 = np.maximum(y - 1, 0), np.maximum(h - 1, 0)
    out["qwk"] = wqwk(y, h); out["qwk_rw"] = wqwk(y, h, w)
    out["qwk4"] = wqwk(y4, h4, k=4); out["qwk4_rw"] = wqwk(y4, h4, w, k=4)
    out["acc"] = float((y == h).mean())
    t, p = y >= 2, h >= 2
    out["sens"] = (p & t).sum() / max(t.sum(), 1); out["spec"] = (~p & ~t).sum() / max((~t).sum(), 1)
    out["ppv"] = (p & t).sum() / max(p.sum(), 1); out["npv"] = (~p & ~t).sum() / max((~p).sum(), 1)
    out["ref_auc"] = auc(P[:, 2:].sum(1), t)
    out["ece"] = ece(P, y)
    err = y != h
    Ps = np.sort(P, 1)
    sc = {"msp": 1 - Ps[:, -1], "margin": -(Ps[:, -1] - Ps[:, -2]),
          "entropy": -(P * np.log(np.clip(P, 1e-12, 1))).sum(1), "random": rnd}
    for k, s in sc.items():
        if k != "random":
            out[f"err_auc_{k}"] = auc(s, err)
        order = np.argsort(s, kind="mergesort")
        n = len(y)
        for r in REF_RATES:
            kept = order[: n - int(round(r * n))]
            out[f"missed_{k}_{int(r*100)}"] = ((y[kept] >= 2) & (h[kept] < 2)).sum() / max(t.sum(), 1)
            if k == "msp":
                out[f"acc_kept_{k}_{int(r*100)}"] = float((y[kept] == h[kept]).mean())
                out[f"qwk_kept_{k}_{int(r*100)}"] = wqwk(y[kept], h[kept])
    out["missed_0"] = ((y >= 2) & (h < 2)).sum() / max(t.sum(), 1)
    return out


def run_model(m):
    order, keep, clus = reference_order()
    idx = boot_indices(order, keep, clus)
    data = load(m, order)
    yI = data["APTOS-internal"][0][keep]
    ref_prev = np.bincount(yI, minlength=K) / len(yI)
    rngr = np.random.default_rng(SEED)
    rnd = {d: rngr.random(len(order[d])) for d in DS}
    point, reps = {}, {}
    sets = {"APTOS-internal": np.where(keep)[0], "APTOS-full": np.arange(len(keep)),
            "Messidor-2": np.arange(len(order["Messidor-2"])), "IDRiD": np.arange(len(order["IDRiD"]))}
    for s, base in sets.items():
        d = "APTOS-internal" if s.startswith("APTOS") else s
        y, h, P = data[d]
        point[s] = metrics(y[base], h[base], P[base], ref_prev, rnd[d][base])
        keys = list(point[s])
        R = np.full((B, len(keys)), np.nan)
        for b in range(B):
            ii = idx[s][b]
            mt = metrics(y[ii], h[ii], P[ii], ref_prev, rnd[d][ii])
            R[b] = [mt[k] for k in keys]
        reps[s] = (keys, R)
    (OUT / "reps").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "reps" / f"{m}.npz",
                        **{f"{s}__R": reps[s][1] for s in reps},
                        **{f"{s}__keys": np.array(reps[s][0]) for s in reps},
                        point=json.dumps(point))
    print("done", m, flush=True)


# ------------------------------------------------------------------ merge
def pval(x):
    x = x[~np.isnan(x)]
    return float(min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean())))


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def merge():
    avail = [m for m in ALL_MODELS if (OUT / "reps" / f"{m}.npz").exists()]
    Z = {m: np.load(OUT / "reps" / f"{m}.npz", allow_pickle=True) for m in avail}
    point = {m: json.loads(str(Z[m]["point"])) for m in avail}

    def rep(m, s, k):
        keys = list(Z[m][f"{s}__keys"]); return Z[m][f"{s}__R"][:, keys.index(k)]
    rows = []
    for m in avail:
        for s in ["APTOS-internal", "APTOS-full", "Messidor-2", "IDRiD"]:
            for k, v in point[m][s].items():
                r = rep(m, s, k)
                rows.append(dict(model=m, set=s, metric=k, value=v,
                                 lo=np.nanpercentile(r, 2.5), hi=np.nanpercentile(r, 97.5)))
    ci = pd.DataFrame(rows); ci.to_csv(OUT / "ci.csv", index=False)

    # drops and contrasts
    con = []
    base = "baseline_efficientnet_b3"

    def drop(m, ext, k, internal="APTOS-internal"):
        return point[m][internal][k] - point[m][ext][k], rep(m, internal, k) - rep(m, ext, k)
    for internal in ["APTOS-internal", "APTOS-full"]:
        for m in avail:
            for ext in ["Messidor-2", "IDRiD"]:
                for k in ["qwk", "qwk_rw", "qwk4", "qwk4_rw", "ref_auc"]:
                    d, dr = drop(m, ext, k, internal)
                    row = dict(internal=internal, model=m, external=ext, metric=k, drop=d,
                               drop_lo=np.nanpercentile(dr, 2.5), drop_hi=np.nanpercentile(dr, 97.5))
                    if m != base:
                        db, dbr = drop(base, ext, k, internal)
                        dd = dr - dbr
                        row.update(ddrop=d - db, ddrop_lo=np.nanpercentile(dd, 2.5),
                                   ddrop_hi=np.nanpercentile(dd, 97.5), p=pval(dd))
                    con.append(row)
    # seed-averaged contrasts: mean drop over available seeds of a family minus mean over EfficientNet-B3 seeds
    def group_drop(g, ext, k, internal):
        ms = [m for m in SEED_GROUPS[g] if m in avail]
        ds = [drop(m, ext, k, internal) for m in ms]
        return ms, float(np.mean([x[0] for x in ds])), np.mean([x[1] for x in ds], axis=0), [x[0] for x in ds]

    def group_reps(g, ext, k, internal):
        """per-seed replicate drops, shape (n_seeds, B)."""
        ms = [m for m in SEED_GROUPS[g] if m in avail]
        return np.vstack([drop(m, ext, k, internal)[1] for m in ms])
    rng_h = np.random.default_rng(SEED + 1)

    def hier(g, ext, k, internal):
        """two-level bootstrap: seeds resampled with replacement within each family,
        images resampled as before (same image replicate for all models)."""
        A = group_reps(g, ext, k, internal); Bm = group_reps("effb3", ext, k, internal)
        ia = rng_h.integers(0, A.shape[0], (B, A.shape[0])); ib = rng_h.integers(0, Bm.shape[0], (B, Bm.shape[0]))
        cols = np.arange(B)
        da = A[ia, cols[:, None]].mean(1); db = Bm[ib, cols[:, None]].mean(1)
        return da - db
    seedrows = []
    for internal in ["APTOS-internal", "APTOS-full"]:
        for ext in ["Messidor-2", "IDRiD"]:
            for k in ["qwk", "qwk_rw", "qwk4_rw", "ref_auc"]:
                mb, db, dbr, per_b = group_drop("effb3", ext, k, internal)
                seedrows.append(dict(internal=internal, group="effb3", external=ext, metric=k, n_seeds=len(mb),
                                     seeds=";".join(mb), drop=db, drop_sd=float(np.std(per_b, ddof=1)) if len(per_b) > 1 else np.nan,
                                     drop_lo=np.nanpercentile(dbr, 2.5), drop_hi=np.nanpercentile(dbr, 97.5),
                                     per_seed_drops=";".join(f"{x:.4f}" for x in per_b)))
                for g in ["retfound_linprobe", "retfound_full"]:
                    ms, d, dr, per = group_drop(g, ext, k, internal)
                    dd = dr - dbr
                    seedrows.append(dict(internal=internal, group=g, external=ext, metric=k, n_seeds=len(ms),
                                         seeds=";".join(ms), drop=d,
                                         drop_sd=float(np.std(per, ddof=1)) if len(per) > 1 else np.nan,
                                         drop_lo=np.nanpercentile(dr, 2.5), drop_hi=np.nanpercentile(dr, 97.5),
                                         ddrop=d - db, ddrop_lo=np.nanpercentile(dd, 2.5),
                                         ddrop_hi=np.nanpercentile(dd, 97.5), p=pval(dd),
                                         **(lambda h: dict(h_lo=np.nanpercentile(h, 2.5), h_hi=np.nanpercentile(h, 97.5), h_p=pval(h)))(hier(g, ext, k, internal)),
                                         per_seed_drops=";".join(f"{x:.4f}" for x in per)))
    S = pd.DataFrame(seedrows)
    for internal in ["APTOS-internal", "APTOS-full"]:
        fam = (S.internal == internal) & (S.metric == "qwk") & S.group.isin(["retfound_linprobe", "retfound_full"])
        S.loc[fam, "p_holm"] = holm(S.loc[fam, "p"].values)
        S.loc[fam, "h_p_holm"] = holm(S.loc[fam, "h_p"].values)
    S.to_csv(OUT / "seed_contrasts.csv", index=False)
    # matched-recipe pair contrasts (seed 42): model a minus model b, same image replicates
    PAIRS = [("retfound_full", "baseline_efficientnet_b3_rfrecipe", "xrf"),
             ("retfound_full_cnnrecipe", "baseline_efficientnet_b3", "xcnn")]
    pr = []
    for a, b_, tag in PAIRS:
        if a not in avail or b_ not in avail: continue
        for ext in ["Messidor-2", "IDRiD"]:
            for k in ["qwk", "qwk4_rw", "ref_auc"]:
                da, dar = drop(a, ext, k); db_, dbr_ = drop(b_, ext, k); dd = dar - dbr_
                pr.append(dict(tag=tag, a=a, b=b_, external=ext, metric=k, drop_a=da, drop_b=db_, ddrop=da - db_,
                               ddrop_lo=np.nanpercentile(dd, 2.5), ddrop_hi=np.nanpercentile(dd, 97.5), p=pval(dd)))
    pd.DataFrame(pr, columns=["tag", "a", "b", "external", "metric", "drop_a", "drop_b", "ddrop", "ddrop_lo", "ddrop_hi", "p"]).to_csv(OUT / "pair_contrasts.csv", index=False)
    C = pd.DataFrame(con)
    for internal in ["APTOS-internal", "APTOS-full"]:
        fam = (C.internal == internal) & (C.metric == "qwk") & C.model.isin(["retfound_linprobe", "retfound_full"])
        C.loc[fam, "p_holm"] = holm(C.loc[fam, "p"].values)
    # paired external AUC differences (cluster-aware bootstrap) and error-detection score differences
    oth = []
    for m in avail:
        if m == base:
            continue
        for s in ["APTOS-internal", "Messidor-2", "IDRiD"]:
            v = point[m][s]["ref_auc"] - point[base][s]["ref_auc"]
            r = rep(m, s, "ref_auc") - rep(base, s, "ref_auc")
            oth.append(dict(kind="ref_auc_vs_effb3", model=m, set=s, value=v,
                            lo=np.nanpercentile(r, 2.5), hi=np.nanpercentile(r, 97.5), p=pval(r)))
    for m in avail:
        for s in ["APTOS-internal", "Messidor-2", "IDRiD"]:
            for sc in ["entropy", "margin"]:
                v = point[m][s][f"err_auc_{sc}"] - point[m][s]["err_auc_msp"]
                r = rep(m, s, f"err_auc_{sc}") - rep(m, s, "err_auc_msp")
                oth.append(dict(kind=f"err_auc_{sc}_minus_msp", model=m, set=s, value=v,
                                lo=np.nanpercentile(r, 2.5), hi=np.nanpercentile(r, 97.5), p=pval(r)))
    O = pd.DataFrame(oth)
    prim = ["baseline_resnet50", "retfound_linprobe", "retfound_full"]
    f = (O.kind == "ref_auc_vs_effb3") & O.model.isin(prim)
    O.loc[f, "p_holm"] = holm(O.loc[f, "p"].values)
    C.to_csv(OUT / "contrasts.csv", index=False); O.to_csv(OUT / "other_contrasts.csv", index=False)
    json.dump({"models": avail}, open(OUT / "meta.json", "w"))
    print("merged", avail)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--merge", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.list:
        print([m for m in ALL_MODELS if model_files(m)])
    elif a.merge:
        merge()
    else:
        run_model(a.model)
