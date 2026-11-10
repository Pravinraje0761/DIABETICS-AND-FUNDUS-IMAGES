"""
Phase 32 - write every manuscript number as a LaTeX macro (paper/numbers.tex) and
the results tables (paper/tables/*.tex) directly from the analysis outputs, so the
text, tables and supplement cannot disagree. In main.tex: \\nv{key}. A missing key
prints ??key?? in bold red, which the final check searches for.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P31 = ROOT / "outputs" / "phase31"
OUTD = ROOT / "paper" / "submission_io" / "generated"
OUTD.mkdir(parents=True, exist_ok=True)
ALIAS = {"baseline_efficientnet_b3": "eff", "baseline_resnet50": "res", "retfound_linprobe": "lp",
         "retfound_full": "full", "baseline_efficientnet_b3_ls01": "effls",
         "baseline_efficientnet_b3_rfrecipe": "effrf", "retfound_full_cnnrecipe": "fullcnn",
         "retfound_full_seed43": "fullA", "retfound_full_seed44": "fullB",
         "baseline_efficientnet_b3_seed43": "effA", "baseline_efficientnet_b3_seed44": "effB",
         "baseline_efficientnet_b3_seed45": "effC", "retfound_linprobe_seed43": "lpA",
         "retfound_linprobe_seed44": "lpB", "retfound_linprobe_seed45": "lpC"}
SET = {"APTOS-internal": "int", "APTOS-full": "intfull", "Messidor-2": "mes", "IDRiD": "idr"}
GRP = {"effb3": "geff", "retfound_linprobe": "glp", "retfound_full": "gfull"}
MET = {"qwk_rw": "qwkrw", "qwk4": "qwkfour", "qwk4_rw": "qwkfourrw", "ref_auc": "refauc"}
V: dict[str, str] = {}


def f3(x): return "--" if pd.isna(x) else f"{x:.3f}".replace("-", "$-$")
def f2(x): return "--" if pd.isna(x) else f"{x:.2f}".replace("-", "$-$")
def sg(x):
    if pd.isna(x): return "--"
    if round(x, 3) == 0: return "0.000"
    return f"+{x:.3f}" if x > 0 else f"$-${abs(x):.3f}"
def pc(x): return "--" if pd.isna(x) else f"{100*x:.0f}"
def pv(x):
    if pd.isna(x): return "--"
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def key(*parts): return "".join(str(p) for p in parts)


def build():
    ci = pd.read_csv(P31 / "ci.csv")
    con = pd.read_csv(P31 / "contrasts.csv")
    oth = pd.read_csv(P31 / "other_contrasts.csv")
    sc = pd.read_csv(P31 / "seed_contrasts.csv")
    # per-model metrics
    for _, r in ci.iterrows():
        if r.model not in ALIAS: continue
        m, s, k = ALIAS[r.model], SET[r.set], r.metric.replace("_", "")
        fmt = pc if (k.startswith("missed") or k in ("sens", "spec", "ppv", "npv", "acc") or k.startswith("acckept")) else f3
        V[key(m, s, k)] = fmt(r.value); V[key(m, s, k, "lo")] = fmt(r.lo); V[key(m, s, k, "hi")] = fmt(r.hi)
        if k in ("sens", "spec", "ppv", "npv", "acc"):
            V[key(m, s, k, "f")] = f2(r.value)
    # drops and single-model contrasts
    for _, r in con.iterrows():
        if r.model not in ALIAS: continue
        m, e = ALIAS[r.model], SET[r.external]
        mk = MET.get(r.metric, r.metric)
        pre = "d" if r.internal == "APTOS-internal" else "dfull"
        V[key(pre, m, e, mk)] = f3(r["drop"]); V[key(pre, m, e, mk, "lo")] = f3(r.drop_lo); V[key(pre, m, e, mk, "hi")] = f3(r.drop_hi)
        if not pd.isna(r.get("ddrop", np.nan)):
            V[key(pre, "x", m, e, mk)] = sg(r.ddrop); V[key(pre, "x", m, e, mk, "lo")] = sg(r.ddrop_lo)
            V[key(pre, "x", m, e, mk, "hi")] = sg(r.ddrop_hi); V[key(pre, "x", m, e, mk, "p")] = pv(r.p)
            if not pd.isna(r.get("p_holm", np.nan)):
                V[key(pre, "x", m, e, mk, "holm")] = pv(r.p_holm)
    # seed-averaged contrasts
    for _, r in sc.iterrows():
        g, e, mk = GRP[r.group], SET[r.external], MET.get(r.metric, r.metric)
        pre = "s" if r.internal == "APTOS-internal" else "sfull"
        V[key(pre, g, e, mk)] = f3(r["drop"]); V[key(pre, g, e, mk, "sd")] = f3(r.drop_sd)
        V[key(pre, g, e, mk, "lo")] = f3(r.drop_lo); V[key(pre, g, e, mk, "hi")] = f3(r.drop_hi)
        V[key(pre, g, "n")] = str(int(r.n_seeds))
        if isinstance(r.get("per_seed_drops", np.nan), str):
            per = [float(x) for x in r.per_seed_drops.split(";")]
            V[key(pre, g, e, mk, "min")] = f3(min(per)); V[key(pre, g, e, mk, "max")] = f3(max(per))
        if not pd.isna(r.get("ddrop", np.nan)):
            V[key(pre, "x", g, e, mk)] = sg(r.ddrop); V[key(pre, "x", g, e, mk, "lo")] = sg(r.ddrop_lo)
            V[key(pre, "x", g, e, mk, "hi")] = sg(r.ddrop_hi); V[key(pre, "x", g, e, mk, "p")] = pv(r.p)
            if not pd.isna(r.get("p_holm", np.nan)):
                V[key(pre, "x", g, e, mk, "holm")] = pv(r.p_holm)
            if "h_lo" in r.index and not pd.isna(r.h_lo):
                V[key(pre, "x", g, e, mk, "hlo")] = sg(r.h_lo); V[key(pre, "x", g, e, mk, "hhi")] = sg(r.h_hi)
                V[key(pre, "x", g, e, mk, "hp")] = pv(r.h_p)
                if not pd.isna(r.get("h_p_holm", np.nan)): V[key(pre, "x", g, e, mk, "hholm")] = pv(r.h_p_holm)

    # share of the observed drop removed by each correction (point estimates)
    cc = con[con.internal == "APTOS-internal"].set_index(["model", "external", "metric"])["drop"]
    for (mname, ext, met), d0 in cc.items():
        if met != "qwk" or mname not in ALIAS: continue
        for mm in ["qwk_rw", "qwk4", "qwk4_rw"]:
            if (mname, ext, mm) in cc.index:
                V[key("pr", ALIAS[mname], SET[ext], MET[mm])] = f"{100*(1-cc[(mname, ext, mm)]/d0):.0f}"
    # matched-recipe pair contrasts
    pcf = P31 / "pair_contrasts.csv"
    if pcf.exists():
        for _, r in pd.read_csv(pcf).iterrows():
            b = key(r.tag, SET[r.external], MET.get(r.metric, r.metric))
            V[b] = sg(r.ddrop); V[b + "lo"] = sg(r.ddrop_lo); V[b + "hi"] = sg(r.ddrop_hi); V[b + "p"] = pv(r.p)
    # internal dedup vs full: largest absolute QWK change
    cq = ci[(ci.metric == "qwk") & ci.set.isin(["APTOS-internal", "APTOS-full"]) & ci.model.isin(list(ALIAS))]
    pv_ = cq.pivot(index="model", columns="set", values="value")
    V["dedupMaxDiff"] = f"{(pv_['APTOS-internal'] - pv_['APTOS-full']).abs().max():.3f}"
    # other contrasts
    for _, r in oth.iterrows():
        if r.model not in ALIAS: continue
        m, s = ALIAS[r.model], SET[r.set]
        kk = {"ref_auc_vs_effb3": "aucx", "err_auc_entropy_minus_msp": "entx", "err_auc_margin_minus_msp": "marx"}[r.kind]
        V[key(kk, m, s)] = sg(r.value); V[key(kk, m, s, "lo")] = sg(r.lo); V[key(kk, m, s, "hi")] = sg(r.hi)
        V[key(kk, m, s, "p")] = pv(r.p)
        if not pd.isna(r.get("p_holm", np.nan)): V[key(kk, m, s, "holm")] = pv(r.p_holm)
    # fixed facts from other phases
    loc = pd.read_csv(ROOT / "outputs/phase26/localization_summary.csv")
    for _, r in loc[loc.saliency != "cross_image"].iterrows():
        m = {"centre": "ctr"}.get(r.model, ALIAS.get(r.model))
        for k in ["auroc_lesion", "pointing_hit", "lift_any", "top10_precision", "energy_od", "area_any"]:
            V[key("loc", m, k.replace("_", ""))] = f3(r[k]) if k != "pointing_hit" else f2(r[k])
            V[key("loc", m, k.replace("_", ""), "lo")] = f3(r[k + "_lo"]) if k != "pointing_hit" else f2(r[k + "_lo"])
            V[key("loc", m, k.replace("_", ""), "hi")] = f3(r[k + "_hi"]) if k != "pointing_hit" else f2(r[k + "_hi"])
    V["locAreaPct"] = f"{100*loc[loc.model=='baseline_efficientnet_b3'].area_any.iloc[0]:.1f}"
    _lm = loc[(loc.saliency != "cross_image") & (loc.model != "centre")]
    V["locTopMaxPct"] = f"{100*_lm.top10_precision.max():.0f}"
    rm = pd.read_csv(ROOT / "outputs/phase26/resolution_matched_summary.csv")
    for _, r in rm[rm.resolution == "7x7"].iterrows():
        m = ALIAS[r.model]
        for k in ["lift_any", "auroc_lesion", "top10_precision"]:
            V[key("loc7", m, k.replace("_", ""))] = f3(r[k])
    pt = pd.read_csv(ROOT / "outputs/phase26/paired_tests.csv")
    for _, r in pt.iterrows():
        vs = {"centre": "ctr", "cross_image": "cross", "EfficientNet-B3": "eff"}[r.vs]
        V[key("locp", ALIAS[r.model], vs, r.metric.replace("_", ""))] = pv(r.wilcoxon_p)
    rt = pd.read_csv(ROOT / "outputs/phase26/resolution_matched_tests.csv")
    for _, r in rt.iterrows():
        V[key("loc7p", ALIAS[r.model], r.metric.replace("_", ""))] = pv(r.wilcoxon_p)
    pairs = pd.read_csv(ROOT / "outputs/phase30/messidor2_pairs_official.csv")
    V["pairsN"] = str(int(pairs.paired.sum() // 2)); V["pairsSingle"] = str(int((~pairs.paired).sum()))
    V["pairsExams"] = str(int(pairs.exam_id.nunique()))
    lab = pd.read_csv(ROOT / "data_splits/messidor2_manifest.csv").set_index("id_code").diagnosis
    pp = pairs[pairs.paired].groupby("exam_id").id_code.apply(list)
    V["pairsAgree"] = f"{100*np.mean([lab[a] == lab[b] for a, b in pp]):.0f}"
    pg = lab.value_counts(normalize=True).values
    V["pairsAgreeRand"] = f"{100*np.sum(pg**2):.0f}"
    V["pairsWithinOne"] = f"{100*np.mean([abs(lab[a] - lab[b]) <= 1 for a, b in pp]):.0f}"
    p6 = pd.read_csv(ROOT / "outputs/phase6/baseline_efficientnet_b3__Messidor-2_preds.csv").set_index("id_code")
    err = (p6.y_true != p6.y_pred).astype(float)
    a = np.array([[err[x], err[y]] for x, y in pp])
    V["pairsErrCorr"] = f"{np.corrcoef(a[:,0], a[:,1])[0,1]:.2f}"
    rv = json.load(open(ROOT / "outputs/phase30/reconstruction_vs_official.json"))
    V["recPPV"] = f"{100*rv['reconstructed_ppv']:.0f}"; V["recLat"] = f"{100*rv['laterality_accuracy']:.0f}"
    audit = pd.read_csv(ROOT / "outputs/phase28/mirror_vs_official_labels.csv")
    V["mirrorDiff"] = str(int(audit.label_differs.sum())); V["mirrorN"] = str(int(audit.in_mirror.sum()))
    V["mirrorPct"] = f"{100*audit.label_differs.sum()/audit.in_mirror.sum():.1f}"
    V["mirrorMissing"] = str(int((~audit.in_mirror).sum()))
    x = audit[audit.label_differs]
    V["mirrorDiseasedAsZero"] = str(int(((x.diagnosis >= 1) & (x.mirror_diagnosis == 0)).sum()))
    # counts at the no-DR/mild boundary (seed-42 primary models)
    for mname in ["baseline_efficientnet_b3", "retfound_full"]:
        m = ALIAS[mname]
        pm = pd.read_csv(ROOT / f"outputs/phase6/{mname}__Messidor-2_preds.csv")
        V[key("cnt", m, "mesMildToZero")] = str(int(((pm.y_true == 1) & (pm.y_pred == 0)).sum()))
        V[key("cnt", m, "mesModToZero")] = str(int(((pm.y_true == 2) & (pm.y_pred == 0)).sum()))
        pi = pd.read_csv(ROOT / f"outputs/phase28/{mname}__IDRiD-official_preds.csv")
        V[key("cnt", m, "idrZeroToMild")] = str(int(((pi.y_true == 0) & (pi.y_pred == 1)).sum()))
    pm = pd.read_csv(ROOT / "outputs/phase6/baseline_efficientnet_b3__Messidor-2_preds.csv")
    V["cntMesMild"] = str(int((pm.y_true == 1).sum())); V["cntMesMod"] = str(int((pm.y_true == 2).sum()))
    pi = pd.read_csv(ROOT / "outputs/phase28/baseline_efficientnet_b3__IDRiD-official_preds.csv")
    V["cntIdrZero"] = str(int((pi.y_true == 0).sum()))
    # inline-p variants: \nv{<key>eq} renders "= 0.012" or "< 0.001" for use after "$p$"
    for k in [k for k in V if k.endswith(("p", "holm")) or k.startswith(("locp", "loc7p"))]:
        v = V[k]
        if v == "$<$0.001": V[k + "eq"] = "$<$\\,0.001"
        elif v.replace(".", "").isdigit(): V[k + "eq"] = "$=$\\,"+ v
    # write
    lines = ["% generated by scripts/phase32_tex_numbers.py -- do not edit by hand",
             r"\makeatletter",
             r"\newcommand{\nv}[1]{\@ifundefined{nv@#1}{\textcolor{red}{\textbf{??#1??}}}{\csname nv@#1\endcsname}}"]
    for k, v in sorted(V.items()):
        lines.append(rf"\expandafter\def\csname nv@{k}\endcsname{{{v}}}")
    lines.append(r"\makeatother")
    (OUTD / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump(V, open(OUTD / "numbers.json", "w"), indent=0)
    print(len(V), "values written")


if __name__ == "__main__":
    build()
