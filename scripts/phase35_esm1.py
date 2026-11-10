"""Phase 35 - generate the data-driven sections of Online Resource 1 (ESM1) from Phase 31/32
outputs, so that the supplement, tables and text share one source. Writes outputs/phase35/esm1_generated.md"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
P31 = ROOT / "outputs/phase31"; OUT = ROOT / "outputs/phase35"; OUT.mkdir(parents=True, exist_ok=True)
NV = json.load(open(ROOT / "paper/submission_io/generated/numbers.json"))
ci = pd.read_csv(P31 / "ci.csv"); con = pd.read_csv(P31 / "contrasts.csv"); sc = pd.read_csv(P31 / "seed_contrasts.csv")
oth = pd.read_csv(P31 / "other_contrasts.csv")
pc_ = pd.read_csv(P31 / "pair_contrasts.csv") if (P31 / "pair_contrasts.csv").exists() else pd.DataFrame()
NAME = {"baseline_efficientnet_b3": "EfficientNet-B3", "baseline_resnet50": "ResNet-50", "retfound_linprobe": "RETFound LP",
        "retfound_full": "RETFound full", "baseline_efficientnet_b3_ls01": "EfficientNet-B3 + label smoothing (control)",
        "baseline_efficientnet_b3_rfrecipe": "EfficientNet-B3, RETFound recipe (control)",
        "retfound_full_cnnrecipe": "RETFound full, CNN recipe (control)"}
for s_ in (43, 44, 45):
    NAME[f"baseline_efficientnet_b3_seed{s_}"] = f"EfficientNet-B3, seed {s_}"
    NAME[f"retfound_linprobe_seed{s_}"] = f"RETFound LP, seed {s_}"
    NAME[f"retfound_full_seed{s_}"] = f"RETFound full, seed {s_}"
ORDER = [m for m in NAME if m in set(ci.model)]
SETS = ["APTOS-internal", "APTOS-full", "Messidor-2", "IDRiD"]
SETN = {"APTOS-internal": "APTOS internal (n = 692)", "APTOS-full": "APTOS internal incl. duplicates (n = 733)",
        "Messidor-2": "Messidor-2 (n = 1,744)", "IDRiD": "IDRiD official (n = 516)"}
GN = {"effb3": "EfficientNet-B3", "retfound_linprobe": "RETFound LP", "retfound_full": "RETFound full"}
MN = {"qwk": "QWK", "qwk_rw": "QWK, case mix matched", "qwk4": "QWK, grades 0+1 merged", "qwk4_rw": "QWK, both corrections", "ref_auc": "Referable-DR AUC"}
L = []
def f3(x): return "–" if pd.isna(x) else f"{x:.3f}"
def sg(x): return "–" if pd.isna(x) else ("0.000" if round(x, 3) == 0 else f"{x:+.3f}")
def pv(x): return "–" if pd.isna(x) else ("<0.001" if x < 0.001 else f"{x:.3f}")
def cell(m, s, k, pct=False):
    r = ci[(ci.model == m) & (ci.set == s) & (ci.metric == k)]
    if r.empty: return "–"
    r = r.iloc[0]
    out = f"{100*r.value:.1f} ({100*r.lo:.1f}–{100*r.hi:.1f})" if pct else f"{r.value:.3f} ({r.lo:.3f}–{r.hi:.3f})"
    return out.replace("-0.000", "0.000").replace("(-", "(−").replace("–-", "–−")
def table(head, rows):
    L.append("| " + " | ".join(head) + " |"); L.append("|" + "|".join(["---"] + ["--:"] * (len(head) - 1)) + "|")
    for r in rows: L.append("| " + " | ".join(str(x) for x in r) + " |")
    L.append("")

L += ["# S4 Messidor-2 examinations", "",
      f"Messidor-2 contains two macula-centred photographs (one per eye) per examination. Examinations were taken from the official pairing file distributed by ADCIS (messidor-2.csv; one row per examination, left and right image): {NV['pairsExams']} examinations, {NV['pairsN']} with both images gradable and {NV['pairsSingle']} with one gradable image. Fellow eyes had identical reference grades in {NV['pairsAgree']}% of examinations and grades within one level in {NV['pairsWithinOne']}%, against {NV['pairsAgreeRand']}% exact agreement expected for two random images. The within-examination correlation of EfficientNet-B3 grading errors was {NV['pairsErrCorr']}; all Messidor-2 intervals therefore resample whole examinations.", "",
      f"Before the pairing file was available, pairs had been reconstructed from file-name order and optic-disc position (`phase30_messidor_pairs.py`). Against the official file, {NV['recPPV']}% of reconstructed pairs were correct and laterality was inferred correctly for {NV['recLat']}% of images. All reported results use the official file.", ""]

L += ["# S5 Performance of every model and training run", "",
      "Point estimates with 95% bootstrap intervals (2,000 replicates; Messidor-2 resampled by examination). Sensitivity, specificity, PPV and NPV refer to referable DR (grade ≥ 2) at the model's own grade decision, in %. ECE, expected calibration error (15 bins).", ""]
for s in SETS:
    L.append(f"### {SETN[s]}"); L.append("")
    table(["Model", "QWK", "Accuracy, %", "Sensitivity, %", "Specificity, %", "PPV, %", "NPV, %", "Referable AUC", "ECE"],
          [[NAME[m], cell(m, s, "qwk"), cell(m, s, "acc", True), cell(m, s, "sens", True), cell(m, s, "spec", True),
            cell(m, s, "ppv", True), cell(m, s, "npv", True), cell(m, s, "ref_auc"), cell(m, s, "ece")] for m in ORDER])

L += ["# S6 Primary comparison: run-to-run variation, interval methods and the full internal set", "",
      "Excess drop = (internal − external QWK of the model family, averaged over its runs) − (same for EfficientNet-B3). *Conditional* intervals resample images (examinations for Messidor-2) with the observed runs fixed; *two-level* intervals also resample runs within each family (primary). Holm adjustment over the four RETFound contrasts.", ""]
for internal, title in [("APTOS-internal", "Deduplicated internal set (n = 692; primary)"), ("APTOS-full", "All 733 internal-test images (original plan)")]:
    L.append(f"### {title}"); L.append("")
    rows = []
    for _, r in sc[(sc.internal == internal) & (sc.metric == "qwk")].iterrows():
        per = ", ".join(f"{float(x):.3f}" for x in str(r.per_seed_drops).split(";")) if isinstance(r.per_seed_drops, str) else "–"
        rows.append([GN[r.group], r.external, int(r.n_seeds), per, f3(r["drop"]), f3(r.drop_sd),
                     "reference" if r.group == "effb3" else f"{sg(r.ddrop)} ({sg(r.ddrop_lo)} to {sg(r.ddrop_hi)})",
                     "–" if r.group == "effb3" else pv(r.p_holm),
                     "–" if r.group == "effb3" else f"{sg(r.h_lo)} to {sg(r.h_hi)}",
                     "–" if r.group == "effb3" else pv(r.h_p_holm)])
    table(["Family", "External", "Runs", "Drop per run", "Mean drop", "SD", "Excess drop (conditional 95% CI)", "Holm p", "Two-level 95% CI", "Holm p (two-level)"], rows)
    L.append("Single runs (seed 42) versus EfficientNet-B3 seed 42, and secondary contrasts:"); L.append("")
    rows = []
    for _, r in con[(con.internal == internal) & (con.metric == "qwk") & con.ddrop.notna()].iterrows():
        rows.append([NAME.get(r.model, r.model), r.external, f3(r["drop"]), f"{sg(r.ddrop)} ({sg(r.ddrop_lo)} to {sg(r.ddrop_hi)})", pv(r.p),
                     pv(r.p_holm) if "p_holm" in r and not pd.isna(r.p_holm) else "–"])
    table(["Model", "External", "Drop", "Excess drop vs EfficientNet-B3 (95% CI)", "p", "Holm p"], rows)

L += ["# S7 Matched-recipe controls", "",
      "Crossed-recipe experiment (seed 42): RETFound full fine-tuning with the CNN recipe compared with EfficientNet-B3 (CNN recipe), and RETFound full fine-tuning (RETFound recipe) compared with EfficientNet-B3 trained with the RETFound recipe. Recipes are given in Table 2 of the article.", ""]
if not pc_.empty:
    table(["Comparison", "External", "Metric", "Drop, RETFound", "Drop, EfficientNet-B3", "Excess drop (95% CI)", "p"],
          [[("RETFound vs EfficientNet-B3, both RETFound recipe" if r.tag == "xrf" else "RETFound vs EfficientNet-B3, both CNN recipe"),
            r.external, MN[r.metric], f3(r.drop_a), f3(r.drop_b), f"{sg(r.ddrop)} ({sg(r.ddrop_lo)} to {sg(r.ddrop_hi)})", pv(r.p)] for _, r in pc_.iterrows()])

L += ["# S8 Case mix and the no-DR/mild boundary", "",
      "Internal (n = 692) minus external QWK under each correction, seed-42 runs, and the difference from EfficientNet-B3. These corrections describe how the drop depends on case mix and on the no-DR/mild boundary; they do not identify its cause.", ""]
rows = []
for _, r in con[(con.internal == "APTOS-internal") & (con.metric != "ref_auc")].iterrows():
    rows.append([NAME.get(r.model, r.model), r.external, MN[r.metric], f"{f3(r['drop'])} ({f3(r.drop_lo)}–{f3(r.drop_hi)})",
                 "–" if pd.isna(r.ddrop) else f"{sg(r.ddrop)} ({sg(r.ddrop_lo)} to {sg(r.ddrop_hi)})"])
table(["Model", "External", "Measure", "Drop (95% CI)", "Excess drop vs EfficientNet-B3"], rows)
L.append("Referable-DR AUC compared with EfficientNet-B3 (paired bootstrap, Holm adjustment within dataset):"); L.append("")
table(["Model", "Dataset", "Difference (95% CI)", "p", "Holm p"],
      [[NAME.get(r.model, r.model), r.set, f"{sg(r.value)} ({sg(r.lo)} to {sg(r.hi)})", pv(r.p), pv(r.p_holm)]
       for _, r in oth[oth.kind == "ref_auc_vs_effb3"].iterrows()])

L += ["# S9 Uncertainty, error detection and referral", "",
      "Error-detection AUC (incorrect versus correct grade) for maximum softmax probability (MSP), top-two margin and predictive entropy, and the difference from MSP.", ""]
rows = []
for m in ORDER:
    for s in ["APTOS-internal", "Messidor-2", "IDRiD"]:
        def d(kind):
            r = oth[(oth.model == m) & (oth.set == s) & (oth.kind == kind)]
            return "–" if r.empty else f"{sg(r.iloc[0].value)} ({sg(r.iloc[0].lo)} to {sg(r.iloc[0].hi)})"
        rows.append([NAME[m], s, cell(m, s, "err_auc_msp"), cell(m, s, "err_auc_margin"), cell(m, s, "err_auc_entropy"),
                     d("err_auc_margin_minus_msp"), d("err_auc_entropy_minus_msp")])
table(["Model", "Dataset", "MSP", "Margin", "Entropy", "Margin − MSP", "Entropy − MSP"], rows)
L.append("Referral of the least confident images to a human grader: fraction of all referable-DR images missed by the system (retained and graded non-referable), %, and accuracy and QWK on retained images (MSP ranking). Random: fixed random referral order."); L.append("")
rows = []
for m in ORDER:
    for s in ["Messidor-2", "IDRiD", "APTOS-internal"]:
        rows.append([NAME[m], s, cell(m, s, "missed_0", True)] +
                    [cell(m, s, f"missed_{k}_{r}", True) for r in (10, 20, 30) for k in ("msp",)] +
                    [cell(m, s, "missed_entropy_30", True), cell(m, s, "missed_random_30", True),
                     cell(m, s, "acc_kept_msp_30", True), cell(m, s, "qwk_kept_msp_30")])
table(["Model", "Dataset", "Missed, no referral", "MSP 10%", "MSP 20%", "MSP 30%", "Entropy 30%", "Random 30%", "Accuracy retained (30%)", "QWK retained (30%)"], rows)
(OUT / "esm1_generated.md").write_text("\n".join(L), encoding="utf-8")
print("lines", len(L))
