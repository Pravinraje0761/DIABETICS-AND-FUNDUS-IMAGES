"""Phase 30b - official Messidor-2 examination pairs (ADCIS messidor-2.csv: one row per
examination, columns left;right) -> outputs/phase30/messidor2_pairs_official.csv, and validation of
the Phase 30 reconstruction against them (outputs/phase30/reconstruction_vs_official.md)."""
import json
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs/phase30"
o = pd.read_csv(ROOT / "data_splits/messidor2_official_pairs_adcis.csv", sep=";")
o.columns = [c.strip() for c in o.columns]; o = o.apply(lambda s: s.str.strip())
man = pd.read_csv(ROOT / "data_splits/messidor2_manifest.csv"); ids = set(man.id_code)
rows = []
for i, r in o.reset_index(drop=True).iterrows():
    for side in ("left", "right"):
        if r[side] in ids:
            rows.append(dict(id_code=r[side], exam_id=f"X{i:04d}", eye=side))
off = pd.DataFrame(rows)
n_img = off.groupby("exam_id").id_code.transform("size"); off["paired"] = n_img == 2
off.to_csv(OUT / "messidor2_pairs_official.csv", index=False)
rec = pd.read_csv(OUT / "messidor2_pairs.csv").set_index("id_code")
offi = off.set_index("id_code")
# pair-level agreement
rp = rec[rec.paired].groupby("exam_id").apply(lambda g: frozenset(g.index))
op = set(off[off.paired].groupby("exam_id").id_code.apply(frozenset))
correct = sum(p in op for p in rp)
# laterality: reconstructed disc side vs official eye (sign convention unknown: report best mapping)
j = rec.join(offi[["eye"]], how="inner")
a1 = np.mean((j.lat == "R-side") == (j.eye == "right")); acc_lat = max(a1, 1 - a1)
lab = man.set_index("id_code").diagnosis
pp = off[off.paired].groupby("exam_id").id_code.apply(list)
agree = np.mean([lab[a] == lab[b] for a, b in pp]); within1 = np.mean([abs(lab[a] - lab[b]) <= 1 for a, b in pp])
res = dict(official_exams=len(o), official_pairs_gradable=int(len(pp)), images=int(len(off)),
           singletons=int((~off.paired).sum()), reconstructed_pairs=int(len(rp)), reconstructed_correct=int(correct),
           reconstructed_ppv=correct / len(rp), official_pairs_recovered=correct / len(pp),
           laterality_accuracy=float(acc_lat), fellow_agree=float(agree), fellow_within1=float(within1))
json.dump(res, open(OUT / "reconstruction_vs_official.json", "w"), indent=1)
(OUT / "reconstruction_vs_official.md").write_text(
    "# Messidor-2: reconstructed versus official examination pairs\n\n" + "\n".join(f"- {k}: {v}" for k, v in res.items()) + "\n")
print(res)
