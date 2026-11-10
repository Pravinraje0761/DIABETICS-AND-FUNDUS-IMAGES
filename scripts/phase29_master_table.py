"""Phase 29e - master table on the official IDRiD release: QWK [95% CI], accuracy, macro-F1,
balanced accuracy, macro OvR AUC per model x dataset (bootstrap B=2000, seed 42)."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/"outputs/phase29/preds_official"; OUT=ROOT/"outputs/phase29/master"; OUT.mkdir(parents=True,exist_ok=True)
def qwk(y,h,k=5):
    O=np.zeros((k,k)); np.add.at(O,(y,h),1); W=(np.subtract.outer(np.arange(k),np.arange(k))**2)/(k-1)**2
    E=np.outer(O.sum(1),O.sum(0))/O.sum(); return 1-(W*O).sum()/(W*E).sum()
rows=[]; rng=np.random.default_rng(42)
for m in ["baseline_efficientnet_b3","baseline_resnet50","retfound_linprobe","retfound_full"]:
  for d in ["APTOS-internal","Messidor-2","IDRiD"]:
    p=pd.read_csv(P/f"{m}__{d}_preds.csv"); y=p.y_true.to_numpy(); h=p.y_pred.to_numpy(); Pr=p[[f"prob_{k}" for k in range(5)]].to_numpy()
    bs=[qwk(y[s],h[s]) for s in rng.integers(0,len(y),(2000,len(y)))]
    rows.append(dict(model=m,dataset=d,n=len(y),qwk=qwk(y,h),qwk_lo=np.percentile(bs,2.5),qwk_hi=np.percentile(bs,97.5),
      accuracy=(y==h).mean(),macro_f1=f1_score(y,h,average="macro"),balanced_acc=balanced_accuracy_score(y,h),
      auc_ovr=roc_auc_score(y,Pr,multi_class="ovr",average="macro")))
df=pd.DataFrame(rows); df.to_csv(OUT/"master_official.csv",index=False); print(df.round(3).to_string())
