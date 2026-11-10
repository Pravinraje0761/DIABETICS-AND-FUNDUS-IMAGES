"""
Phase 30 - reconstruct Messidor-2 examination pairs (two eyes per examination)
without the ADCIS pairing spreadsheet, and validate the reconstruction.

Rules (no labels or predictions are used):
  * Laterality of each image from the optic-disc position: the disc is the brightest
    large blob; in a macula-centred photograph it lies right of centre in one eye and
    left of centre in the other.
  * Messidor-Extension (IM######.JPG): candidate pairs are consecutive file numbers.
  * Messidor-Original (date_time_dept_PP.png): candidate pairs are consecutive images
    within the same date and department, ordered by the time field.
  * A candidate is accepted only if the two images have OPPOSITE laterality.
Output: outputs/phase30/messidor2_pairs.csv (image -> exam_id, laterality, paired flag)
        outputs/phase30/pairing_report.md
"""
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs/phase30"; OUT.mkdir(parents=True,exist_ok=True)
man=pd.read_csv(ROOT/"data_splits/messidor2_manifest.csv")
def disc_x(p):
    a=np.asarray(Image.open(ROOT/p).convert("RGB"),dtype=float).mean(2)
    m=a>15; a[~m]=0; b=gaussian_filter(a,6)
    y,x=np.unravel_index(np.argmax(b),b.shape)
    xs=np.where(m.any(0))[0]; c=(xs.min()+xs.max())/2; w=(xs.max()-xs.min())/2
    return (x-c)/w
rows=[]
for _,r in man.iterrows():
    rows.append(dict(id_code=r.id_code, disc=disc_x(r.processed_path)))
d=pd.DataFrame(rows)
d["lat"]=np.where(d.disc>0,"R-side","L-side"); d["conf"]=d.disc.abs()
d["ext"]=d.id_code.str.startswith("IM")
pairs=[]; used=set()
# Extension
e=d[d.ext].copy(); e["num"]=e.id_code.str[2:8].astype(int); e=e.sort_values("num")
nums=e.num.tolist(); ids=e.id_code.tolist(); lat=dict(zip(d.id_code,d.lat))
i=0
while i<len(ids)-1:
    if nums[i+1]-nums[i]==1 and lat[ids[i]]!=lat[ids[i+1]]:
        pairs.append((ids[i],ids[i+1],"extension")); used|={ids[i],ids[i+1]}; i+=2
    else: i+=1
# Original
o=d[~d.ext].copy(); parts=o.id_code.str.split("_",expand=True)
o["date"]=parts[0]; o["t"]=parts[1].astype(int); o["dept"]=parts[2]
for (dt,dp),g in o.groupby(["date","dept"]):
    g=g.sort_values("t"); ids=g.id_code.tolist(); i=0
    while i<len(ids)-1:
        if lat[ids[i]]!=lat[ids[i+1]]:
            pairs.append((ids[i],ids[i+1],"original")); used|={ids[i],ids[i+1]}; i+=2
        else: i+=1
ex={}
for k,(a,b,src) in enumerate(pairs): ex[a]=ex[b]=f"E{k:04d}"
d["exam_id"]=d.id_code.map(lambda x: ex.get(x,"S_"+x)); d["paired"]=d.id_code.isin(used)
d.to_csv(OUT/"messidor2_pairs.csv",index=False)
n_e=(d.ext).sum(); n_o=(~d.ext).sum()
pe=sum(1 for p in pairs if p[2]=="extension"); po=sum(1 for p in pairs if p[2]=="original")
lab=pd.read_csv(ROOT/"data_splits/messidor2_manifest.csv").set_index("id_code").diagnosis
pr=[(lab[a],lab[b]) for a,b,_ in pairs]; same=np.mean([x==y for x,y in pr]); within1=np.mean([abs(x-y)<=1 for x,y in pr])
# chance agreement baseline: random pairs
rng=np.random.default_rng(0); allg=lab.values; rs=np.mean(rng.choice(allg,5000)==rng.choice(allg,5000))
rep=f"""# Phase 30 - Messidor-2 examination pairs reconstructed

- Images: {len(d)} (Extension {n_e}, Original {n_o}). Expected examinations: 874 (Extension 345, Original 529; 4 ungradable images excluded).
- Pairs accepted: {len(pairs)} (Extension {pe}, Original {po}); images paired: {int(d.paired.sum())} ({100*d.paired.mean():.1f}%).
- Laterality split: {int((d.lat=='R-side').sum())} R-side / {int((d.lat=='L-side').sum())} L-side; median |disc offset| {d.conf.median():.2f} of half-width.
- Validation (labels not used for pairing): fellow-eye grade agreement in accepted pairs {100*same:.1f}% (exact), {100*within1:.1f}% within one grade, vs {100*rs:.1f}% exact agreement for random image pairs.
"""
(OUT/"pairing_report.md").write_text(rep); print(rep)
