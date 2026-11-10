"""Phase 29c - paired DeLong test (DeLong 1988; fast midrank algorithm, Sun & Xu 2014)
for referable-DR AUC (P(grade>=2)), each model vs EfficientNet-B3, official IDRiD; Holm over the 9 contrasts."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/"outputs/phase29/preds_official"; OUT=ROOT/"outputs/phase29/delong"
def midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2
def delong(y, s1, s2):
    pos=y==1; m=pos.sum(); n=(~pos).sum(); aucs=[]; V10=[]; V01=[]
    for s in (s1,s2):
        X=s[pos]; Y=s[~pos]; tx=midrank(X); ty=midrank(Y); tz=midrank(np.r_[X,Y])
        auc=(tz[:m].sum()-m*(m+1)/2)/(m*n); aucs.append(auc)
        V10.append((tz[:m]-tx)/n); V01.append(1-(tz[m:]-ty)/m)
    S10=np.cov(np.vstack(V10)); S01=np.cov(np.vstack(V01)); S=S10/m+S01/n
    d=aucs[0]-aucs[1]; var=S[0,0]+S[1,1]-2*S[0,1]; z=d/np.sqrt(var)
    return aucs[0],aucs[1],d,d-1.96*np.sqrt(var),d+1.96*np.sqrt(var),2*norm.sf(abs(z))
rows=[]
for d in ["APTOS-internal","Messidor-2","IDRiD"]:
    b=pd.read_csv(P/f"baseline_efficientnet_b3__{d}_preds.csv"); y=(b.y_true>=2).astype(int).to_numpy()
    sb=b[["prob_2","prob_3","prob_4"]].sum(1).to_numpy()
    for m,l in [("baseline_resnet50","ResNet-50"),("retfound_linprobe","RETFound linear probe"),("retfound_full","RETFound full fine-tune")]:
        x=pd.read_csv(P/f"{m}__{d}_preds.csv"); assert (x.id_code==b.id_code).all()
        sm=x[["prob_2","prob_3","prob_4"]].sum(1).to_numpy()
        a1,a0,dd,lo,hi,p=delong(y,sm,sb); rows.append(dict(dataset=d,model=l,auc_model=a1,auc_effb3=a0,diff=dd,lo=lo,hi=hi,p=p))
df=pd.DataFrame(rows); o=np.argsort(df.p.values); k=len(df); adj=np.zeros(k); run=0
for r,i in enumerate(o): run=max(run,min(1,(k-r)*df.p.values[i])); adj[i]=run
df["p_holm"]=adj; df.to_csv(OUT/"delong_referable_official.csv",index=False); print(df.round(4).to_string())
