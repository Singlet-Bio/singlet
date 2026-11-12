import numpy as np, pandas as pd
W="/mnt/projects/debruinz_project/debruinz/singlet"; O=W+"/bench/calls-sota/sex"
e=pd.read_parquet(O+"/evidence/esc_dev.parquet")
s=pd.read_csv(W+"/bpub/artifacts/chromosomal-sex/sex_samples.tsv",sep="\t"); tr=dict(zip(s.gsm,s.truth))
rows=[]
for n in ("lcl_pool","osorio"):
    ev=pd.read_parquet(O+f"/evidence/{n}.parquet")[["col","unit","truth"]]
    d=e[e["sample"]==n].merge(ev,on="col"); d["u"]=n+":"+d.unit.astype(str); rows.append(d)
t2=e[~e["sample"].isin(["lcl_pool","osorio"])].copy(); t2["truth"]=t2["sample"].map(tr); t2["u"]=t2["sample"]; t2["tier"]=2
t1=pd.concat(rows); t1["tier"]=1
d=pd.concat([t1,t2]); d=d[d.truth.isin(["female","male"])]
g=d.groupby("u").agg(tier=("tier","first"),truth=("truth","first"),n=("x","size"),total=("total","sum"),esc=("esc","sum"),x=("x","sum"),y=("y","sum"),med_total=("total","median"))
g=g[g.n>=200]; g["esc_per10k"]=g.esc/g.total*1e4; g["x_pc"]=g.x/g.n; g["y_pc"]=g.y/g.n; g["y_per10k"]=g.y/g.total*1e4; g["x_per10k"]=g.x/g.total*1e4
def auc(a,b):
    a=np.asarray(a);b=np.asarray(b); return float((a[:,None]>b[None,:]).mean()+0.5*(a[:,None]==b[None,:]).mean())
for t in (1,2):
    h=g[g.tier==t]; f=h[h.truth=="female"]; m=h[h.truth=="male"]
    print(f"tier{t}: F {len(f)} M {len(m)}")
    for c in ("esc_per10k","x_per10k","y_per10k"):
        print(f"  {c}: female median {f[c].median():.2f} (5-95% {f[c].quantile(.05):.2f}-{f[c].quantile(.95):.2f}) male median {m[c].median():.2f} ({m[c].quantile(.05):.2f}-{m[c].quantile(.95):.2f}) AUC F>M {auc(f[c],m[c]):.3f}")
g.to_csv(O+"/esc_dev_groups.tsv",sep="\t")
# XIST-silent females in dev: x_per10k lowest females
print(g[g.truth=="female"].sort_values("x_per10k").head(8)[["tier","n","x_per10k","y_per10k","esc_per10k"]].to_string())
print(g[g.truth=="male"].sort_values("y_per10k").head(8)[["tier","n","x_per10k","y_per10k","esc_per10k"]].to_string())
