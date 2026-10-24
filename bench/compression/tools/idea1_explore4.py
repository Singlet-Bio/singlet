import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idea1_lib import *
EDGES_B = np.array([0,1,2,4,7,12,20,34,58,100,170,290,500,850,1450,2500,10**9])
CMAX = 16   # symbols 1..15, 16 = >=16
def count_hist(d, cls, edges):
    D,k,b = col_info(d, edges); bn = nnz_bucket(d,b)
    c = np.minimum(d['data'], CMAX).astype(np.int64) - 1
    ncls = cls.max()+1
    key = (cls[d['indices']]*NB + bn)*CMAX + c
    return np.bincount(key, minlength=ncls*NB*CMAX).reshape(ncls, NB, CMAX).astype(np.float64), np.maximum(d['data']-CMAX+1,1)
def escape_bits(data):
    # exp-golomb of (c-16) for c>=16
    v = data[data>=CMAX]-CMAX+1
    return (2*np.floor(np.log2(v))+1).sum() if len(v) else 0.0
def run(sid, mem, edges):
    d = load(sid, mem); nnz=len(d['indices'])
    tr = train_set(sid, mem)
    n, N = pool(tr, mem, edges)
    tot = n.sum(1)
    # feature classes by log rate
    def mkcls(nc):
        r = np.log10((tot+0.5))
        cl = np.clip(np.floor(r/ (np.log10(tot.max()+1)/ (nc-1))).astype(int), 0, nc-1)
        cl[tot==0] = 0
        return cl
    out={}
    # training count hist
    for nc in (1, 4, 8, 16):
        cls = mkcls(nc) if nc>1 else np.zeros(d['m'],int)
        H = None
        for t in tr:
            dd = load(t, mem); h,_ = count_hist(dd, cls, edges)
            H = h if H is None else H+h
        for dep in (False, True):
            HH = H.sum(1,keepdims=True).repeat(NB,1) if not dep else H
            P = (HH+0.05)/(HH+0.05).sum(2,keepdims=True)
            h,_ = count_hist(d, cls, edges)
            bits = -(h*np.log2(P)).sum() + escape_bits(d['data'])
            out[f'cls{nc}{"xd" if dep else ""}'] = bits/nnz
    # entropy of order 0
    c = np.minimum(d['data'],CMAX)-1
    pc = np.bincount(c, minlength=CMAX)/nnz; pc=pc[pc>0]
    out['H0'] = -(pc*np.log2(pc)).sum()
    return out
for sid, mem in [('pbmc_v2_typical','exon'),('tcell_5p','exon'),('brain_sn_v3_deep','exon'),('brain_sn_v3_deep','intron')]:
    print(sid, mem, {k: round(float(v),3) for k,v in run(sid, mem, EDGES_B).items()}, flush=True)
