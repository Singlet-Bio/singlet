import os, sys, numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE); CORP = os.path.join(BC, "corpus_w4_mtx")
P = "star_Solo.out__Gene__"
for d in sys.argv[1:]:
    p = os.path.join(CORP, d)
    G = [l.split("\t") for l in open(os.path.join(p, "gene_expression.tsv")).read().split("\n")[1:-1]]
    gid = [g[0] for g in G]; cnt = np.array([float(g[3]) for g in G]); el = np.array([float(g[2]) for g in G]); tpm = np.array([float(g[4]) for g in G]); fpkm = np.array([float(g[5]) for g in G])
    print(f"[{d}] gene_expression rows {len(G)} count sum {cnt.sum():.0f} header={open(os.path.join(p,'gene_expression.tsv')).readline().strip()}")
    feat = [l.split("\t")[0] for l in open(os.path.join(p, P + "raw__features.tsv")).read().split("\n")[:-1]]
    gidx = {g: i for i, g in enumerate(feat)}
    print("   gene_expression gene order == features order:", gid == feat, "; sorted by id:", gid == sorted(gid), "; in features:", sum(g in gidx for g in gid), "/", len(gid))
    # TPM / FPKM from count and effective length
    rate = cnt / np.maximum(el, 1e-9); tpm_p = rate / rate.sum() * 1e6; fpkm_p = cnt / (el / 1e3) / (cnt.sum() / 1e6)
    print("   TPM  = count/len/sum(count/len)*1e6: max abs err", np.abs(tpm_p - tpm).max(), " max rel", np.max(np.abs(tpm_p - tpm) / np.maximum(tpm, 1e-9)))
    print("   FPKM = count*1e9/(len*total)       : max abs err", np.abs(fpkm_p - fpkm).max(), " max rel", np.max(np.abs(fpkm_p - fpkm) / np.maximum(fpkm, 1e-9)))
    # counts versus sources
    b = open(os.path.join(p, P + "raw__matrix.mtx"), "rb").read().split(b"\n", 3)[3]
    T = np.fromstring(b, dtype=np.int64, sep=" ").reshape(-1, 3)
    mrow = np.bincount(T[:, 0] - 1, weights=T[:, 2], minlength=len(feat))
    fb = open(os.path.join(p, P + "filtered__matrix.mtx"), "rb").read().split(b"\n", 3)[3]
    TF = np.fromstring(fb, dtype=np.int64, sep=" ").reshape(-1, 3)
    frow = np.bincount(TF[:, 0] - 1, weights=TF[:, 2], minlength=len(feat))
    cnt_f = np.array([cnt[gid.index(f)] if False else 0 for f in []])
    order = np.array([gidx[g] for g in gid])
    for tag, arr in (("raw mtx row sums", mrow), ("filtered mtx row sums", frow)):
        a = arr[order]; print(f"   count vs {tag}: sum {a.sum():.0f}; equal rows {int((a == cnt).sum())}/{len(cnt)}; corr {np.corrcoef(a, cnt)[0,1]:.4f}")
    for kind in ("exon", "intron"):
        r = read_1pz(os.path.join(p, f"{kind}_counts.1pz"))
        rg = np.array([gidx[n.split("_")[0]] for n in r["rownames"]])
        s = np.bincount(rg, weights=np.bincount(r["indices"].astype(np.int64), weights=r["data"].astype(np.float64), minlength=r["m"])[:], minlength=len(feat)) if False else None
        rs = np.bincount(r["indices"].astype(np.int64), weights=r["data"].astype(np.float64), minlength=r["m"])
        s = np.bincount(rg, weights=rs, minlength=len(feat))[order]
        print(f"   count vs {kind} 1pz gene sums (all columns): sum {s.sum():.0f}; equal rows {int((s == cnt).sum())}/{len(cnt)}; corr {np.corrcoef(s, cnt)[0,1]:.4f}")
    # cell_qc
    Q = [l.split("\t") for l in open(os.path.join(p, "cell_qc_metrics.tsv")).read().split("\n")[1:-1]]
    qb = [q[0] for q in Q]; nu = np.array([int(q[1]) for q in Q]); ng = np.array([int(q[2]) for q in Q])
    fbc = open(os.path.join(p, P + "filtered__barcodes.tsv")).read().split("\n")[:-1]
    print(f"   cell_qc rows {len(Q)} vs filtered {len(fbc)}; same set: {set(qb)==set(fbc)}; same order: {qb==fbc}")
    col = {b_: i for i, b_ in enumerate(fbc)}
    cs = np.bincount(np.searchsorted(np.arange(1, 1 + len(fbc)), TF[:, 1]) , weights=TF[:, 2], minlength=len(fbc) + 1)
    cs = np.bincount(TF[:, 1] - 1, weights=TF[:, 2], minlength=len(fbc)); gn = np.bincount(TF[:, 1] - 1, minlength=len(fbc))
    idx = np.array([col[x] for x in qb if x in col])
    ok = len(idx) == len(qb)
    if ok:
        print(f"   n_umi == filtered mtx column sum: {int((cs[idx]==nu).sum())}/{len(nu)}; n_genes == filtered mtx col nnz: {int((gn[idx]==ng).sum())}/{len(ng)}")
        print("   corr n_umi vs col sum", np.corrcoef(cs[idx], nu)[0, 1])
    print("   cell_qc columns:", open(os.path.join(p, "cell_qc_metrics.tsv")).readline().strip())
