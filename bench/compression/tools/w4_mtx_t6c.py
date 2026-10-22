import os, sys, numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE); CORP = os.path.join(BC, "corpus_w4_mtx")
d = "GSE309702.GSM9277757"; p = os.path.join(CORP, d)
G = [l.split("\t") for l in open(os.path.join(p, "gene_expression.tsv")).read().split("\n")[1:-1]]
el = np.array([float(g[2]) for g in G]); cnt = np.array([float(g[3]) for g in G]); tpm = np.array([float(g[4]) for g in G]); fpkm = np.array([float(g[5]) for g in G])
rate = cnt / el; Sr = rate.sum(); T = cnt.sum()
print("c_obs=", (fpkm[cnt > 0] / tpm[cnt > 0]).mean(), " 1e3*Sr/T=", 1e3 * Sr / T, " Sr/T*1e3 ; also Sr/1e3?", Sr / 1e3)
tp = rate / Sr * 1e6; fp = cnt * 1e9 / (el * T)
raw = open(os.path.join(p, "gene_expression.tsv")).read().split("\n")
regen = [raw[0]] + ["%s\t%s\t%d\t%d\t%.6f\t%.6f" % (g[0], g[1], int(float(g[2])), int(float(g[3])), a, b) for g, a, b in zip(G, tp, fp)]
ok = sum(1 for x, y in zip(regen[1:], raw[1:]) if x == y)
print("rows byte-identical with TPM=rate/sum*1e6 (%.6f), FPKM=cnt*1e9/(len*T) (%.6f):", ok, "of", len(G))
print("example", raw[1:3], regen[1:3])
# with FPKM = rate*1e3/(T/1e6) 
# cell_qc
r = read_1pz(os.path.join(p, "exon_counts.1pz")); ri = read_1pz(os.path.join(p, "intron_counts.1pz"))
Q = [l.split("\t") for l in open(os.path.join(p, "cell_qc_metrics.tsv")).read().split("\n")[1:-1]]
ng = np.array([int(q[2])for q in Q]); ip_pct = np.array([float(q[5]) for q in Q]); nr = np.array([int(q[6]) for q in Q])
ip = r["indptr"].astype(np.int64); ipi = ri["indptr"].astype(np.int64)
gid_e = np.array([hash(n.split("_")[0]) for n in r["rownames"]]); gid_i = np.array([hash(n.split("_")[0]) for n in ri["rownames"]])
ce = np.array([r["data"][ip[j]:ip[j+1]].sum() for j in range(r["n"])]); ci = np.array([ri["data"][ipi[j]:ipi[j+1]].sum() for j in range(ri["n"])])
ngen = np.empty(r["n"], dtype=np.int64)
for j in range(r["n"]):
    s = set(gid_e[r["indices"][ip[j]:ip[j+1]]].tolist()) | set(gid_i[ri["indices"][ipi[j]:ipi[j+1]]].tolist()); ngen[j] = len(s)
print("n_genes == #distinct genes with exon or intron count:", int((ngen == ng).sum()), "of", len(ng))
pct = np.where(ce + ci > 0, 100.0 * ci / (ce + ci), 0)
print("intronic_pct == round(100*intron/(exon+intron),4):", int((np.abs(pct - ip_pct) < 5.1e-5).sum()), "of", len(ng))
print("n_reads vs n_umi: corr", np.corrcoef(nr, ce + ci)[0, 1], " dup rate = 1-umi/reads?", Q[0][7], 1 - int(Q[0][1]) / int(Q[0][6]))
