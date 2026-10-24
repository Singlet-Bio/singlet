import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *

def eval_presence(test, mem, scheme, step=0.25, alpha=0.5, use_depth=True, verbose=False):
    org = ORG[test]; g, ng = gene_index(org, mem)
    # training pool: everything of same organism from other studies
    pool = [k for k in STUDY if ORG[k] == org and STUDY[k] != STUDY[test]]
    pool = [k for k in pool if k != test]
    names = [f'{k}__{mem}' for k in pool]
    m = get(names[0])['m']
    nb = scheme[1] if use_depth else 1
    def bsch(d): return bucketize(d, scheme) if use_depth else np.zeros(d['n'], dtype=np.int32)
    s_all = profile(names, m); c_all, nc = make_classes(s_all, g, ng, step)
    # calibrate P with leave-one-study-out inside the pool
    Acal = np.zeros((nb, nc)); Ncal = np.zeros(nb)
    studies = sorted(set(STUDY[k] for k in pool))
    for st in studies:
        inn = [f'{k}__{mem}' for k in pool if STUDY[k] != st]; out = [f'{k}__{mem}' for k in pool if STUDY[k] == st]
        if not inn: continue
        s_in = profile(inn, m); c_in, _ = make_classes(s_in, g, ng, step)
        mc_in = np.bincount(c_in, minlength=nc).astype(np.float64)
        for nm in out:
            d = get(nm); A, Nb = counts_AN(d, bsch(d), c_in, nb, nc)
            Acal += A; Ncal += Nb
            # note: A in terms of class assignment c_in; accumulate absent counts via mc_in
            if 'Mcal' not in locals(): Mcal = np.zeros((nb, nc))
            Mcal += Nb[:, None] * mc_in[None, :]
    P = (Acal + alpha * 0.001) / (Mcal + alpha * 0.001 * 2 + 1e-9)
    P = np.clip(P, 1e-7, 0.999)
    d = get(f'{test}__{mem}'); b = bsch(d)
    A, Nb = counts_AN(d, b, c_all, nb, nc); mc = np.bincount(c_all, minlength=nc).astype(np.float64)
    bits = presence_cost_bits(A, Nb, mc, P)
    bb = entropy_bits(np.bincount(b, minlength=nb).astype(np.float64)) if use_depth else 0
    return bits / len(d['indices']), bb / len(d['indices']), len(d['indices'])

if __name__ == '__main__':
    mem = sys.argv[1] if len(sys.argv) > 1 else 'exon'
    schemes = [('none', 1, 0, 1), ('tot', 16, 0, 16), ('tot', 32, 0, 16), ('nz', 16, 0, 14), ('nz', 32, 0, 14), ('nz', 56, 0, 14)]
    for t in TESTS:
        out = []
        for sc in schemes:
            use = sc[0] != 'none'
            pb, bb, nnz = eval_presence(t, mem, sc, use_depth=use)
            out.append(f"{sc[0]}{sc[1]}: {pb:.3f}+{bb:.3f}={pb+bb:.3f}")
        print(f"{t:22s} {mem:6s} ", ' | '.join(out), flush=True)
