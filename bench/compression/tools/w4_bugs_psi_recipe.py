"""w4_bugs items 1+3: independent numpy/Python implementation of the PSI recipe (splice_psi.h) and the 'donor events take priority'
variant promised by the comment at splice_psi.h:105-106.  Compared bit-for-bit against the float32 output of the production C++
compute_psi() (corpus_w4_bugs/psi_head_<gsm>.psi_f32.bin from bin/w4_bugs_psi.exe) and against the shipped member.
usage: python -P tools/w4_bugs_psi_recipe.py [GSM ...]"""
import os, re, sys, json
import numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz
root = 'corpus_w4_bugs/GSE145370/samples/'

def parse(name):
    # splice_psi.h:40-56: first ':' , first '-' after it, stoul() on both pieces (digits prefix), cast uint32
    c = name.find(':')
    if c < 0: return None
    d = name.find('-', c + 1)
    if d < 0: return None
    m1 = re.match(r'\s*[+-]?(\d+)', name[c + 1:d]); m2 = re.match(r'\s*[+-]?(\d+)', name[d + 1:])
    if not m1 or not m2: return None
    return name[:c], int(m1.group(1)) & 0xFFFFFFFF, int(m2.group(1)) & 0xFFFFFFFF

def build_events(names):
    don, acc = {}, {}
    for i, n in enumerate(names):
        p = parse(n)
        if p is None: continue
        ch, s, e = p
        don.setdefault(f'{ch}:{s}-D', []).append(i); acc.setdefault(f'{ch}:{e}-A', []).append(i)
    ev = [(k, v) for k, v in don.items() if len(v) >= 2] + [(k, v) for k, v in acc.items() if len(v) >= 2]
    ev.sort(key=lambda t: t[0].encode())     # std::string operator< is bytewise
    return ev

def assign_first(ev, m):
    je = np.full(m, -1, np.int64)
    for e, (_, js) in enumerate(ev):
        for j in js:
            if je[j] == -1: je[j] = e
    return je

def assign_donor_first(ev, m):
    je = np.full(m, -1, np.int64)
    order = sorted(range(len(ev)), key=lambda e: (not ev[e][0].endswith('-D'), ev[e][0].encode()))  # all donors, then acceptors
    for e in order:
        for j in ev[e][1]:
            if je[j] == -1: je[j] = e
    return je

def psi_f32(sj, je):
    n, ind, ptr, dat = sj['n'], np.asarray(sj['indices']), np.asarray(sj['indptr']), np.asarray(sj['data']).astype(np.float64)
    col = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    e = je[ind]; keep = e >= 0
    col, e, ind, dat = col[keep], e[keep], ind[keep], dat[keep]
    key = col * (je.max() + 1) + e
    uk, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, weights=dat)[inv]
    assert s.max() < 2**24
    return (dat.astype(np.float32) / s.astype(np.float32)), ind, col

if __name__ == '__main__':
    gsms = sys.argv[1:] or sorted(os.listdir(root))
    out = []
    for gsm in gsms:
        D = root + gsm + '/'
        sj = read_1pz(D + 'sj_counts.1pz'); ship = read_1pz(D + 'splice_psi.1pz')
        names = sj['rownames']; m = sj['m']
        ev = build_events(names)
        je1, je2 = assign_first(ev, m), assign_donor_first(ev, m)
        f1, ind1, col1 = psi_f32(sj, je1)
        f2, ind2, col2 = psi_f32(sj, je2)
        prod = np.fromfile(f'corpus_w4_bugs/psi_head_{gsm}.psi_f32.bin', dtype=np.float32)
        # the python recipe lists entries column-major in the sj order restricted to assigned junctions = same order as C++
        same_bits = bool(len(prod) == len(f1) and (prod.view(np.uint32) == f1.view(np.uint32)).all())
        both = (je1 >= 0)
        differ_j = int((je1 != je2).sum())
        # junctions where the two policies pick different events
        dj = np.flatnonzero(je1 != je2)
        # effect on entries: entries whose junction is in dj
        in_dj = np.isin(ind1, dj)
        # compare per-entry PSI between policies when the structures are the same (they are: same set of assigned junctions, different sums)
        sh = np.asarray(ship['data'])
        r = dict(gsm=gsm, n_junctions=m, n_events=len(ev), n_assigned=int(both.sum()),
                 numpy_vs_cpp_bit_identical=same_bits, n_entries=len(f1),
                 junctions_policy_differs=differ_j, entries_on_those_junctions=int(in_dj.sum()))
        if differ_j:
            same_struct = len(f2) == len(f1) and (ind1 == ind2).all() and (col1 == col2).all()
            r['same_structure'] = bool(same_struct)
            if same_struct:
                r['entries_psi_changed'] = int((f1 != f2).sum())
                r['entries_stored_value_changed_after_rint'] = int((np.rint(f1) != np.rint(f2)).sum())
                r['shipped_matches_sorted_first_rint'] = bool((np.rint(f1).astype(np.uint32) == sh).all())
                r['shipped_matches_donor_first_rint'] = bool((np.rint(f2).astype(np.uint32) == sh).all())
        r['names_of_differing_junctions'] = [names[j] for j in dj[:5]]
        r['events_chosen_sorted_first'] = [ev[je1[j]][0] for j in dj[:5]]
        r['events_chosen_donor_first'] = [ev[je2[j]][0] for j in dj[:5]]
        out.append(r); print(json.dumps(r))
    json.dump(out, open('results/w4_bugs_psi_recipe.json', 'w'), indent=1)
