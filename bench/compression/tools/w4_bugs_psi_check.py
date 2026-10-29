"""w4_bugs item 1: shipped splice_psi.1pz vs production compute_psi (C++ driver w4_bugs_psi) vs numpy recipe.
usage: python -P tools/w4_bugs_psi_check.py GSM4317429   (needs bin/w4_bugs_psi.exe output in corpus_w4_bugs/)
"""
import sys, numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz
gsm = sys.argv[1] if len(sys.argv) > 1 else 'GSM4317429'
D = f'corpus_w4_bugs/GSE145370/samples/{gsm}/'
P = f'corpus_w4_bugs/psi_{gsm}'
ship = read_1pz(D + 'splice_psi.1pz')
sj = read_1pz(D + 'sj_counts.1pz')
intree = read_1pz(P + '.intree.1pz')
f32 = np.fromfile(P + '.psi_f32.bin', dtype=np.float32)
idx = np.fromfile(P + '.psi_idx.bin', dtype=np.int32)
ptr = np.fromfile(P + '.psi_ptr.bin', dtype=np.int32)
print(gsm, 'shipped: m,n,nnz,vt', ship['m'], ship['n'], ship['nnz'], ship['vt_code'], ' sj nnz', sj['nnz'])
assert (ship['indptr'] == ptr).all() and (ship['indices'] == idx).all(), 'structure differs from compute_psi'
sv = np.asarray(ship['data']); tv = np.asarray(intree['data'])
print('shipped value histogram:', dict(zip(*np.unique(sv, return_counts=True))))
print('in-tree  value histogram:', dict(zip(*np.unique(tv, return_counts=True))))
print('PSI==1.0 exactly: %.4f' % (f32 == 1.0).mean(), ' PSI==0.5: %d' % (f32 == 0.5).sum(),
      ' PSI>0.5<1: %d' % ((f32 > 0.5) & (f32 < 1)).sum(), ' PSI<0.5: %d' % (f32 < 0.5).sum())
for name, fn in [('trunc', np.trunc), ('rint(half-even)', np.rint), ('floor(x+.5)(half-up)', lambda x: np.floor(x + 0.5)),
                 ('ceil', np.ceil)]:
    v = fn(f32).astype(np.uint32)
    print('  match shipped with %-22s %d / %d  (mismatch %d = %.3f%%)' % (name, (v == sv).sum(), len(sv), (v != sv).sum(), 100 * (v != sv).mean()))
print('  in-tree writer (file produced by pz_writer.h today) vs trunc: equal =', (tv == np.trunc(f32).astype(np.uint32)).all())
mm = sv != tv
print('shipped != in-tree writer: %d / %d = %.3f%%' % (mm.sum(), len(sv), 100 * mm.mean()))
print('   of those, PSI range:', f32[mm].min(), f32[mm].max(), ' all shipped=1,intree=0:', ((sv[mm] == 1) & (tv[mm] == 0)).all())
# is any real fractional information left? sum of squares of information: number of distinct values
print('distinct stored values shipped/intree:', len(np.unique(sv)), len(np.unique(tv)), ' distinct float PSI:', len(np.unique(f32)))
