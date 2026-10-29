"""w4_reader: cProfile + perf_counter decomposition of the production singlet.load() path on a local bundle.

Usage: python -P tools/w4_reader_profile.py [bundle] [--reps N] [--sample GSM]
Production code is imported read-only from ../../python (compiled _pz.pyd as built on this laptop).
"""
import sys, os, time, io, cProfile, pstats, warnings, argparse, json, hashlib, zlib, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..', 'python')))
warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser()
ap.add_argument('bundle', nargs='?', default=os.path.join(HERE, '..', 'corpus', 'bundles', 'GSE145370.singlet'))
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--sample', default=None)
ap.add_argument('--profile-only', action='store_true')
a = ap.parse_args()
from singlet.bundle import SingletBundle
from singlet._integrity import verify_bundle
import singlet._pz as _pz

def T(f, *args, **kw):
    t = time.perf_counter(); r = f(*args, **kw); return time.perf_counter() - t, r

print('bundle', a.bundle, os.path.getsize(a.bundle), 'bytes')
b = SingletBundle.open(a.bundle)
gsms = b.gsm_ids
print('gsms', len(gsms))

def prof(fn, label, top_names):
    pr = cProfile.Profile(); pr.enable(); r = fn(); pr.disable()
    s = pstats.Stats(pr)
    rows = []
    for (fn_, ln, name), (cc, nc, tt, ct, callers) in s.stats.items():
        rows.append((fn_, ln, name, nc, tt, ct))
    total = sum(r_[4] for r_ in rows)
    print(f'--- {label}: profiled total {total*1e3:.0f} ms (tottime sum); top by tottime')
    for r_ in sorted(rows, key=lambda x: -x[4])[:22]:
        print(f'  {r_[4]*1e3:9.1f} ms tot {r_[5]*1e3:9.1f} ms cum  n={r_[3]:6d}  {os.path.basename(r_[0])}:{r_[1]}({r_[2]})')
    return r, rows

# warm
b.to_anndata(verbose=False)
if not a.profile_only:
    ts = []
    for _ in range(a.reps):
        t, ad_ = T(lambda: SingletBundle.open(a.bundle).to_anndata(verbose=False)); ts.append(t)
    print('to_anndata whole bundle (layers=True): reps', [f'{x:.3f}' for x in ts], 'min', f'{min(ts):.3f}', 's')
    ts = []
    for _ in range(a.reps):
        t, _r = T(lambda: SingletBundle.open(a.bundle).to_anndata(verbose=False, layers=False)); ts.append(t)
    print('to_anndata whole bundle (layers=False): reps', [f'{x:.3f}' for x in ts], 'min', f'{min(ts):.3f}', 's')
    print('adata', ad_.shape, 'nnz', ad_.X.nnz, 'X dtype', ad_.X.dtype, 'X fmt', ad_.X.format)
    t, _ = T(verify_bundle, a.bundle, hashes=True); print('verify_bundle(hashes=True)', f'{t:.3f}', 's')
    t, _ = T(verify_bundle, a.bundle, hashes=False); print('verify_bundle(hashes=False)', f'{t*1e3:.1f}', 'ms')
    # raw hash costs over the file
    data = open(a.bundle, 'rb').read()
    t, _ = T(lambda: hashlib.sha256(data).hexdigest()); print('sha256 whole file', f'{t:.3f}', 's', f'{len(data)/t/1e6:.0f} MB/s')
    t, _ = T(lambda: zlib.crc32(data)); print('zlib.crc32 whole file', f'{t:.3f}', 's', f'{len(data)/t/1e6:.0f} MB/s')
    del data

prof(lambda: SingletBundle.open(a.bundle).to_anndata(verbose=False), 'to_anndata whole bundle', None)

g = a.sample or gsms[3]
print('sample', g)
def one():
    bb = SingletBundle.open(a.bundle)
    mat, bcs = bb._load_gsm_gene_counts(g)
    sp_ = bb._gene_level_part(g, 'exon', bcs)
    un_ = bb._gene_level_part(g, 'intron', bcs)
    return mat, bcs
prof(one, f'one sample {g} (X + spliced + unspliced)', None)
