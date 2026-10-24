"""w4_reader: prototype threaded zero-copy decoder (bin/w4_reader_dll.dll via ctypes) vs the production pyd read_1pz.

Checks the outputs are identical to the production arrays, then times (min of N reps, timing lock must be held by caller).
Usage: python -P tools/w4_reader_proto.py [--reps 5] [--check-only] files...
"""
import sys, os, time, ctypes, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, '..', '..', 'python')))
import numpy as np
import singlet._pz as pz

ap = argparse.ArgumentParser()
ap.add_argument('files', nargs='*')
ap.add_argument('--reps', type=int, default=5)
ap.add_argument('--check-only', action='store_true')
ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'w4_reader_proto.tsv'))
a = ap.parse_args()
os.add_dll_directory(os.path.join(ROOT, 'bin')) if hasattr(os, 'add_dll_directory') else None
L = ctypes.CDLL(os.path.join(ROOT, 'bin', 'w4_reader_dll.dll'))
c_p = ctypes.c_void_p
L.w4r_open.restype = c_p
L.w4r_open.argtypes = [c_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
L.w4r_shape.argtypes = [c_p, c_p]
L.w4r_indptr.argtypes = [c_p, c_p]
L.w4r_decode.restype = ctypes.c_int
L.w4r_decode.argtypes = [c_p, ctypes.c_int64, ctypes.c_int64, c_p, c_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
L.w4r_close.argtypes = [c_p]


def proto(buf, threads, packed_sort, kind=4, verify_file=0):
    err = ctypes.create_string_buffer(256)
    h = L.w4r_open(buf.ctypes.data, buf.size, verify_file, err, 256)
    if not h:
        raise RuntimeError(err.value.decode())
    try:
        sh = np.zeros(6, dtype=np.uint64)
        L.w4r_shape(h, sh.ctypes.data)
        m, n, nnz = int(sh[0]), int(sh[1]), int(sh[2])
        indptr = np.empty(n + 1, dtype=np.int32)
        L.w4r_indptr(h, indptr.ctypes.data)
        indices = np.empty(nnz, dtype=np.int32)
        dt = {1: np.uint8, 2: np.uint16, 4: np.int32, 5: np.float32, 8: np.float64}[kind]
        values = np.empty(nnz, dtype=dt)
        rc = L.w4r_decode(h, 0, -1, indices.ctypes.data, values.ctypes.data, kind, threads, packed_sort, err, 256)
        if rc:
            raise RuntimeError(err.value.decode())
        return m, n, nnz, indptr, indices, values
    finally:
        L.w4r_close(h)


rows = []
files = a.files
for f in files:
    buf = np.fromfile(f, dtype=np.uint8)
    ref = pz.read_1pz(f)
    ok = True
    for th, ps in ((1, 0), (1, 1), (4, 1), (8, 1)):
        m, n, nnz, indptr, indices, values = proto(buf, th, ps)
        e = (m == ref['m'] and n == ref['n'] and nnz == ref['nnz'] and np.array_equal(indptr, ref['indptr'])
             and np.array_equal(indices, ref['indices']) and np.array_equal(values, ref['data'].astype(np.int32)))
        ok = ok and e
    print(f'{os.path.relpath(f, ROOT)}: m={ref["m"]} n={ref["n"]} nnz={ref["nnz"]} identical_to_production={ok}', flush=True)
    if a.check_only:
        continue
    def best(fn):
        ts = []
        for _ in range(a.reps):
            t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
        return min(ts) * 1e3
    t_prod = best(lambda: pz.read_1pz(f))
    res = {'prod_ms': t_prod}
    for th, ps in ((1, 0), (1, 1), (2, 1), (4, 1), (8, 1)):
        res[f't{th}_ps{ps}'] = best(lambda: proto(buf, th, ps))
    res['t4_noverify_u16'] = best(lambda: proto(buf, 4, 1, kind=2))
    res['t4_with_filecrc'] = best(lambda: proto(buf, 4, 1, verify_file=1))
    line = f'{os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f):44s} nnz={ref["nnz"]:9d} ' + ' '.join(f'{k}={v:7.1f}' for k, v in res.items())
    print(line, flush=True)
    rows.append((f, ref['nnz'], res))
if rows:
    with open(a.out, 'w') as fo:
        keys = list(rows[0][2].keys())
        fo.write('file\tnnz\t' + '\t'.join(keys) + '\n')
        for f, nnz, res in rows:
            fo.write(f'{os.path.relpath(f, ROOT)}\t{nnz}\t' + '\t'.join(f'{res[k]:.2f}' for k in keys) + '\n')
