"""Laptop check for the early-exit STAR localSearch (src/star/SequenceFuns.cpp): the same ixBest as the plain double loop.

Run: python bench/host_cost/model_localsearch.py   (numpy + numba; no STAR, no compiler needed)

ref():  line-for-line transcription of the original localSearch (x, y numeric bases 0-3, N = 4; pMM a double).
fast(): transcription of the new C++, including the per-pMM table kmax[m] of how many mismatches a candidate of m positions may
        have and still qualify, and the `m < nMatchBest` stop.
Both use float64 division as C++ does (a/b is correctly rounded, so monotone in a and in b: the argument for the early exit).
Also reports the inner-loop iteration counts, the cost model for the 10x 5' TSO clip (30-base adapter, 91-base reads).
"""
import sys
import numpy as np
from numba import njit


@njit(cache=True)
def ref(x, y, pMM):
    nx, ny = len(x), len(y)
    nMatchBest = 0; nMMbest = 0; ixBest = nx; it = 0
    for ix in range(nx):
        nMatch = 0; nMM = 0
        for iy in range(min(ny, nx - ix)):
            it += 1
            if x[ix + iy] > 3:
                continue
            if x[ix + iy] == y[iy]:
                nMatch += 1
            else:
                nMM += 1
        if nMatch > 0:
            ratio = nMM / nMatch
        elif nMM > 0:
            ratio = np.inf
        else:
            ratio = np.nan
        if (nMatch > nMatchBest or (nMatch == nMatchBest and nMM < nMMbest)) and ratio <= pMM:
            ixBest = ix; nMatchBest = nMatch; nMMbest = nMM
    return ixBest, it


@njit(cache=True)
def alive(k, m, pMM):
    d = m - k
    if d == 0:
        r = np.inf if k > 0 else np.nan
    else:
        r = k / d
    return not (r > pMM)


@njit(cache=True)
def table(M, pMM):
    kmax = np.zeros(M + 1, np.int64)
    for m in range(1, M + 1):
        k = kmax[m - 1]
        while k + 1 <= m and alive(k + 1, m, pMM):
            k += 1
        kmax[m] = k
    return kmax


@njit(cache=True)
def fast(x, y, pMM, kmax):
    nx, ny = len(x), len(y)
    nMatchBest = 0; nMMbest = 0; ixBest = nx; it = 0
    for ix in range(nx):
        m = min(ny, nx - ix)
        if m < nMatchBest:
            break
        nMatch = 0; nMM = 0; dead = False
        for iy in range(m):
            it += 1
            c = x[ix + iy]
            if c > 3:
                continue
            if c == y[iy]:
                nMatch += 1
            else:
                nMM += 1
                if nMM > kmax[m]:
                    dead = True
                    break
        if dead:
            continue
        if nMatch > 0:
            ratio = nMM / nMatch
        elif nMM > 0:
            ratio = np.inf
        else:
            ratio = np.nan
        if (nMatch > nMatchBest or (nMatch == nMatchBest and nMM < nMMbest)) and ratio <= pMM:
            ixBest = ix; nMatchBest = nMatch; nMMbest = nMM
    return ixBest, it


def main():
    rng = np.random.default_rng(1)
    ad = np.frombuffer(b"AAGCAGTGGTATCAACGCAGAGTACATGGG", np.uint8).copy()
    lut = np.zeros(256, np.int8); lut[ord('C')] = 1; lut[ord('G')] = 2; lut[ord('T')] = 3
    ad = lut[ad]
    n = 0; bad = 0; itr = 0; itf = 0
    # 1) the 10x 5' case: 91-base reads, the TSO adapter, MMp 0.1; random reads, reads ending in a partial/full adapter, with N
    kmax = table(700, 0.1)
    for t in range(200000):
        L = int(rng.integers(20, 92))
        x = rng.integers(0, 4, L).astype(np.int8)
        r = rng.random()
        if r < 0.3:        # adapter at the 3' end, with mismatches
            a = int(rng.integers(1, 31)); a = min(a, L)
            x[L - a:] = ad[:a]
            for _ in range(int(rng.integers(0, 4))):
                x[int(rng.integers(L - a, L))] = rng.integers(0, 4)
        if r > 0.9:        # Ns
            x[rng.integers(0, L, int(rng.integers(1, 6)))] = 4
        a1, i1 = ref(x, ad, 0.1); a2, i2 = fast(x, ad, 0.1, kmax)
        n += 1; bad += a1 != a2; itr += i1; itf += i2
    print(f"TSO/MMp0.1: {n} reads, {bad} differ; inner iterations per read: plain {itr/n:.0f}, early-exit {itf/n:.0f} ({itr/itf:.1f}x fewer)")
    assert bad == 0
    # 2) any pMM (incl. 0, 0.5, 1, > 1, negative, inf, nan), adapters of any length (the polyA adapter is DEF_readSeqLengthMax long), N in y
    for pMM in [0.0, 0.05, 0.1, 0.2, 1 / 3, 0.5, 1.0, 2.0, 1e9, np.inf, -0.1, np.nan]:
        km = table(700, pMM)
        for t in range(20000):
            nx = int(rng.integers(0, 120)); ny = int(rng.integers(1, 80))
            x = rng.integers(0, 5 if rng.random() < 0.3 else 4, nx).astype(np.int8)
            if rng.random() < 0.5 and nx > 0:      # plant a copy of the adapter
                y = rng.integers(0, 4, ny).astype(np.int8)
                s = int(rng.integers(0, nx)); k = min(ny, nx - s); x[s:s + k] = y[:k]
                for _ in range(int(rng.integers(0, 4))):
                    x[int(rng.integers(s, nx))] = rng.integers(0, 4)
            else:
                y = rng.integers(0, 5 if rng.random() < 0.1 else 4, ny).astype(np.int8)
            a1, _ = ref(x, y, pMM); a2, _ = fast(x, y, pMM, km)
            n += 1
            if a1 != a2:
                bad += 1; print("DIFF", pMM, x, y, a1, a2); sys.exit(1)
    # 3) polyA adapter (all A, length 650): O(L^2/2) plain
    km = table(700, 0.1); y = np.zeros(650, np.int8); itr = itf = 0
    for t in range(3000):
        L = int(rng.integers(20, 150)); x = rng.integers(0, 4, L).astype(np.int8)
        if rng.random() < 0.5:
            a = int(rng.integers(5, L)); x[L - a:] = 0
            for _ in range(int(rng.integers(0, 3))): x[int(rng.integers(L - a, L))] = rng.integers(0, 4)
        a1, i1 = ref(x, y, 0.1); a2, i2 = fast(x, y, 0.1, km); itr += i1; itf += i2
        assert a1 == a2
    print(f"polyA(650)/MMp0.1: iterations per read plain {itr/3000:.0f}, early-exit {itf/3000:.0f} ({itr/itf:.1f}x fewer)")
    print(f"OK: {n} cases, identical ixBest")


if __name__ == "__main__":
    main()
