"""Independent reference decoder for pz2 members, written ONLY from docs/format-v2.md (sections 4.1, 4.2, 4.5, 4.6, 5.1, 5.2).

It shares no code with include/singlet/pz2 and exists to prove that the spec text is sufficient to write a decoder.  It is slow
(pure Python, one element at a time): use it on small members.

    python -P bench/pz2/ref_decode.py member.pz2 [expected.csc | expected.1pz]
        decodes (canonical order), checks the stored digest, and, if given, compares with the expected matrix.
"""
import hashlib
import struct
import sys
import zlib

import numpy as np
import zstandard

ZD = zstandard.ZstdDecompressor()


class FormatError(Exception):
    pass


def u16(b, o): return struct.unpack_from('<H', b, o)[0]
def u32(b, o): return struct.unpack_from('<I', b, o)[0]
def u64(b, o): return struct.unpack_from('<Q', b, o)[0]


def leb(b, o):
    v = 0
    s = 0
    while True:
        x = b[o]
        o += 1
        v |= (x & 127) << s
        if not x & 128:
            return v, o
        s += 7


def stream(b, o):
    """STREAM: u32 raw_len, u32 zstd_len, zstd frame (raw_len 0: nothing follows)."""
    raw, z = struct.unpack_from('<II', b, o)
    o += 8
    if raw == 0:
        if z:
            raise FormatError('stream length fields inconsistent')
        return b'', o
    out = ZD.decompress(bytes(b[o:o + z]), max_output_size=raw)
    if len(out) != raw:
        raise FormatError('stream raw length mismatch')
    return out, o + z


def header(b):
    if len(b) < 256 or b[:4] != b'PZ2M':
        raise FormatError('bad magic')
    if zlib.crc32(bytes(b[:252])) & 0xFFFFFFFF != u32(b, 252):
        raise FormatError('header crc')
    h = dict(version=u16(b, 4), codec=u16(b, 6), m=u32(b, 8), n=u32(b, 12), nnz=u64(b, 16), value_type=b[24], row_order=b[25], cpo=b[26], flags=b[27],
             chunk_cols=u32(b, 28), n_chunks=u32(b, 32), lanes=b[36], pb=b[37], pbv=b[38], ctx_kind=b[39], row_names_id=bytes(b[40:48]),
             col_names_id=bytes(b[48:56]), digest=bytes(b[56:72]), total_len=u64(b, 184), rows_kind=b[192], cols_kind=b[193])
    names = ['rows', 'cols', 'kv', 'tables', 'chunk_index', 'chunks', 'extra']
    h['sec'] = {nm: (u64(b, 72 + 16 * i), u64(b, 80 + 16 * i)) for i, nm in enumerate(names)}
    if h['version'] != 1:
        raise FormatError('version')
    return h


def chunk_index(b, h):
    off, ln = h['sec']['chunk_index']
    assert ln == (h['n_chunks'] + 1) * 24
    recs = []
    for c in range(h['n_chunks'] + 1):
        o = off + 24 * c
        recs.append(dict(off=u64(b, o), len=u32(b, o + 8), first_col=u32(b, o + 12), nnz_before=u64(b, o + 16)))
    return recs


# ---- section 5.1 pieces ---------------------------------------------------------------------------------------------------
def cls_split_inverse(c):
    e = c >> 2
    mm = c & 3
    nb = max(e - 1, 0)
    base = (mm | (4 if e > 0 else 0)) << nb
    return base, nb


def posb(p):
    q = p + 2
    lq = q.bit_length() - 1
    return 2 * lq + ((q >> (lq - 1)) & 1) - 2


def parse_table(b, o, nctx, alpha, pbits):
    raw, o = stream(b, o)
    p = 0
    nused, p = leb(raw, p)
    ctx = 0
    tabs = {}
    for u in range(nused):
        d, p = leb(raw, p)
        ctx += d
        ns, p = leb(raw, p)
        sym = 0
        cum = 0
        ent = []
        for i in range(ns):
            ds, p = leb(raw, p)
            sym += ds
            f, p = leb(raw, p)
            for s in range(f):
                ent.append((sym, f, s))
            cum += f
        assert cum == (1 << pbits) and ctx < nctx
        tabs[ctx] = ent
    return tabs, o


KCFG = {0: (0, 39, 0, 11), 1: (0, 39, 1, 5), 2: (1, 19, 1, 5), 3: (0, 39, 0, 0), 4: (1, 19, 0, 0)}


def decode_chunk_rans(blob, nc, nnz_expected, h, TC, TG, TR):
    L = h['lanes']
    pb, pbv = h['pb'], h['pbv']
    sh, cap, ks, kcap = KCFG[h['ctx_kind']]
    npc = cap + 1
    p = 0
    nwc, p = leb(blob, p)
    nw, p = leb(blob, p)
    # --- column counts
    X = u32(blob, p)
    wp = p + 4
    wend = wp + 2 * nwc
    ks_list = []

    def refill_c(x):
        nonlocal wp
        if x < 65536:
            x = (x << 16) | u16(blob, wp)
            wp += 2
        return x
    for j in range(nc):
        sym, f, bias = TC[0][X & 4095]
        X = refill_c(f * (X >> 12) + bias)
        base, nb = cls_split_inverse(sym)
        lo = min(nb, 16)
        ex = X & ((1 << lo) - 1)
        X = refill_c(X >> lo)
        if nb > 16:
            hi = nb - 16
            ex |= (X & ((1 << hi) - 1)) << 16
            X = refill_c(X >> hi)
        ks_list.append(base + ex)
    assert X == 65536 and wp == wend, 'count stream integrity'
    N = sum(ks_list)
    assert N == nnz_expected, 'counts disagree with chunk index'
    p = wend
    # --- lanes
    S, t = divmod(N, L)
    steps = S + (1 if t > 0 else 0)
    e0 = [l * S + min(l, t) for l in range(L)]
    cnt = [S + (1 if l < t else 0) for l in range(L)]
    colptr = [0]
    for k in ks_list:
        colptr.append(colptr[-1] + k)
    lane = []
    for l in range(L):
        if cnt[l] == 0:
            lane.append(None)
            continue
        j = 0
        while colptr[j + 1] <= e0[l]:
            j += 1
        o = e0[l] - colptr[j]
        pp0 = 0
        if o > 0:
            pp0, p = leb(blob, p)
        # remaining non-empty columns after j that this lane needs
        nxt = []
        need = cnt[l] - (ks_list[j] - o)
        jj = j + 1
        while need > 0:
            if ks_list[jj]:
                nxt.append(ks_list[jj])
                need -= ks_list[jj]
            jj += 1
        lane.append(dict(rem=ks_list[j] - o, k=ks_list[j], PP=pp0, nxt=nxt))
    Xs = [u32(blob, p + 4 * l) for l in range(L)]
    wp = p + 4 * L
    wend = wp + 2 * nw

    def refill(x):
        nonlocal wp
        if x < 65536:
            x = (x << 16) | u16(blob, wp)
            wp += 2
        return x

    def depth(k):
        return (k + 1).bit_length() - 1
    out_id = [0] * N
    out_val = [0] * N
    sym = [0] * L
    rank = [0] * L
    nbA = [0] * L
    exA = [0] * L
    e2 = [None] * L
    for s in range(steps):
        act = [l for l in range(L) if s < cnt[l]]
        for l in act:   # A
            st = lane[l]
            kraw = depth(st['k'])
            kc = min(kraw >> ks, kcap)
            pc = min(posb(st['PP']) >> sh, cap)
            ctx = kc * npc + pc
            sy, f, bias = TG[ctx][Xs[l] & ((1 << pb) - 1)]
            Xs[l] = refill(f * (Xs[l] >> pb) + bias)
            sym[l] = sy
        for l in act:   # B
            c = sym[l] >> 1
            base, nb = cls_split_inverse(c)
            lo = min(nb, 16)
            exA[l] = Xs[l] & ((1 << lo) - 1)
            nbA[l] = nb
            Xs[l] = refill(Xs[l] >> lo)
        for l in act:   # B2
            if nbA[l] > 16:
                hi = nbA[l] - 16
                exA[l] |= (Xs[l] & ((1 << hi) - 1)) << 16
                Xs[l] = refill(Xs[l] >> hi)
            rank[l] = lane[l]['PP'] + cls_split_inverse(sym[l] >> 1)[0] + exA[l]
        flagged = [l for l in act if sym[l] & 1]
        for l in flagged:   # C1
            st = lane[l]
            dk = min(11 * depth(st['k']) // 32, 3)
            rc = min(posb(rank[l]) >> 1, 20) * 4 + dk
            sy, f, bias = TR[rc][Xs[l] & ((1 << pbv) - 1)]
            Xs[l] = refill(f * (Xs[l] >> pbv) + bias)
            e2[l] = sy
        nb2 = {}
        ex2 = {}
        for l in flagged:   # C2
            base, nb = cls_split_inverse(e2[l])
            lo = min(nb, 16)
            ex2[l] = Xs[l] & ((1 << lo) - 1)
            nb2[l] = nb
            Xs[l] = refill(Xs[l] >> lo)
        for l in flagged:   # C3
            if nb2[l] > 16:
                hi = nb2[l] - 16
                ex2[l] |= (Xs[l] & ((1 << hi) - 1)) << 16
                Xs[l] = refill(Xs[l] >> hi)
        for l in act:   # emit
            if sym[l] & 1:
                q = cls_split_inverse(e2[l])[0] + ex2[l]
                v = 0 if q == 0 else q + 1
            else:
                v = 1
            out_id[e0[l] + s] = rank[l]
            out_val[e0[l] + s] = v
            st = lane[l]
            st['PP'] = rank[l] + 1
            st['rem'] -= 1
            if st['rem'] == 0 and s + 1 < cnt[l]:
                k = st['nxt'].pop(0)
                st['rem'] = k
                st['k'] = k
                st['PP'] = 0
    assert wp == wend and all(x == 65536 for x in Xs), 'lane stream integrity'
    return ks_list, out_id, out_val


def decode_chunk_generic(blob, nc, nnz_expected, h):
    o = 0
    cnt, o = stream(blob, o)
    gaps, o = stream(blob, o)
    vals, o = stream(blob, o)
    p = 0
    ks_list = []
    for j in range(nc):
        k, p = leb(cnt, p)
        ks_list.append(k)
    assert sum(ks_list) == nnz_expected
    ids, vs = [], []
    gp = 0
    vp = 0
    raw = (h['value_type'] == 4 and not (h['flags'] & 1))
    for k in ks_list:
        prev = -1
        for _ in range(k):
            g, gp = leb(gaps, gp)
            prev = prev + 1 + g
            ids.append(prev)
            if raw:
                vs.append(u32(vals, vp))
                vp += 4
            else:
                v, vp = leb(vals, vp)
                vs.append(v)
    assert gp == len(gaps) and vp == len(vals)
    return ks_list, ids, vs


def names_part(b, h, which):
    kind = h['rows_kind'] if which == 'rows' else h['cols_kind']
    off, ln = h['sec']['extra' if which == 'rows' else 'cols']
    count = h['m'] if which == 'rows' else h['n']
    if kind == 4:
        return kind, b'', []
    if kind == 0:
        raw, _ = stream(b, off)
        return kind, raw, raw
    if kind == 2:
        raw, _ = stream(b, off)
        idx, p, prev = [], 0, 0
        for _ in range(count):
            z, p = leb(raw, p)
            d = (z >> 1) ^ -(z & 1)
            prev += d
            idx.append(prev)
        return kind, b'', idx
    return kind, b'', []


def decode(path):
    b = open(path, 'rb').read()
    h = header(b)
    recs = chunk_index(b, h)
    m, n, nnz = h['m'], h['n'], h['nnz']
    tabs = None
    perm = None
    if h['codec'] == 1:
        o = h['sec']['tables'][0]
        pbits = (12, h['pb'], h['pbv'])
        sh, cap, ks, kcap = KCFG[h['ctx_kind']]
        TC, o = parse_table(b, o, 1, 124, 12)
        TG, o = parse_table(b, o, (cap + 1) * (kcap + 1), 248, h['pb'])
        TR, o = parse_table(b, o, 84, 124, h['pbv'])
        ro = h['sec']['rows'][0]
        raw, _ = stream(b, ro)
        keys, p = [], 0
        for i in range(m):
            k, p = leb(raw, p)
            keys.append(k)
        # stored order: key descending, canonical id ascending
        perm = sorted(range(m), key=lambda i: (-keys[i], i))
    indptr = [0]
    indices = []
    values = []
    cbase = h['sec']['chunks'][0]
    for c in range(h['n_chunks']):
        r, r2 = recs[c], recs[c + 1]
        nc = r2['first_col'] - r['first_col']
        expect = r2['nnz_before'] - r['nnz_before']
        blob = b[cbase + r['off']: cbase + r['off'] + r['len']] + bytes(64)
        if h['codec'] == 1:
            ks_list, ids, vs = decode_chunk_rans(blob, nc, expect, h, TC, TG, TR)
        else:
            ks_list, ids, vs = decode_chunk_generic(blob, nc, expect, h)
        pos = 0
        for k in ks_list:
            col = list(zip(ids[pos:pos + k], vs[pos:pos + k]))
            pos += k
            if perm is not None:
                col = [(perm[i], v) for i, v in col]
            col.sort()
            assert all(col[i][0] < col[i + 1][0] for i in range(len(col) - 1)) and all(x[0] < m for x in col)
            indices += [i for i, _ in col]
            values += [v for _, v in col]
            indptr.append(indptr[-1] + k)
    assert len(indices) == nnz and len(indptr) == n + 1
    # digest (4.6)
    rk, rblob, _ = names_part(b, h, 'rows')
    ck, cblob, _ = names_part(b, h, 'cols')
    return h, b, indptr, indices, values, perm


def digest(h, indptr, indices, values, rblob, cblob, kv):
    hs = hashlib.sha256()
    hs.update(b'PZ2CANON1')
    hs.update(struct.pack('<IIQB', h['m'], h['n'], h['nnz'], h['value_type']))
    for j in range(h['n']):
        a, e = indptr[j], indptr[j + 1]
        hs.update(struct.pack('<I', e - a))
        hs.update(np.asarray(indices[a:e], dtype='<u4').tobytes())
        hs.update(np.asarray(values[a:e], dtype='<u4').tobytes())
    hs.update(struct.pack('<I', rblob.count(b'\0')))
    hs.update(rblob)
    hs.update(struct.pack('<I', cblob.count(b'\0')))
    hs.update(cblob)
    for k, v in sorted(kv.items()):
        hs.update(k + b'\0' + v + b'\0')
    return hs.digest()[:16]


def read_kv(b, h):
    off, ln = h['sec']['kv']
    if not ln:
        return {}
    npairs = u32(b, off)
    raw, _ = stream(b, off + 4)
    parts = raw.split(b'\0')[:-1]
    assert len(parts) == 2 * npairs
    return {parts[2 * i]: parts[2 * i + 1] for i in range(npairs)}


def read_csc(path):
    b = open(path, 'rb').read()
    m, n, nnz = struct.unpack_from('<IIQ', b, 0)
    o = 18
    ip = np.frombuffer(b, '<u4', n + 1, o); o += 4 * (n + 1)
    ix = np.frombuffer(b, '<u4', nnz, o); o += 4 * nnz
    vv = np.frombuffer(b, '<u4', nnz, o)
    return m, n, ip, ix, vv


def main():
    path = sys.argv[1]
    h, b, indptr, indices, values, perm = decode(path)
    print('decoded: codec %d m %d n %d nnz %d chunks %d' % (h['codec'], h['m'], h['n'], h['nnz'], h['n_chunks']))
    rk, rblob, _ = names_part(b, h, 'rows')
    ck, cblob, _ = names_part(b, h, 'cols')
    kv = read_kv(b, h)
    ok = True
    if len(sys.argv) > 2 and sys.argv[2].endswith('.csc'):
        m, n, ip, ix, vv = read_csc(sys.argv[2])
        ok = (m == h['m'] and n == h['n'] and list(ip) == indptr and list(ix) == indices and list(vv) == values)
        print('matches expected csc:', ok)
    elif len(sys.argv) > 2:
        sys.path.insert(0, 'python')
        from singlet._pz import read_1pz
        r = read_1pz(sys.argv[2])
        ok = (list(r['indptr']) == indptr and list(r['indices']) == indices and list(r['data']) == values) if isinstance(r, dict) else None
        print('matches legacy:', ok)
    # digest: names are the logical names; for pack/cells kinds the caller passes them, so only verify when stored inline or none
    if (h['rows_kind'] in (0, 4)) and (h['cols_kind'] in (0, 4)):
        d = digest(h, indptr, indices, values, rblob if h['rows_kind'] == 0 else b'', cblob if h['cols_kind'] == 0 else b'', kv)
        good = d == h['digest']
        print('stored digest reproduced:', good)
        ok = ok and good
    elif h['digest'] != bytes(16):
        # logical names unavailable here: g<i> / c<j> convention used by pz2tool synth
        rb = b''.join(b'g%d\0' % i for i in range(h['m'])) if h['rows_kind'] != 0 else rblob
        cb = b''.join(b'c%d\0' % j for j in range(h['n'])) if h['cols_kind'] != 0 else cblob
        d = digest(h, indptr, indices, values, rb, cb, kv)
        good = d == h['digest']
        print('stored digest reproduced (logical names g<i>/c<j>):', good)
        ok = ok and good
    print('kv:', kv)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
