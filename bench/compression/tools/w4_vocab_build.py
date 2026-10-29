#!/usr/bin/env python
"""w4_vocab: build the shared feature-name packs ("SVP0") from public bundles' row names and prove the roundtrip.

    python -P tools/w4_vocab_build.py            # needs corpus_w4_vocab/ from tools/w4_vocab_fetch.py
        -> results/w4_vocab_packs/<build>.svp, results/w4_vocab_sizes.tsv, results/w4_vocab_roundtrip.txt

A pack holds, for one reference build, the exon row names, the intron row names (derived, 0 bytes), and the gene table
(feature_vocab.json, derived, 0 bytes) as ONE structured payload:
  per gene   : gene id (numeric delta), symbol kind (0 empty / 1 == gene id / 2 explicit) + explicit symbols, chromosome index,
               exon count, first exon start
  per exon   : length (end-start) and the gap to the next exon of the same gene
Introns are the gaps (end_i, start_i+1); feature_vocab.json is json.dumps(indent=2) of the gene table.
Each integer stream is stored as LEB128 or as 4 byte planes (whichever zstd-19 packs smaller), zstd-19 per stream.
The barnyard build is a composite of the GRCh38 and GRCm39 packs with a species tag inserted after the symbol.
"""
import collections, hashlib, io, json, lzma, os, struct, sys, time, zlib
import numpy as np
import zstandard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4_vocab_lib as L

MAGIC = b"SVP0"
Z = zstandard.ZstdCompressor(level=19)
ZD = zstandard.ZstdDecompressor()
OUT = os.path.join(L.ROOT, "results", "w4_vocab_packs")


# ---------------------------------------------------------------- integer streams
def leb_enc(vals):
    out = bytearray()
    for v in vals.tolist():
        while v >= 128:
            out.append((v & 127) | 128); v >>= 7
        out.append(v)
    return bytes(out)


def leb_dec(buf, n):
    out = np.empty(n, np.uint64)
    p = 0
    for i in range(n):
        v = 0; s = 0
        while True:
            b = buf[p]; p += 1
            v |= (b & 127) << s
            if b < 128: break
            s += 7
        out[i] = v
    return out


def planes_enc(vals):
    v = vals.astype(np.uint32)
    return b"".join(((v >> (8 * k)) & 255).astype(np.uint8).tobytes() for k in range(4))


def planes_dec(buf, n):
    a = np.frombuffer(buf, np.uint8).reshape(4, n).astype(np.uint64)
    return a[0] | (a[1] << 8) | (a[2] << 16) | (a[3] << 24)


def pack_stream(vals):
    """-> bytes: u8 mode (0 leb, 1 planes), u32 n, u32 clen, zstd frame (smaller of the two)."""
    vals = np.asarray(vals, np.uint64)
    best = None
    for mode, enc in ((0, leb_enc), (1, planes_enc)):
        z = Z.compress(enc(vals))
        if best is None or len(z) < len(best[1]):
            best = (mode, z)
    return struct.pack("<BII", best[0], len(vals), len(best[1])) + best[1]


def unpack_stream(buf, p):
    mode, n, cl = struct.unpack_from("<BII", buf, p); p += 9
    raw = ZD.decompress(buf[p:p + cl]); p += cl
    return (leb_dec(raw, n) if mode == 0 else planes_dec(raw, n)), p


def pack_blob(b):
    z = Z.compress(b)
    return struct.pack("<I", len(z)) + z


def unpack_blob(buf, p):
    (cl,) = struct.unpack_from("<I", buf, p); p += 4
    return ZD.decompress(buf[p:p + cl]), p + cl


# ---------------------------------------------------------------- structure
def build_struct(exon, intron, vocab):
    genes_tab = [(g["gene_id"], g["gene_name"]) for g in vocab["genes"]]
    gid, sym, chrom, st, en = L.parse_rows(exon, "exon", genes_tab, "")
    grp = L.group_genes(gid, sym, chrom, st, en)
    assert [g[0] for g in grp] == [g for g, _ in genes_tab], "gene order differs from feature_vocab.json"
    for g, (vg, vn) in zip(grp, genes_tab):
        assert (g[1] if g[1] else g[0]) == vn, "gene_name != symbol/id"
    ids = [g[0] for g in grp]
    import re
    pref = re.match(r"^[A-Za-z]+", ids[0]).group(0)
    width = len(ids[0]) - len(pref)
    nums = np.array([int(i[len(pref):]) for i in ids], np.int64)
    assert all(i == pref + str(n).zfill(width) for i, n in zip(ids, nums)), "gene ids not <prefix><zero padded number>"
    assert (np.diff(nums) > 0).all(), "gene ids not increasing"
    chroms = []
    cidx = {}
    gch = []
    for g in grp:
        if g[2] not in cidx:
            cidx[g[2]] = len(chroms); chroms.append(g[2])
        gch.append(cidx[g[2]])
    kind = np.array([0 if g[1] == "" else (1 if g[1] == g[0] else 2) for g in grp], np.uint8)
    S = dict(build=vocab["reference_build"], schema=vocab.get("schema_version", "1.0"), prefix=pref, width=width, nums=nums, kind=kind,
             syms=[g[1] for g, k in zip(grp, kind) if k == 2], chroms=chroms, gch=np.array(gch, np.uint32),
             nex=np.array([len(g[3]) for g in grp], np.uint32), gstart=np.array([g[3][0][0] for g in grp], np.uint64))
    elen, egap = [], []
    for g in grp:
        ee = g[3]
        for i, (a, b) in enumerate(ee):
            assert b >= a
            elen.append(b - a)
            if i + 1 < len(ee):
                assert ee[i + 1][0] >= b
                egap.append(ee[i + 1][0] - b)
    S["elen"] = np.array(elen, np.uint64); S["egap"] = np.array(egap, np.uint64)
    return S


def expand(S, tag=""):
    """struct -> (exon names, intron names, vocab genes list [(id, name)])"""
    ex, it, vg = [], [], []
    pos_l = pos_g = 0
    ns = 0
    nums = S["nums"].tolist(); kind = S["kind"].tolist(); nex = S["nex"].tolist(); gch = S["gch"].tolist()
    gstart = S["gstart"].tolist(); elen = S["elen"].tolist(); egap = S["egap"].tolist(); syms = S["syms"]; chroms = S["chroms"]
    pref, width = S["prefix"], S["width"]
    for g in range(len(nums)):
        gid = pref + str(nums[g]).zfill(width)
        if kind[g] == 0: sym = ""; name = gid
        elif kind[g] == 1: sym = gid; name = gid
        else: sym = syms[ns]; ns += 1; name = sym
        vg.append((gid, name))
        head = gid + "_" + sym + "_"
        ch = tag + chroms[gch[g]]
        a = gstart[g]
        for i in range(nex[g]):
            b = a + elen[pos_l + i]
            ex.append("%s%s:%d-%d" % (head, ch, a, b))
            if i + 1 < nex[g]:
                nxt = b + egap[pos_g + i]
                it.append("%sintron_%s:%d-%d" % (head, ch, b, nxt))
                a = nxt
        pos_l += nex[g]; pos_g += nex[g] - 1
    return ex, it, vg


def vocab_json(build, schema, genes):
    return json.dumps({"schema_version": schema, "reference_build": build, "n_genes": len(genes),
                       "genes": [{"gene_id": a, "gene_name": b} for a, b in genes]}, indent=2).encode()


def canon_id(exon, intron):
    """content id: sha256(sha256hex(exon names) || sha256hex(intron names)) -> 16 hex chars shown (8 bytes)."""
    h = hashlib.sha256((L.names_sha256(exon) + L.names_sha256(intron)).encode()).hexdigest()
    return h


# ---------------------------------------------------------------- serialisation
def encode(S):
    parts = collections.OrderedDict()
    b = S["build"].encode()
    hdr = MAGIC + struct.pack("<HHBBHII", 1, 0, len(S["prefix"]), S["width"], len(S["chroms"]), len(S["nums"]), len(S["elen"]))
    hdr += struct.pack("<B", len(b)) + b + S["prefix"].encode() + S["schema"].encode().ljust(8, b"\0")
    parts["header"] = hdr
    nums = S["nums"].astype(np.uint64)
    d = np.concatenate([nums[:1], np.diff(nums).astype(np.uint64) - 1])
    parts["gene_id"] = pack_stream(d)
    parts["sym_kind"] = pack_blob(S["kind"].tobytes())
    parts["symbols"] = pack_blob(b"\0".join(s.encode() for s in S["syms"]))
    parts["chrom_names"] = pack_blob(b"\0".join(c.encode() for c in S["chroms"]))
    parts["gene_chrom"] = pack_stream(S["gch"])
    parts["exons_per_gene"] = pack_stream(S["nex"].astype(np.uint64) - 1)
    parts["gene_start"] = pack_stream(S["gstart"])
    parts["exon_len"] = pack_stream(S["elen"])
    parts["exon_gap"] = pack_stream(S["egap"])
    return b"".join(parts.values()), {k: len(v) for k, v in parts.items()}


def decode(buf):
    assert buf[:4] == MAGIC
    ver, flags, pl, width, nchrom, ngenes, nexons = struct.unpack_from("<HHBBHII", buf, 4)
    p = 4 + struct.calcsize("<HHBBHII")
    bl = buf[p]; p += 1
    build = buf[p:p + bl].decode(); p += bl
    prefix = buf[p:p + pl].decode(); p += pl
    schema = buf[p:p + 8].rstrip(b"\0").decode(); p += 8
    S = dict(build=build, schema=schema, prefix=prefix, width=width)
    d, p = unpack_stream(buf, p)
    S["nums"] = np.cumsum(np.concatenate([d[:1], d[1:] + 1]).astype(np.int64))
    kb, p = unpack_blob(buf, p); S["kind"] = np.frombuffer(kb, np.uint8)
    sb, p = unpack_blob(buf, p); S["syms"] = [x.decode() for x in sb.split(b"\0")] if (S["kind"] == 2).any() else []
    cb, p = unpack_blob(buf, p); S["chroms"] = [x.decode() for x in cb.split(b"\0")]
    S["gch"], p = unpack_stream(buf, p)
    nx, p = unpack_stream(buf, p); S["nex"] = (nx + 1).astype(np.int64)
    S["gstart"], p = unpack_stream(buf, p)
    S["elen"], p = unpack_stream(buf, p)
    S["egap"], p = unpack_stream(buf, p)
    assert p == len(buf)
    return S


def composite_encode(build, parts):
    """parts: [(pack_id_hex16, tag)] -> a few dozen bytes"""
    b = build.encode()
    out = b"SVPC" + struct.pack("<HB", 1, len(parts)) + struct.pack("<B", len(b)) + b
    for pid, tag in parts:
        t = tag.encode()
        out += bytes.fromhex(pid[:16]) + struct.pack("<B", len(t)) + t
    return out


# ---------------------------------------------------------------- main
def sizes_of(names_ex, names_it, vocab_raw):
    r = collections.OrderedDict()
    ex_b = b"".join(s.encode() + b"\0" for s in names_ex); it_b = b"".join(s.encode() + b"\0" for s in names_it)
    r["exon_raw"] = len(ex_b); r["intron_raw"] = len(it_b)
    r["names_zstd19_each"] = len(Z.compress(ex_b)) + len(Z.compress(it_b))
    r["names_xz9e_each"] = len(lzma.compress(ex_b, preset=9 | lzma.PRESET_EXTREME)) + len(lzma.compress(it_b, preset=9 | lzma.PRESET_EXTREME))
    r["vocab_json_raw"] = len(vocab_raw)
    r["vocab_json_deflate9"] = len(zlib.compress(vocab_raw, 9))
    r["vocab_json_zstd19"] = len(Z.compress(vocab_raw))
    r["vocab_json_xz9e"] = len(lzma.compress(vocab_raw, preset=9 | lzma.PRESET_EXTREME))
    return r


BASELINES = "--baselines" in sys.argv


def main():
    os.makedirs(OUT, exist_ok=True)
    import csv
    index = list(csv.DictReader(open(os.path.join(L.DATA, "index.tsv")), delimiter="\t"))
    by = collections.OrderedDict()
    for r in index:
        by.setdefault(r["build"], r)
    log = []
    rows = []
    stash = {}
    order = [b for b in by if "barnyard" not in b] + [b for b in by if "barnyard" in b]
    for build in order:
        r0 = by[build]
        gse, gsm = r0["gse"], r0["gsm"]
        ex, _, _ = L.load_meta(os.path.join(L.DATA, "%s.%s.exon.meta" % (gse, gsm)))
        it, _, _ = L.load_meta(os.path.join(L.DATA, "%s.%s.intron.meta" % (gse, gsm)))
        vraw = open(os.path.join(L.DATA, "%s.feature_vocab.json" % gse), "rb").read()
        vocab = json.loads(vraw)
        rec = collections.OrderedDict(build=build, sample="%s/%s" % (gse, gsm), exon_rows=len(ex), intron_rows=len(it), genes=len(vocab["genes"]))
        if BASELINES:
            rec.update(sizes_of(ex, it, vraw))
        else:
            rec["exon_raw"] = sum(len(x) + 1 for x in ex); rec["intron_raw"] = sum(len(x) + 1 for x in it); rec["vocab_json_raw"] = len(vraw)
        cid = canon_id(ex, it)
        rec["pack_id"] = cid[:16]
        t0 = time.time()
        if "barnyard" not in build:
            S = build_struct(ex, it, vocab)
            pk, parts = encode(S)
            S2 = decode(pk)
            ex2, it2, vg2 = expand(S2)
            ok_ex = ex2 == ex; ok_it = it2 == it
            ok_v = vocab_json(S2["build"], S2["schema"], vg2) == vraw
            rec["pack_bytes"] = len(pk)
            for k, v in parts.items(): rec["sec_" + k] = v
            # same streams through xz-9e for information (varint streams concatenated)
            leb = b"".join(leb_enc(S[k].astype(np.uint64)) for k in ("elen", "egap", "gstart")) + leb_enc(S["nex"].astype(np.uint64))
            rec["info_leb_streams_xz"] = len(lzma.compress(leb, preset=9 | lzma.PRESET_EXTREME))
            open(os.path.join(OUT, build + ".svp"), "wb").write(pk)
            stash[build] = (ex, it, vocab, cid)
            rec["pack_sha256"] = hashlib.sha256(pk).hexdigest()[:16]
        else:
            # composite: GRCh38 + GRCm39 packs with species tags
            hs, ms = stash["GRCh38-2024-A"], stash["GRCm39-2024-A"]
            Sh = decode(open(os.path.join(OUT, "GRCh38-2024-A.svp"), "rb").read())
            Sm = decode(open(os.path.join(OUT, "GRCm39-2024-A.svp"), "rb").read())
            exh, ith, vgh = expand(Sh, "GRCh38___"); exm, itm, vgm = expand(Sm, "GRCm39___")
            ex2, it2 = exh + exm, ith + itm
            ok_ex = ex2 == ex; ok_it = it2 == it
            ok_v = vocab_json(build, vocab.get("schema_version", "1.0"), vgh + vgm) == vraw
            pk = composite_encode(build, [(hs[3], "GRCh38___"), (ms[3], "GRCm39___")])
            rec["pack_bytes"] = len(pk)
            open(os.path.join(OUT, build + ".svp"), "wb").write(pk)
            rec["pack_sha256"] = hashlib.sha256(pk).hexdigest()[:16]
        rec["roundtrip_exon_ok"] = int(ok_ex); rec["roundtrip_intron_ok"] = int(ok_it); rec["roundtrip_vocab_json_ok"] = int(ok_v)
        rec["build_decode_expand_s"] = round(time.time() - t0, 1)
        rows.append(rec)
        log.append("%-24s id=%s pack=%d B  exon_ok=%s intron_ok=%s vocab_json_ok=%s" % (build, cid[:16], rec["pack_bytes"], ok_ex, ok_it, ok_v))
        print(log[-1], flush=True)
    # cross-study roundtrip: every fetched study's names equal the expansion of that build's pack
    chk = []
    cache = {}
    for r in index:
        b = r["build"]
        if "barnyard" in b: continue
        if b not in cache:
            cache[b] = expand(decode(open(os.path.join(OUT, b + ".svp"), "rb").read()))
        rows_, _, _ = L.load_meta(os.path.join(L.DATA, "%s.%s.%s.meta" % (r["gse"], r["gsm"], r["kind"])))
        ex2, it2, _ = cache[b]
        ok = (ex2 if r["kind"] == "exon" else it2) == rows_
        chk.append((r["gse"], r["gsm"], r["kind"], b, ok))
    log.append("")
    log.append("cross-study check: names of every fetched (study, sample, kind) equal the pack expansion of its build:")
    for c in chk: log.append("  %s/%s %-6s %-24s %s" % (c[0], c[1], c[2], c[3], "OK" if c[4] else "MISMATCH"))
    open(os.path.join(L.ROOT, "results", "w4_vocab_roundtrip.txt"), "w").write("\n".join(log) + "\n")
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(os.path.join(L.ROOT, "results", ("w4_vocab_sizes_baselines.tsv" if BASELINES else "w4_vocab_sizes.tsv")), "w") as o:
        o.write("metric\t" + "\t".join(r["build"] for r in rows) + "\n")
        for k in keys:
            o.write(k + "\t" + "\t".join(str(r.get(k, "")) for r in rows) + "\n")
    print(open(os.path.join(L.ROOT, "results", ("w4_vocab_sizes_baselines.tsv" if BASELINES else "w4_vocab_sizes.tsv"))).read())


def stash_done(chk):
    return {(c[0], c[1], c[2]) for c in chk}


if __name__ == "__main__":
    main()
