#!/usr/bin/env python
"""Track A, deliverable 1: component accounting.

(a) per .1pz file: header / perm / ptr / chunk table / chunk headers / chunk payload / colsums /
    metadata (split into rownames, colnames, user_kv; each also compressed alone with zstd-3) / footer.
(b) per bundle: zip entries by type, per sample, per called cell, zip overhead.

    python -P tools/track_a_account.py            # writes results/track_a_account_files.tsv and
                                                  # results/track_a_account_bundle.tsv, prints tables
"""
import glob, io, json, os, struct, sys, zipfile
import zstandard as zstd

HDR = "<IHBBIIQBBHIIIIIQIIQII"
HDR_SZ = 96
Z = zstd.ZstdDecompressor()
C3 = zstd.ZstdCompressor(level=3)


def parse_1pz(buf, name=""):
    (magic, ver, vt, flags, m, n, nnz, pw, lvl, _pad, nch, perm_z, ptr_z, chunk_cols, ff,
     meta_off, meta_z, colsums_z, t_off, t_z, t_ch) = struct.unpack(HDR, buf[:80])
    assert magic == 0x5A315054, name
    sz = len(buf)
    ct_off = HDR_SZ + perm_z + ptr_z
    ctable = struct.unpack("<%dI" % nch, buf[ct_off:ct_off + 4 * nch]) if nch else ()
    chunk_bytes = sum(ctable)
    d = dict(bytes=sz, m=m, n=n, nnz=nnz, vt=vt, flags=flags, level=lvl, chunk_cols=chunk_cols, chunks=nch,
             header=HDR_SZ, perm_z=perm_z, ptr_z=ptr_z, ctable=4 * nch, chunk_hdr=20 * nch,
             chunk_payload=chunk_bytes - 20 * nch, colsums_z=colsums_z, meta_z=meta_z, footer=16,
             transpose_z=t_z)
    # metadata split
    d.update(rown_raw=0, coln_raw=0, kv_raw=0, rown_z3=0, coln_z3=0, kv_z3=0, n_rown=0, n_coln=0)
    if meta_z:
        raw = Z.decompress(buf[meta_off:meta_off + meta_z], max_output_size=1 << 30)
        p = 0
        while p < len(raw):
            tag = raw[p]; p += 1
            if tag == 0:
                break
            (ln,) = struct.unpack("<I", raw[p:p + 4]); p += 4
            body = raw[p:p + ln]; p += ln
            if tag == 1:
                d["rown_raw"] = ln; d["rown_z3"] = len(C3.compress(body)); d["n_rown"] = body.count(b"\0")
            elif tag == 2:
                d["coln_raw"] = ln; d["coln_z3"] = len(C3.compress(body)); d["n_coln"] = body.count(b"\0")
            elif tag == 3:
                d["kv_raw"] = ln; d["kv_z3"] = len(C3.compress(body))
    acc = (d["header"] + perm_z + ptr_z + d["ctable"] + chunk_bytes + colsums_z + meta_z + 16 + t_z)
    d["unaccounted"] = sz - acc
    return d


FIELDS = ["bytes", "m", "n", "nnz", "vt", "level", "chunk_cols", "chunks", "header", "perm_z", "ptr_z", "ctable",
          "chunk_hdr", "chunk_payload", "colsums_z", "meta_z", "footer", "unaccounted", "n_rown", "n_coln",
          "rown_raw", "coln_raw", "kv_raw", "rown_z3", "coln_z3", "kv_z3"]


def called_cells_from_tsv(text):
    n = 0
    for i, line in enumerate(text.splitlines()):
        if i == 0:
            continue
        f = line.split("\t")
        if len(f) > 1 and f[1] == "TRUE":
            n += 1
    return n


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    os.makedirs("results", exist_ok=True)
    # ---------------- per-file ----------------
    rows = []
    sample_total = {}
    for dpath in sorted(glob.glob("corpus/*/")):
        sid = os.path.basename(dpath.rstrip("/\\"))
        if sid == "bundles":
            continue
        called = 0
        cc = os.path.join(dpath, "cell_calls.tsv")
        if os.path.exists(cc):
            called = called_cells_from_tsv(open(cc).read())
        side = {f: os.path.getsize(os.path.join(dpath, f)) for f in os.listdir(dpath)
                if not f.endswith(".1pz")}
        for member in ("exon_counts.1pz", "intron_counts.1pz"):
            p = os.path.join(dpath, member)
            d = parse_1pz(open(p, "rb").read(), p)
            d.update(sample=sid, member=member, called=called)
            rows.append(d)
        sample_total[sid] = (called, side)
    cols = ["sample", "member", "called"] + FIELDS
    with open("results/track_a_account_files.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for d in rows:
            f.write("\t".join(str(d[c]) for c in cols) + "\n")
    print("\n### Per-file component accounting (bytes)\n")
    print("| sample | member | bytes | nnz | perm | ptr | ctbl+chdr | chunks | colsums | meta(rown_z3 / coln_z3 / kv_z3) | meta_z | names % | B/called cell |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in rows:
        names = d["meta_z"]
        print("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %d / %d / %d | %d | %.1f%% | %.0f |" % (
            d["sample"], d["member"].replace("_counts.1pz", ""), d["bytes"], d["nnz"], d["perm_z"], d["ptr_z"],
            d["ctable"] + d["chunk_hdr"], d["chunk_payload"], d["colsums_z"], d["rown_z3"], d["coln_z3"],
            d["kv_z3"], d["meta_z"], 100.0 * names / d["bytes"], d["bytes"] / max(d["called"], 1)))

    # ---------------- per-sample side files (corpus) ----------------
    print("\n### Corpus sample directories (exon+intron+side files, bytes)\n")
    print("| sample | called | n_barcodes | exon | intron | cell_calls.tsv | summary.json | feature_vocab.json (study-level) | exon+intron per called cell |")
    print("|---|---|---|---|---|---|---|---|---|")
    for sid, (called, side) in sample_total.items():
        ex = [r for r in rows if r["sample"] == sid and r["member"].startswith("exon")][0]
        it = [r for r in rows if r["sample"] == sid and r["member"].startswith("intron")][0]
        print("| %s | %d | %d | %d | %d | %d | %d | %d | %.0f |" % (
            sid, called, ex["n"], ex["bytes"], it["bytes"], side.get("cell_calls.tsv", 0),
            side.get("summary.json", 0), side.get("feature_vocab.json", 0), (ex["bytes"] + it["bytes"]) / max(called, 1)))

    # ---------------- bundle ----------------
    bpath = "corpus/bundles/GSE145370.singlet"
    z = zipfile.ZipFile(bpath)
    total = os.path.getsize(bpath)
    infos = z.infolist()
    by_type = {}
    per_sample = {}
    brows = []
    called_by = {}
    for i in infos:
        parts = i.filename.split("/")
        if len(parts) == 3:
            gsm, member = parts[1], parts[2]
        else:
            gsm, member = "(study)", parts[-1]
        typ = member
        by_type.setdefault(typ, [0, 0, 0])
        by_type[typ][0] += 1
        by_type[typ][1] += i.compress_size
        by_type[typ][2] += i.file_size
        per_sample.setdefault(gsm, {})[member] = i.compress_size
        if member == "cell_calls.tsv":
            called_by[gsm] = called_cells_from_tsv(z.read(i).decode())
        if member.endswith(".1pz"):
            d = parse_1pz(z.read(i), i.filename)
            d.update(sample=gsm, member=member)
            brows.append(d)
    zip_overhead = total - sum(i.compress_size for i in infos)
    print("\n### Bundle GSE145370.singlet: %d bytes total, %d entries, zip framing overhead %d bytes (%.4f%%)\n" % (
        total, len(infos), zip_overhead, 100.0 * zip_overhead / total))
    print("| member type | files | stored bytes (zip) | uncompressed | share of bundle | per sample (mean) |")
    print("|---|---|---|---|---|---|")
    for typ, (cnt, cs, fs) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
        print("| %s | %d | %d | %d | %.2f%% | %.0f |" % (typ, cnt, cs, fs, 100.0 * cs / total, cs / cnt))
    print("| zip framing (local hdrs + central dir) | | %d | | %.4f%% | |" % (zip_overhead, 100.0 * zip_overhead / total))

    # inside the .1pz matrices of the bundle: names/perm/etc by member type
    agg = {}
    for d in brows:
        a = agg.setdefault(d["member"], {k: 0 for k in FIELDS if k not in ("vt", "level", "chunk_cols", "chunks")})
        a.setdefault("files", 0)
        a["files"] += 1
        for k in a:
            if k in d and k != "files":
                a[k] += d[k]
    print("\n### Bundle .1pz members: component sums over the 14 samples (bytes)\n")
    print("| member | files | bytes | nnz | perm | ptr | chunk payload | colsums | meta_z | rownames z3 | colnames z3 | names+perm+colsums % of file |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mem, a in sorted(agg.items(), key=lambda kv: -kv[1]["bytes"]):
        ov = a["meta_z"] + a["perm_z"] + a["colsums_z"] + a["ptr_z"]
        print("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %d | %.1f%% |" % (
            mem, a["files"], a["bytes"], a["nnz"], a["perm_z"], a["ptr_z"], a["chunk_payload"], a["colsums_z"],
            a["meta_z"], a["rown_z3"], a["coln_z3"], 100.0 * ov / max(a["bytes"], 1)))

    print("\n### Bundle: bytes per sample and per called cell\n")
    print("| GSM | called cells | total (all members) | exon | intron | sj | psi | vdj | mt | json+tsv | B per called cell (all) | B per called cell (exon+intron) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    bl = []
    for gsm in sorted(k for k in per_sample if k != "(study)"):
        mem = per_sample[gsm]
        tot = sum(mem.values())
        ex, it = mem["exon_counts.1pz"], mem["intron_counts.1pz"]
        sj, ps, vd, mt = mem["sj_counts.1pz"], mem["splice_psi.1pz"], mem["vdj_gene_usage.1pz"], mem["mt_heteroplasmy.1pz"]
        other = tot - ex - it - sj - ps - vd - mt
        c = called_by[gsm]
        bl.append((gsm, c, tot, ex, it, sj, ps, vd, mt, other))
        print("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %.0f | %.0f |" % (gsm, c, tot, ex, it, sj, ps, vd, mt, other, tot / c, (ex + it) / c))
    st = sum(v for v in per_sample["(study)"].values())
    print("\nstudy-level members (feature_vocab.json, study_meta.json, manifest.json): %d bytes" % st)
    tc = sum(b[1] for b in bl)
    print("total called cells %d; bundle bytes per called cell %.0f" % (tc, total / tc))
    with open("results/track_a_account_bundle.tsv", "w") as f:
        f.write("gsm\tcalled\ttotal\texon\tintron\tsj\tpsi\tvdj\tmt\tother\n")
        for b in bl:
            f.write("\t".join(str(x) for x in b) + "\n")
    # extra: bundle 1pz details
    cols2 = ["sample", "member"] + FIELDS
    with open("results/track_a_account_bundle_1pz.tsv", "w") as f:
        f.write("\t".join(cols2) + "\n")
        for d in brows:
            f.write("\t".join(str(d[c]) for c in cols2) + "\n")
    # sanity: is corpus tumor_3p identical to bundle member?
    a = open("corpus/tumor_3p/exon_counts.1pz", "rb").read()
    b = z.read("samples/GSM4317429/exon_counts.1pz")
    print("\ncorpus/tumor_3p exon == bundle GSM4317429 exon: %s" % (a == b))
    # per-file detail of non-matrix members in bundle
    print("\n### Bundle .1pz per-member rows/cols/nnz (first sample GSM4317423)\n")
    for d in brows:
        if d["sample"] == "GSM4317423":
            print("%-24s bytes=%-8d m=%-7d n=%-6d nnz=%-8d vt=%d chunks=%d perm=%d ptr=%d chunk=%d colsums=%d meta=%d (rown_z3=%d coln_z3=%d)" % (
                d["member"], d["bytes"], d["m"], d["n"], d["nnz"], d["vt"], d["chunks"], d["perm_z"], d["ptr_z"],
                d["chunk_payload"], d["colsums_z"], d["meta_z"], d["rown_z3"], d["coln_z3"]))


if __name__ == "__main__":
    main()
