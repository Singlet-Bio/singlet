#!/usr/bin/env python
"""w4_corpus: sample master table, row-name invariance per (build, member), barcode statistics and whitelist membership.

    python -P tools/w4_corpus_names.py

Reads corpus_w4_corpus/<id>/{exon,intron}_counts.1pz headers + metadata blocks, summary.json, cell_calls.tsv (never executes anything).
Writes
  results/w4_corpus_samples.tsv      one row per sample: protocol, species, build, matrix shapes, called cells, layout class
  results/w4_corpus_rownames.tsv     one row per (sample, member): m, hash of the NUL-joined row names, variant index within (build, member)
  results/w4_corpus_variants.tsv     one row per (build, member, hash): number of samples, rows, example sample
  results/w4_corpus_barcodes.tsv     one row per (sample, member): barcode length/alphabet/duplicates, whitelist membership
  results/w4_corpus_vocab_manifest.tsv  (build, kind, variant, hash, file) used by harness/w4_corpus_models.cpp
Whitelists are those of Track A (results/track_a_whitelist_*.bin, format in kernels/track_a.h).
"""
import hashlib, json, os, struct, sys, glob, csv, collections
import numpy as np
import zstandard

HERE = os.path.dirname(os.path.abspath(__file__))
BC = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import e4_lib as L  # read-only helpers: parse_pz, zdec, parse_tlv
CORP = os.path.join(BC, "corpus_w4_corpus")
RES = os.path.join(BC, "results")
MEMBERS = ["exon_counts.1pz", "intron_counts.1pz"]


def load_whitelist(path):
    b = open(path, "rb").read()
    assert b[:8] == b"TAWL0001"
    nl = struct.unpack_from("<H", b, 8)[0]
    name = b[10:10 + nl].decode()
    p = 10 + nl
    count, zlen, rawlen = struct.unpack_from("<IQQ", b, p)
    p += 4 + 8 + 8
    raw = zstandard.ZstdDecompressor().decompress(b[p:p + zlen], max_output_size=rawlen + 16)
    # delta-1 varints (LEB128), first value absolute
    vals = []
    i = 0
    n = len(raw)
    arr = np.frombuffer(raw, dtype=np.uint8)
    # vectorised varint decode
    cont = (arr & 0x80) != 0
    ends = np.flatnonzero(~cont)
    starts = np.concatenate([[0], ends[:-1] + 1])
    out = np.zeros(len(ends), dtype=np.uint64)
    maxlen = int((ends - starts).max()) + 1
    for k in range(maxlen):
        idx = starts + k
        valid = idx <= ends
        out[valid] |= (arr[idx[valid]] & 0x7F).astype(np.uint64) << np.uint64(7 * k)
    codes = np.cumsum(np.concatenate([[out[0]], out[1:] + 1])).astype(np.uint64)
    assert len(codes) == count, (len(codes), count)
    return name, codes


LUT = np.full(256, 255, dtype=np.uint8)
for i, ch in enumerate(b"ACGT"):
    LUT[ch] = i


def barcodes_to_codes(cols):
    """16-mer ACGT barcodes -> uint64 codes (first base most significant); None if any is not a 16-mer ACGT string."""
    if not cols or any(len(c) != 16 for c in cols):
        return None
    a = np.frombuffer("".join(cols).encode("ascii", "replace"), dtype=np.uint8).reshape(-1, 16)
    m = LUT[a]
    if (m == 255).any():
        return None
    code = np.zeros(len(cols), dtype=np.uint64)
    for i in range(16):
        code = (code << np.uint64(2)) | m[:, i].astype(np.uint64)
    return code


def read_meta(path):
    buf = open(path, "rb").read()
    h, o = L.parse_pz(buf)
    meta = L.parse_tlv(L.zdec(buf, o["meta_off"], h["metadata_z_sz"])) if h["metadata_z_sz"] else {"rownames": [], "colnames": [], "kv": b""}
    return h, o, meta, len(buf), buf


def main():
    ids = sorted(d for d in os.listdir(CORP) if not d.startswith("_") and os.path.isdir(os.path.join(CORP, d)))
    wls = {}
    for p in sorted(glob.glob(os.path.join(RES, "track_a_whitelist_*.bin"))):
        nm, codes = load_whitelist(p)
        wls[nm] = (codes, set())  # sorted codes
        assert np.all(np.diff(codes.astype(np.int64)) > 0)
    print("whitelists:", {k: len(v[0]) for k, v in wls.items()}, flush=True)
    members_rows = {r["id"]: {} for r in csv.DictReader(open(os.path.join(RES, "w4_corpus_members.tsv")), delimiter="\t") if r["id"] != "-"}
    for r in csv.DictReader(open(os.path.join(RES, "w4_corpus_members.tsv")), delimiter="\t"):
        if r["id"] != "-":
            members_rows[r["id"]][r["member"]] = (int(r["csize"]), int(r["usize"]))
    sample_rows, name_rows, bc_rows = [], [], []
    by_variant = collections.OrderedDict()   # (build, member) -> OrderedDict(hash -> dict)
    all_bc = collections.defaultdict(list)   # protocol -> list of code arrays (for cross-sample reuse check)
    for sid in ids:
        d = os.path.join(CORP, sid)
        sj = json.load(open(os.path.join(d, "summary.json")))
        sample = json.load(open(os.path.join(d, "sample.json")))
        mem = members_rows.get(sid, {})
        layout = "A10" if len(mem) <= 10 else ("B%d" % len(mem))
        n_called = sum(1 for _ in open(os.path.join(d, "cell_calls.tsv"))) - 1
        row = dict(id=sid, gse=sample["gse"], gsm=sample["gsm"], protocol=sj.get("protocol_name"), species=sj.get("species"),
                   build=sj.get("reference_build"), n_members=len(mem), layout=layout,
                   n_cells_called=sj.get("n_cells_called"), n_called_rows=n_called, median_umi=sj.get("median_umi_per_cell"),
                   median_genes=sj.get("median_genes_per_cell"), n_input_reads=sj.get("n_input_reads"),
                   singlet_version=sj.get("singlet_version"), schema=sj.get("schema_version"),
                   bundle_sample_bytes=sum(v[0] for v in mem.values()))
        exon_cols = []
        for mname in MEMBERS:
            k = mname.split("_")[0]
            p = os.path.join(d, mname)
            if not os.path.exists(p):
                row.update({f"{k}_bytes": "", f"{k}_m": "", f"{k}_n": "", f"{k}_nnz": ""})
                continue
            h, o, meta, nbytes, buf = read_meta(p)
            row.update({f"{k}_bytes": nbytes, f"{k}_m": h["m"], f"{k}_n": h["n"], f"{k}_nnz": h["nnz"]})
            rn, cn = meta["rownames"], meta["colnames"]
            hh = hashlib.sha1("\0".join(rn).encode()).hexdigest()[:16]
            bk = (row["build"], mname)
            vd = by_variant.setdefault(bk, collections.OrderedDict())
            if hh not in vd:
                vd[hh] = dict(variant=len(vd), samples=[], rows=len(rn), example=sid, first=rn[0] if rn else "", last=rn[-1] if rn else "")
            vd[hh]["samples"].append(sid)
            name_rows.append(dict(id=sid, member=mname, build=row["build"], protocol=row["protocol"], m=h["m"], n_rows_listed=len(rn),
                                  names_hash=hh, variant=vd[hh]["variant"], meta_z=h["metadata_z_sz"], perm_z=h["perm_z_sz"], ptr_z=h["ptr_z_sz"],
                                  colsums_z=h["colsums_z_sz"], num_chunks=h["num_chunks"], chunk_cols=h["chunk_cols"], vt_code=h["vt_code"],
                                  file_bytes=nbytes, nnz=h["nnz"], n=h["n"], flags=h["flags"], version=h["version"], ptr_width=h["ptr_width"]))
            # barcodes
            codes = barcodes_to_codes(cn)
            lens = collections.Counter(len(c) for c in cn)
            alpha = "".join(sorted(set("".join(cn[:2000]))))[:12] if cn else ""
            b = dict(id=sid, member=mname, protocol=row["protocol"], species=row["species"], n=len(cn), len_hist=json.dumps(dict(lens)),
                     alphabet_sample=alpha, n_distinct=len(set(cn)), acgt16=int(codes is not None), first_bc=cn[0] if cn else "")
            if codes is not None:
                for nm, (wc, _) in wls.items():
                    pos = np.searchsorted(wc, codes)
                    pos[pos >= len(wc)] = len(wc) - 1
                    b["in_" + nm] = int((wc[pos] == codes).sum())
                b["in_any"] = int(np.zeros(len(codes), bool).sum())
                inany = np.zeros(len(codes), bool)
                for nm, (wc, _) in wls.items():
                    pos = np.searchsorted(wc, codes)
                    pos[pos >= len(wc)] = len(wc) - 1
                    inany |= (wc[pos] == codes)
                b["in_any"] = int(inany.sum())
                if mname == "exon_counts.1pz":
                    all_bc[row["protocol"]].append((sid, codes))
                # monotonic in raw code order (is the column order the whitelist/sorted order?)
                b["frac_increasing"] = round(float((np.diff(codes.astype(np.int64)) > 0).mean()), 4) if len(codes) > 1 else ""
            else:
                for nm in wls:
                    b["in_" + nm] = 0
                b["in_any"] = 0
                b["frac_increasing"] = ""
            bc_rows.append(b)
            # colnames identical between exon and intron?
            if mname == "exon_counts.1pz":
                exon_cols = cn
            else:
                row["colnames_exon_eq_intron"] = int(cn == exon_cols)
        sample_rows.append(row)
        print(sid, row["protocol"], row["species"], row["build"], row.get("exon_n"), row.get("exon_nnz"), flush=True)

    def wtsv(path, rows, cols=None):
        cols = cols or list(rows[0].keys())
        allk = []
        for r in rows:
            for k in r:
                if k not in allk:
                    allk.append(k)
        cols = cols if cols else allk
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(allk)
            for r in rows:
                w.writerow([r.get(k, "") for k in allk])
    wtsv(os.path.join(RES, "w4_corpus_samples.tsv"), sample_rows)
    wtsv(os.path.join(RES, "w4_corpus_rownames.tsv"), name_rows)
    wtsv(os.path.join(RES, "w4_corpus_barcodes.tsv"), bc_rows)
    vr, mf = [], []
    for (build, mname), vd in by_variant.items():
        for hh, v in vd.items():
            vr.append(dict(build=build, member=mname, names_hash=hh, variant=v["variant"], n_samples=len(v["samples"]), rows=v["rows"],
                           example=v["example"], first_name=v["first"], last_name=v["last"], samples=",".join(v["samples"])))
            mf.append(dict(build=build, kind=mname.split("_")[0] + ("" if v["variant"] == 0 else f"#{v['variant']}"), variant=v["variant"], hash=hh,
                           file=f"corpus_w4_corpus/{v['example']}/{mname}"))
    wtsv(os.path.join(RES, "w4_corpus_variants.tsv"), vr)
    wtsv(os.path.join(RES, "w4_corpus_vocab_manifest.tsv"), mf)
    # cross-sample barcode reuse: for each protocol, fraction of a sample's barcodes seen in any other sample of the protocol
    reuse = []
    for prot, lst in all_bc.items():
        for sid, c in lst:
            others = [x for s2, x in lst if s2 != sid]
            if not others:
                reuse.append(dict(protocol=prot, id=sid, n=len(c), n_other_samples=0, frac_in_other=""))
                continue
            u = np.unique(np.concatenate(others))
            pos = np.searchsorted(u, c); pos[pos >= len(u)] = len(u) - 1
            reuse.append(dict(protocol=prot, id=sid, n=len(c), n_other_samples=len(others), frac_in_other=round(float((u[pos] == c).mean()), 4)))
    wtsv(os.path.join(RES, "w4_corpus_barcode_reuse.tsv"), reuse)
    print("variants:")
    for r in vr:
        print(" ", r["build"], r["member"], r["names_hash"], "variant", r["variant"], "samples", r["n_samples"], "rows", r["rows"])


if __name__ == "__main__":
    main()
