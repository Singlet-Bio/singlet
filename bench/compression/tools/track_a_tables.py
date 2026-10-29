#!/usr/bin/env python
"""Render the Track A result TSVs as markdown tables (pasted into notes/track_a.md).

    python -P tools/track_a_tables.py names|perm|sweep|run|profile|container
"""
import csv, os, sys
from collections import defaultdict

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def load(name):
    with open(os.path.join(R, name), newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def kb(x):
    return "%.0f" % (float(x) / 1000)


def names():
    rows = load("track_a_names.tsv")
    print("### Row names (identical for every sample of a reference build and matrix kind; measured once per list)\n")
    seen = {}
    for r in rows:
        if r["field"] == "rownames" and not r["scheme"].startswith("vocab"):
            seen.setdefault((r["member"], r["raw_bytes"]), {})[r["scheme"]] = r
    print("| list | names | raw bytes | zstd-3 (today) | zstd-19 | zstd-22 +27-bit window +LDM | xz -9e | brotli-11 w24 | bzip2 -9 | vocab id (8 B) + model (best coder) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    vocab = {(r["member"], r["raw_bytes"]): r for r in rows if r["scheme"].startswith("vocab")}
    for (mem, raw), d in seen.items():
        v = vocab.get((mem, raw))
        z = lambda k: "{:,}".format(int(d[k]["bytes"])) if k in d else ""
        print("| %s (%s) | | %s | %s | %s | %s | %s | %s | %s | 8 B + %s |" % (
            mem.replace("_counts.1pz", ""), next(iter(d.values()))["sample"], "{:,}".format(int(raw)), z("zstd3"), z("zstd19"), z("zstd22w27ldm"), z("xz9e"),
            z("brotli11w24"), z("bzip2"), "{:,}".format(int(v["model_bytes"])) if v else "n/a"))
    print("\nDecode time (ms, single thread, min of 3) of the same lists:\n")
    print("| list | zstd-3 | zstd-19 | xz | brotli | bzip2 |")
    print("|---|---|---|---|---|---|")
    for (mem, raw), d in seen.items():
        print("| %s | %s |" % (mem.replace("_counts.1pz", "") + " " + raw, " | ".join("%.0f" % float(d[k]["dec_ms_min"]) for k in ("zstd3", "zstd19", "xz9e", "brotli11w24", "bzip2"))))
    print("\n### Barcodes (colnames), per file; bytes. n = number of columns\n")
    by = defaultdict(dict)
    for r in rows:
        if r["field"] == "colnames" and r["member"].startswith("exon"):
            by[r["sample"]][r["scheme"]] = r
    cols = ["zstd3", "zstd19", "xz9e", "brotli11w24", "bzip2", "pack2bit_file_order", None, "whitelist_entropy_bound", "sorted_rice_delta+order_perm", "  of which sorted_rice_delta stream", "  of which order permutation", "sorted_set_bound(order canonicalised)"]
    print("| sample | n | raw | zstd-3 (today) | zstd-19 | xz | brotli | bzip2 | 2-bit packed, file order (4 B) | whitelist index | whitelist entropy bound | sorted Rice + order perm | (rice) | (order perm) | sorted-set bound if order were canonical |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    ns = {"pbmc_v2_typical": 15185, "pbmc_v2_shallow": 96446, "pbmc_v3_lowgenes": 71061, "brain_sn_v3_deep": 84424, "brain_sn_v3_intronic": 78894, "tcell_5p": 69249, "tumor_3p": 25067}
    for s, d in by.items():
        wl = [v for k, v in d.items() if k.startswith("whitelist_index")]
        cells = []
        for c in cols:
            if c is None:
                cells.append("{:,}".format(int(wl[0]["bytes"])) if wl else "")
            else:
                cells.append("{:,}".format(int(d[c]["bytes"])) if c in d else "")
        raw = next(iter(d.values()))["raw_bytes"]
        print("| %s | %d | %s | %s |" % (s, ns.get(s, 0), "{:,}".format(int(raw)), " | ".join(cells)))


def perm():
    rows = load("track_a_perm.tsv")
    print("| file | m | perm today (file) | row counts, best coder | row counts varint+zstd-3 | order-0 entropy of counts | chunk bytes: own order | fixed reference order (leave-one-study-out) | delta | no permutation | delta | ref-order model (zstd-3) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        own, ref, idn = int(r["chunks_own_perm"]), int(r["chunks_ref_perm"]), int(r["chunks_identity_perm"])
        print("| %s %s | %s | %s | %s | %s | %.0f | %s | %s (%s) | %+.1f%% | %s | %+.1f%% | %s |" % (
            r["sample"], r["member"].replace("_counts.1pz", ""), r["m"], "{:,}".format(int(r["perm_today_file"])), "{:,}".format(int(r["rowcounts_best"])),
            "{:,}".format(int(r["rowcounts_z3_plain"])), float(r["rowcounts_order0_entropy"]), "{:,}".format(own), "{:,}".format(ref), r["ref_training"],
            100.0 * (ref / own - 1), "{:,}".format(idn), 100.0 * (idn / own - 1), "{:,}".format(int(r["ref_order_model_z3"]))))


def sweep(fn="track_a_sweep.tsv"):
    rows = load(fn)
    bykey = defaultdict(dict)
    for r in rows:
        if r["coder"].startswith("parts"):
            continue
        bykey[(r["sample"] + " " + r["member"].replace("_counts.1pz", ""), r["chunk_cols"])][r["coder"]] = r
    coders = []
    for r in rows:
        if r["coder"] not in coders and not r["coder"].startswith("parts"):
            coders.append(r["coder"])
    print("Payload bits/nnz (sum of entropy-coder output, excl. 20-byte blob headers and chunk table). ok = matrix roundtrip\n")
    print("| file | chunk_cols | " + " | ".join(coders) + " |")
    print("|---|---|" + "---|" * len(coders))
    for (f, cc), d in bykey.items():
        print("| %s | %s | %s |" % (f, "whole" if cc == "0" or int(cc) > 20000 else cc, " | ".join(("%.3f" % float(d[c]["bits_per_nnz_payload"])) if c in d else "" for c in coders)))


def run():
    rows = load("track_a_run.tsv")
    codecs = []
    for r in rows:
        if r["codec"] not in codecs:
            codecs.append(r["codec"])
    by = defaultdict(dict)
    for r in rows:
        by[(r["sample"], r["member"])][r["codec"]] = r
    print("Total bits/nnz (and bytes) per codec, whole file including names; matrix_ok must be 1 everywhere (it is: %d of %d rows)\n" % (sum(1 for r in rows if r["matrix_ok"] == "1"), len(rows)))
    print("| file | nnz | original | " + " | ".join(codecs) + " |")
    print("|---|---|---|" + "---|" * len(codecs))
    for (s, m), d in by.items():
        any_ = next(iter(d.values()))
        print("| %s %s | %s | %s | %s |" % (s, m.replace("_counts.1pz", ""), any_["nnz"], "{:,}".format(int(any_["orig_bytes"])), " | ".join(
            ("%.2f (%s)%s" % (float(d[c]["bits_per_nnz"]), "{:,}".format(int(d[c]["enc_bytes"])), "" if d[c]["matrix_ok"] == "1" else " FAIL")) if c in d else "" for c in codecs)))
    print("\n(The wall-clock decode column of this run is not shown: the machine ran at 100% load from other jobs and the numbers were 2-6x too slow and irreproducible; see the cycle-count tables below.)")


def codec_summary():
    rows = load("track_a_run.tsv")
    by = defaultdict(dict)
    for r in rows:
        by[(r["sample"], r["member"])][r["codec"]] = r
    files = list(by)
    nnz = sum(int(by[f]["ref_1pz_v1_L3"]["nnz"]) for f in files)
    ref = sum(int(by[f]["ref_1pz_v1_L3"]["enc_bytes"]) for f in files)
    # payload-only from the sweep (chunk_cols 1024)
    sw = []
    for fn in ("track_a_sweep_exon.tsv", "track_a_sweep_intron.tsv"):
        sw += load(fn)
    pay = defaultdict(dict)
    for r in sw:
        if r["chunk_cols"] == "1024":
            pay[(r["sample"], r["member"])][r["coder"]] = int(r["payload_bytes"])
    def payload(coder, fs):
        return sum(pay[f][coder] for f in fs if coder in pay[f]), sum(int(by[f]["ref_1pz_v1_L3"]["nnz"]) for f in fs if coder in pay[f])
    p3, n3 = payload("z3", files)
    spd = {}
    if os.path.exists(os.path.join(R, "track_a_speed_full.tsv")):
        for r in load("track_a_speed_full.tsv"):
            spd[(r["name"], r["sample"] + " " + r["member"].replace("_counts.1pz", ""))] = float(r["dec_cycles_per_nnz"])
    if os.path.exists(os.path.join(R, "track_a_speed_ref.tsv")):
        for r in load("track_a_speed_ref.tsv"):
            spd[("ref_1pz_v1_L3", r["sample"] + " " + r["member"].replace("_counts.1pz", ""))] = float(r["dec_cycles_per_nnz"])
    dense, sparse = "tcell_5p exon", "tumor_3p intron"
    spec = [
        ("ref_1pz_v1_L3", "production writer, re-emitted (zstd 1.5.5, level 3, 1024 cols)", "z3", 0),
        ("a_names_as_model", "row names -> vocabulary id; barcodes -> whitelist index", "z3", None),
        ("a_perm_from_rowcounts", "permutation block -> per-row nnz counts", "z3", 0),
        ("a_names_perm_model", "both + no ptr block", "z3", None),
        ("a_vocsc_zstd19", "chunk entropy coder zstd-19", "z19", 0),
        ("a_vocsc_zstd22long", "zstd-22, window 2^27, LDM", "z22l", 0),
        ("a_vocsc_xz", "xz -9e (raw LZMA2)", "xz", 0),
        ("a_vocsc_brotli", "brotli-11, window 2^24", "br", 0),
        ("a_vocsc_zstd_dict", "zstd-19 + out-of-study dictionary (10 files; others plain zstd-19)", "zd19", 0),
    ]
    mb = {"a_names_as_model": "3.6 MB (human) + 4.5 MB (mouse) vocab, 3-6 KB whitelist", "a_names_perm_model": "same", "a_vocsc_zstd_dict": "dict 10 KB - 1 MB per held-out sample"}
    print("| codec | change vs production | total bytes, 14 files | total bits/nnz | vs production | payload-only bits/nnz (chunks) | payload vs zstd-3 | model | decode cycles/nnz: dense tcell_5p exon / sparse tumor_3p intron |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, what, coder, _ in spec:
        tot = sum(int(by[f][name]["enc_bytes"]) for f in files)
        pb, pn = payload(coder, files)
        base_pb, base_pn = payload("z3", [f for f in files if coder in pay[f]])
        c1, c2 = spd.get((name, dense)), spd.get((name, sparse))
        print("| %s | %s | %s | %.2f | %.3f | %.3f | %.3f | %s | %s |" % (name, what, "{:,}".format(tot), tot * 8.0 / nnz, tot / ref, pb * 8.0 / pn, (pb / pn) / (base_pb / base_pn), mb.get(name, "-"),
                                                                      ("%.0f / %.0f" % (c1, c2)) if c1 and c2 else "n/a"))
    print("\nProduction payload (zstd-3 chunks) over the same 14 files: %.3f bits/nnz; total file size under the production writer: %.2f bits/nnz (%d files, %s nnz)." % (p3 * 8.0 / n3, ref * 8.0 / nnz, len(files), "{:,}".format(nnz)))


def speed():
    out = []
    for mode in ("ref", "full", "entropy"):
        fn = "track_a_speed_%s.tsv" % mode
        if not os.path.exists(os.path.join(R, fn)):
            continue
        rows = load(fn)
        if mode == "ref":
            print("**Production reader (`read_1pz_buffer`) on the original files**, thread CPU cycles per nnz (min of 5) and Mnnz/s at 3.0 GHz.\n")
            print("| file | nnz | cycles/nnz | Mnnz/s @3GHz |")
            print("|---|---|---|---|")
            for r in rows:
                print("| %s %s | %s | %.0f | %.2f |" % (r["sample"], r["member"].replace("_counts.1pz", ""), "{:,}".format(int(r["nnz"])), float(r["dec_cycles_per_nnz"]), float(r["dec_Mnnz_s_at_3GHz"])))
        elif mode == "full":
            names = []
            for r in rows:
                if r["name"] not in names:
                    names.append(r["name"])
            by = defaultdict(dict)
            files = []
            for r in rows:
                k = (r["sample"], r["member"].replace("_counts.1pz", ""))
                if k not in files:
                    files.append(k)
                by[k][r["name"]] = r
            print("\n**Whole-codec decode** (names, permutation, chunks, re-sort: everything `decode()` does), cycles/nnz and Mnnz/s @3GHz.\n")
            print("| file | " + " | ".join(names) + " |")
            print("|---|" + "---|" * len(names))
            for k in files:
                print("| %s %s | %s |" % (k[0], k[1], " | ".join(("%.0f (%.1f)" % (float(by[k][n]["dec_cycles_per_nnz"]), float(by[k][n]["dec_Mnnz_s_at_3GHz"]))) if n in by[k] else "" for n in names)))
        else:
            names = []
            for r in rows:
                if r["name"] not in names:
                    names.append(r["name"])
            by = defaultdict(dict)
            files = []
            for r in rows:
                k = (r["sample"], r["member"].replace("_counts.1pz", ""))
                if k not in files:
                    files.append(k)
                by[k][r["name"]] = r
            print("\n**Entropy stage alone** (decompress every chunk, chunk_cols 1024), cycles per nnz (payload bits/nnz in brackets; 'store' = memcpy floor).\n")
            print("| file | " + " | ".join(names) + " |")
            print("|---|" + "---|" * len(names))
            for k in files:
                print("| %s %s | %s |" % (k[0], k[1], " | ".join(("%.1f (%.2f)" % (float(by[k][n]["dec_cycles_per_nnz"]), float(by[k][n]["bytes"]) * 8.0 / float(by[k][n]["nnz"]))) if n in by[k] else "" for n in names)))


def profile():
    rows = load("track_a_profile.tsv")
    by = defaultdict(list)
    for r in rows:
        by[(r["sample"], r["member"])].append(r)
    for (s, m), rs in by.items():
        print("\n**%s %s**  nnz=%s\n" % (s, m, rs[0]["nnz"]))
        print("| stage | ms | ns/nnz | % |")
        print("|---|---|---|---|")
        for r in rs:
            print("| %s | %.2f | %.2f | %.1f |" % (r["stage"], float(r["ms"]), float(r["ns_per_nnz"]), float(r["pct_of_stage_sum"])))


def account():
    rows = load("track_a_account_files.tsv")
    print("| file | bytes | nnz | names (meta_z) | perm | colsums | ptr | chunks | other | names % | perm % | colsums % | chunks % | bits/nnz total | bits/nnz chunks | bytes per called cell |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        b = int(r["bytes"]); nnz = int(r["nnz"])
        other = b - sum(int(r[k]) for k in ("perm_z", "ptr_z", "meta_z", "colsums_z", "chunk_payload"))
        pc = lambda k: 100.0 * int(r[k]) / b
        print("| %s %s | %s | %s | %s | %s | %s | %s | %s | %s | %.1f | %.1f | %.1f | %.1f | %.2f | %.2f | %.0f |" % (
            r["sample"], r["member"].replace("_counts.1pz", ""), "{:,}".format(b), "{:,}".format(nnz), "{:,}".format(int(r["meta_z"])), "{:,}".format(int(r["perm_z"])),
            "{:,}".format(int(r["colsums_z"])), "{:,}".format(int(r["ptr_z"])), "{:,}".format(int(r["chunk_payload"])), "{:,}".format(other),
            pc("meta_z"), pc("perm_z"), pc("colsums_z"), pc("chunk_payload"), b * 8.0 / nnz, int(r["chunk_payload"]) * 8.0 / nnz, b / max(int(r["called"]), 1)))


if __name__ == "__main__":
    {"names": names, "perm": perm, "sweep": sweep, "run": run, "profile": profile, "account": account, "codec_summary": codec_summary, "speed": speed}[sys.argv[1]]()
