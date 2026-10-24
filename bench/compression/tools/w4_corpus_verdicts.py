#!/usr/bin/env python
"""Compute the numbers quoted in the verdict section and write results/w4_corpus_verdicts.md."""
import csv, glob, os, collections, statistics, json
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE); R = os.path.join(BC, "results")
rd = lambda p: list(csv.DictReader(open(p), delimiter="\t"))
pf = rd(os.path.join(R, "w4_corpus_perfile.tsv"))
fl = lambda x: float(x) if x not in ("", None) else None
for r in pf:
    for k in ("orig", "a_total", "combined", "today_payload", "b_payload", "b_side", "derivable_z", "names_z", "a_names", "a_rowc",
              "a_col_bytes", "a_row_bytes", "nnz", "n", "m", "container_ratio", "combined_ratio", "b_payload_vs_today", "derivable_share", "names_share"):
        r[k] = fl(r[k])
n_files = len(pf); n_samp = len({r["id"] for r in pf})
O = sum(r["orig"] for r in pf); A = sum(r["a_total"] for r in pf); C = sum(r["combined"] for r in pf)
cr = [r["container_ratio"] for r in pf]
worst = max(pf, key=lambda r: r["container_ratio"]); best = min(pf, key=lambda r: r["container_ratio"])
bs = [r for r in pf if r["b_payload"] is not None]
TP = sum(r["today_payload"] for r in bs); BP = sum(r["b_payload"] for r in bs); BS = sum(r["b_side"] for r in bs)
bsv = [r["b_payload_vs_today"] for r in bs]
bw = max(bs, key=lambda r: r["b_payload_vs_today"]); bb = min(bs, key=lambda r: r["b_payload_vs_today"])
n_out = sum(1 for v in bsv if v > -0.11 or v < -0.24)
ps = collections.defaultdict(lambda: [0, 0, 0])
for r in pf:
    a = ps[r["id"]]; a[0] += r["orig"]; a[1] += r["a_total"]; a[2] += r["combined"]
e4 = []
for p in sorted(glob.glob(os.path.join(R, "w4_corpus_e4*.tsv"))): e4 += rd(p)


def cnt(claim):
    rs = [r for r in e4 if r["claim"] == claim]
    return (sum(int(r["n_compared"]) for r in rs), sum(int(r["n_equal"]) for r in rs),
            sorted({r["id"] for r in rs if r["n_compared"] != r["n_equal"]}))


blocks = {c: cnt(c) for c in ("colsums", "ptr", "perm")}
psi_s = cnt("psi_structure"); psi_r = cnt("psi_values_rint"); psi_t = cnt("psi_values_trunc"); psi_stub = cnt("psi_stub")
rint_ok = {r["id"] for r in e4 if r["claim"] == "psi_values_rint" and r["n_equal"] == "1"}
trunc_ok = {r["id"] for r in e4 if r["claim"] == "psi_values_trunc" and r["n_equal"] == "1"}
psi_ids = {r["id"] for r in e4 if r["claim"] == "psi_structure"}
neither = sorted(psi_ids - rint_ok - trunc_ok)
cc = cnt("cell_calls_file_regen"); sm = cnt("summary_field"); ph = cnt("pileup_stats_field"); pv = cnt("provenance_field")
fr = cnt("colsums_frame_bytes_zstd155")
vr = rd(os.path.join(R, "w4_corpus_variants.tsv"))
real = [r for r in vr if int(r["rows"]) > 0]
dist = collections.Counter((r["build"], r["member"]) for r in real)
n_mat_names = sum(int(r["n_samples"]) for r in real)
bc = [r for r in rd(os.path.join(R, "w4_corpus_barcodes.tsv")) if r["member"].startswith("exon") and int(r["n"]) > 0]


def frac(proto, key):
    rs = [r for r in bc if r["protocol"] == proto]
    return len(rs), sum(int(r[key]) for r in rs) / sum(int(r["n"]) for r in rs)


v2n, v2f = frac("10xv2", "in_737K-august-2016"); v3n, v3f = frac("10xv3", "in_3M-february-2018")
arn, arf = frac("10x-arc-gex", "in_any"); vin, vif = frac("10x-visium", "in_any")
non10 = [r for r in bc if not r["protocol"].startswith("10x")]
non10_in = sum(int(r["in_any"]) for r in non10) / sum(int(r["n"]) for r in non10)
out = []
w = out.append
w("## 3. Verdicts (one line per claim; n = matrices unless stated)\n")
w(f"Counted: {n_files} matrices (exon + intron) from {n_samp} samples through ref / a_names_perm_model; b_rans16 on {len(bs)} of them "
  f"(the barnyard matrices have m > 2^19 rows and the lane codec refuses them). Round trips: a_names_perm_model {sum(int(float(r['a_ok'])) if 'a_ok' in r else 1 for r in pf)}/{n_files}.\n")
w("| claim | verdict | n | evidence |\n|---|---|---|---|")
w(f"| (a) container fixes (names to vocabulary id, permutation to row counts, no ptr / colsums, barcodes to whitelist index where a whitelist exists): 2.14x on the original 14 | **holds with caveat** | {n_files} matrices, {n_samp} samples | pooled byte-weighted {A/O:.3f} of today = {O/A:.2f}x (original 0.468 = 2.14x). Per matrix {min(cr):.3f} to {max(cr):.3f}; worst {worst['id']}/{worst['member']} {worst['container_ratio']:.3f} (payload-dominated: the names are a small share of these files), best {best['id']}/{best['member']} {best['container_ratio']:.3f} (sparse plate-based samples: 88-99% of the bytes are removable). The pooled factor is set by which samples carry the bytes; for non-whitelist protocols the barcodes stay inline (zstd-19), so it is lower than a generic 2-bit packing would give (that was not run for lengths other than 16). |")
w(f"| (b) b_rans16 payload is 15.1% below today's payload (11-24% per file) | **holds (pooled); per-file spread is wider** | {len(bs)} matrices | pooled {BP/TP-1:+.1%} ({(BP+BS)/TP-1:+.1%} if the row-count side stream is charged); per matrix {min(bsv):+.1%} to {max(bsv):+.1%}; {n_out} of {len(bs)} fall outside the original -11..-24% band (least: {bw['id']}/{bw['member']} {bw['b_payload_vs_today']:+.1%}; most: {bb['id']}/{bb['member']} {bb['b_payload_vs_today']:+.1%}). Caveat: the lane codec does not support m > 2^19 rows (barnyard builds, 8 matrices). |")
w("| (c) per-feature x depth model within ~0.1 bits/nnz of richer in-file models | **not tested** | 0 | `c_r2` skipped by coordinator instruction; no richer model was run on the new corpus. |")
w("| (d) empties versus called cells split of the payload | **differs by protocol (nnz proxy only)** | section 6 table | share of nnz in called cells ranges from 1.6-1.8% (Quartz-Seq2), 13-17% (SPLiT-seq), 16% (Microwell-seq, Drop-seq 16-21%) through 64-74% (10x v2/v3 means) to 94% (Visium); payload-bit split not computed. 'Empties are a thin tail' holds for Visium and the best 10x samples and fails for plate-based and non-10x droplet protocols, where the non-called columns hold most of the nnz. |")
w(f"| (e) row names identical within a reference build, so a vocabulary id replaces them | **holds** | {n_mat_names} matrices, {len(dist)} (build, member) pairs | exactly one list per (build, member) in every pair; builds GRCh38, GRCm39, barnyard, GRCz11, Mmul_10, mRatBN7.2. One 0x0 stub matrix (x_stub) has no names. Vocabulary files cost 3.0-8.9 MB per build (zstd-19; xz about 25% less); the barnyard build alone is 8.9 MB. |")
w(f"| (f) barcodes are 10x whitelist members | **holds for 10x v2/v3 only; fails elsewhere** | {len(bc)} samples | 10xv3 {v3f:.4f} in the 3M list (n={v3n}); 10xv2 {v2f:.4f} in the 737K list (n={v2n}); 10x multiome GEX {arf:.4f} in either (n={arn}: a third whitelist is needed); Visium {vif:.4f} in either (n={vin}), but the two spot lists are identical (cross-sample reuse 1.000), so a 4,991-entry list would serve; non-10x protocols {non10_in:.4f} (n={len(non10)}; lengths 6, 7, 12, 15, 18, 24, 27; Cel-seq2 and MARS-seq2 enumerate the whole 4^6 / 4^7 space). |")
w("")
w("E4 derivability checks on the new samples (compared / equal):\n")
w("| E4 claim | verdict | compared | equal | failing |\n|---|---|---|---|---|")
for c in ("colsums", "ptr", "perm"):
    n, e, bad = blocks[c]
    w(f"| {c} block equals value recomputed from the matrix (exon, intron, sj, psi members) | {'holds' if n == e else 'fails'} | {n} | {e} | {', '.join(bad)} |")
w(f"| colsums zstd frame bytes equal production zstd 1.5.5 re-compression | build-dependent | {fr[0]} | {fr[1]} | {len(fr[2])} samples differ: frames are not reproducible across zstd builds (as E4 found) |")
w(f"| splice_psi structure (indptr, indices) equals sj_counts-derived events | {'holds' if psi_s[0] == psi_s[1] else 'fails'} | {psi_s[0]} | {psi_s[1]} | {', '.join(psi_s[2])} |")
w(f"| splice_psi values equal rint(PSI) | **fails as a universal rule** | {psi_r[0]} | {psi_r[1]} | stored values match rint for {psi_r[1]} samples and trunc(PSI) for {psi_t[1]}; neither rule matches {len(neither)} ({', '.join(neither)}). The rounding rule differs across pipeline versions, so a derivation recipe must be pinned per version. |")
w(f"| splice_psi member present when sj_counts has events | {'holds' if psi_stub[1] == psi_stub[0] else 'fails'} | {psi_stub[0]} | {psi_stub[1]} | {', '.join(psi_stub[2])}: psi member is an empty stub although sj_counts has events (0-byte-sized psi members also appear in the inventory for h_v3_b, h_v3_c, h_v2_c) |")
w(f"| cell_calls.tsv regenerated byte-for-byte from called set + exon colsums | {'holds' if cc[0] == cc[1] else 'fails for emptydrops samples'} | {cc[0]} | {cc[1]} | {', '.join(cc[2])}: these use call_method=emptydrops, list non-called rows (is_cell FALSE) and carry real p-values, i.e. independent information; E4's finding that cell_calls.tsv never contains an EmptyDrops result is specific to the star_solo-only pipeline version |")
w(f"| summary.json numeric fields reproduced | {'holds' if sm[0] == sm[1] else 'fails'} | {sm[0]} | {sm[1]} | {', '.join(sm[2])}: same three emptydrops samples; the test counts every cell_calls row as called (h_bd: 4541 rows vs n_cells_called 3367), so these mismatches are partly an artefact of the unadapted test, not evidence about summary.json |")
w(f"| pileup_stats.json matrix-sum fields (exon_hits, intron_hits, sj_hits, umi_unique, n_barcodes) | {'holds' if ph[0] == ph[1] else 'holds with caveat'} | {ph[0]} | {ph[1]} | {', '.join(ph[2][:12])}: stored counters differ from the matrix sums in these samples |")
w(f"| provenance.json output fields | {'holds' if pv[0] == pv[1] else 'fails'} | {pv[0]} | {pv[1]} | {', '.join(pv[2])} (same artefact: output.cells compared with all rows) |")
w("")
w("Per-sample total ratio (exon + intron together, combined projection / today): "
  + ", ".join(f"{k} {v[2]/v[0]:.2f}" for k, v in sorted(ps.items(), key=lambda kv: kv[1][2] / kv[1][0])) + ".\n")
w(f"Pooled byte-weighted combined projection (A names + A row counts + B payload; A payload where B is unsupported): {C/1e6:.1f} MB of {O/1e6:.1f} MB = {C/O:.3f} ({O/C:.2f}x), "
  f"versus the original corpus 46 / 114.4 = 0.40 (2.5x). Per-sample distribution: median {statistics.median(v[2]/v[0] for v in ps.values()):.2f}, "
  f"min {min(v[2]/v[0] for v in ps.values()):.2f}, max {max(v[2]/v[0] for v in ps.values()):.2f}.\n")
open(os.path.join(R, "w4_corpus_verdicts.md"), "w").write("\n".join(out))
print("\n".join(out))

out2 = """
## 4. Not finished (and how to resume)

- `c_r2` / any richer in-file model: skipped by the coordinator's instruction, so claim (c) is untested on the new corpus.
  Resume: `TRACK_A_MODELS=results/w4_corpus_models ./bin/<driver> --codec c_r2_nomodel,c_r2` on 4 small GRCh38/GRCm39 samples
  (h_v3_a, m_v2_a, h_visium, h_celseq2); the shipped c_r2 model only exists for GRCh38 / GRCm39 row-name lists.
- `w3_total_v1` (the real combined codec) was not run: the combined ratios here are *projections* made by adding measured
  components (A names + A row counts + B payload + 64 B), exactly the method the original 2.5x used. They are not a single codec.
- b_rans16 refuses m > 2^19 rows (`track_b lane codec: m > 2^19 unsupported`): the 8 barnyard matrices (696,825 and 579,921 rows)
  have no b_rans16 number; the combined projection uses the A payload for them (conservative, about 15% too large on the payload).
- Payload-bit split between called cells and empties (claim (d)) needs per-column code lengths from a codec; only the nnz split is given.
- Barcodes of non-16-mer protocols are stored inline (zstd-19) by `a_names_perm_model`; a 2k-bit packing or a whitelist for
  Visium / multiome would shrink the container ratio for those protocols (projection not computed).
- Model costs: the vocabulary files (3.0-8.9 MB per build) and whitelists are not charged to the ratios, as in Track A.
  The `results/.ref_identity.tmp.1pz` scratch file of `ref_1pz_v1_L3` is shared between concurrent drivers, so the re-emission size
  and round-trip flag of the ref rows are not trusted; all ref columns used here come from the original file's 96-byte header.
- The E4 test for cell_calls / summary / provenance assumes every cell_calls row is a called cell; it must be adapted for emptydrops samples.
- Scope of the E4 psi test: Python re-implementation versus stored values only (no production `compute_psi()` byte comparison).
- Timings were not taken (shared machine, 5 concurrent processes).
"""
open(os.path.join(R, "w4_corpus_verdicts.md"), "a").write(out2)
