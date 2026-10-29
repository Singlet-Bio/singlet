"""Feature-barcode kernels (phase 1f) and the Visium spot gather against the Python reference, on real reads:
  ADT   k_bl_adt_match   R2 of the public 10x pbmc_1k_protein_v3 antibody library
  guide k_bl_guide_match real cDNA reads (pbmc1k_v3 R2); the guides are 20-mers taken from real reads, plus a duplicate, a nested guide and a 60-mer
  spot  k_bl_spot_gather Visium v1 spot list against real 10x barcodes and Visium barcodes
The tag table for ADT is taken from the data (the 17 most frequent 15-mers at offset 0); two synthetic tags are added so that the
Hamming-1 ambiguity code (-2) is exercised. The reads are real; the table is labelled as built here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src" / "engine_gpu"))
import bl_io as io  # noqa: E402
import barcode_layer_ref as ref  # noqa: E402

W = "/mnt/projects/debruinz_project/debruinz/singlet"
ADT = (f"{W}/pipeline-benchmark/data/pbmc1k_cite_v3/pbmc_1k_protein_v3_fastqs/pbmc_1k_protein_v3_antibody_fastqs/"
       "pbmc_1k_protein_v3_antibody_S2_L001_R2_001.fastq.gz")
R2 = f"{W}/data/barcode-layer/pbmc1k_v3/R2.fq"
VISIUM_COORD = f"{W}/pipeline-benchmark/whitelists/visium-v1_coordinates.tsv"
WL10X = f"{W}/pipeline-benchmark/whitelists/3M-february-2018.txt"


def sh(cmd):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"$ {' '.join(cmd)[:200]}  [{time.time() - t:.1f}s rc={r.returncode}]\n{r.stdout.strip()}", flush=True)
    if r.returncode:
        print(r.stderr[-3000:])
        raise SystemExit("harness failed")


def adt(a, rep):
    seqs = io.read_fastq_seqs(ADT, a.n_adt)
    cnt = Counter(s[:15] for s in seqs[:200000])
    tags = [t for t, _ in cnt.most_common(17)]
    t1 = tags[0]
    var = Counter()
    for s in seqs[:400000]:
        w = s[:15]
        if w != t1 and sum(x != y for x, y in zip(w, t1)) == 1:
            var[w] += 1
    v = var.most_common(1)[0][0]
    pe = next(i for i in range(15) if v[i] != t1[i])
    pa, pb = [i for i in range(15) if i != pe][:2]
    swap = lambda x: "ACGT"[("ACGT".index(x) + 1) % 4]  # noqa: E731
    A = v[:pa] + swap(v[pa]) + v[pa + 1:]
    B = v[:pb] + swap(v[pb]) + v[pb + 1:]
    tags += [A, B]
    rep["tags"] = {"from_data": 17, "synthetic": 2, "variant_shared_by_A_B_and_tag0": v, "variant_reads_in_first_400k": var[v]}
    tf = f"{a.work}/tags.txt"
    open(tf, "w").write("\n".join(tags) + "\n")
    rd = f"{a.work}/adt.bin"
    io.write_reads_from_strings(rd, seqs)
    out = f"{a.work}/adt"
    sh(a.exe.split() + ["mode=adt", f"reads={rd}", f"tags={tf}", f"out={out}", "max_offset=5"])
    k = io.load_arr(out, "tag", "i32")
    m = ref.AdtMatcher(tags)
    r = np.array([m.match(s, 5) for s in seqs], dtype=np.int32)
    vals, cnts = np.unique(r, return_counts=True)
    rep["adt"] = {"reads": len(seqs), "identical": int((k == r).sum()), "matched": int((r >= 0).sum()), "ambiguous_-2": int((r == -2).sum()),
                  "unmatched": int((r == -1).sum()), "ref_counts": {int(x): int(y) for x, y in zip(vals, cnts)}}
    # the offset window: prepend 0..7 random bases to 200000 reads
    rng = random.Random(1)
    sh_seqs = [("".join(rng.choice("ACGT") for _ in range(rng.randrange(8))) + s) for s in seqs[:200000]]
    rd2 = f"{a.work}/adt_shift.bin"
    io.write_reads_from_strings(rd2, sh_seqs)
    sh(a.exe.split() + ["mode=adt", f"reads={rd2}", f"tags={tf}", f"out={out}s", "max_offset=5"])
    k2 = io.load_arr(out + "s", "tag", "i32")
    r2 = np.array([m.match(s, 5) for s in sh_seqs], dtype=np.int32)
    rep["adt_shifted"] = {"reads": len(sh_seqs), "identical": int((k2 == r2).sum()), "matched": int((r2 >= 0).sum())}


def guide(a, rep):
    seqs = io.read_fastq_seqs(R2, a.n_guide)
    rng = random.Random(4)
    src = rng.sample(range(len(seqs)), 40)
    guides = [seqs[i][30:50] for i in src]
    guides.append(guides[3])                 # a sequence listed twice: every read of guide 3 is ambiguous
    guides.append(guides[5][2:17])           # nested in guide 5
    guides.append(seqs[src[7]][10:70])       # a 60-mer
    gf = f"{a.work}/guides.txt"
    open(gf, "w").write("\n".join(guides) + "\n")
    # a BAM record holds the reverse complement for reverse-strand alignments; the caller restores sequencing orientation
    strand = [rng.random() < 0.5 for _ in seqs]
    bam = [ref.revcomp(s) if r else s for s, r in zip(seqs, strand)]
    oriented = [ref.revcomp(b) if r else b for b, r in zip(bam, strand)]
    rd = f"{a.work}/guide.bin"
    io.write_reads_from_strings(rd, oriented)
    out = f"{a.work}/guide"
    sh(a.exe.split() + ["mode=guide", f"reads={rd}", f"guides={gf}", f"out={out}"])
    k = io.load_arr(out, "gid", "i32")
    r = np.array([ref.match_guide_bam(b, s, guides)[0] for b, s in zip(bam, strand)], dtype=np.int32)
    vals, cnts = np.unique(r, return_counts=True)
    rep["guide"] = {"reads": len(seqs), "guides": len(guides), "identical": int((k == r).sum()),
                    "ref_counts_by_id": {int(x): int(y) for x, y in zip(vals, cnts)}}


def spot(a, rep):
    lines = [x.split("\t")[0] for x in open(VISIUM_COORD).read().splitlines()[1:] if x.strip()]
    sp = np.unique(io.pack2_matrix(np.frombuffer("".join(lines).encode(), dtype=np.uint8).reshape(len(lines), 16))[0])
    wl = io.read_lines(WL10X)
    rng = random.Random(9)
    cells = rng.sample(lines, 3000) + rng.sample(wl, 500)
    cv = io.pack2_matrix(np.frombuffer("".join(cells).encode(), dtype=np.uint8).reshape(len(cells), 16))[0]
    io.write_u64_list(f"{a.work}/spots.bin", sp)
    io.write_u64_list(f"{a.work}/cells.bin", cv)
    out = f"{a.work}/spot"
    sh(a.exe.split() + ["mode=spot", f"cells={a.work}/cells.bin", f"spots={a.work}/spots.bin", f"out={out}"])
    k = io.load_arr(out, "row", "i32")
    d = {int(v): i for i, v in enumerate(sp)}
    r = np.array([d.get(int(v), -1) for v in cv], dtype=np.int32)
    rep["spot"] = {"cells": len(cells), "spots": len(sp), "identical": int((k == r).sum()), "found": int((r >= 0).sum())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-adt", type=int, default=1000000)
    ap.add_argument("--n-guide", type=int, default=1000000)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    rep = {}
    for f in (adt, guide, spot):
        try:
            f(a, rep)
        except SystemExit:
            raise
        except Exception as e:  # keep the other kernels' results
            rep[f.__name__ + "_error"] = repr(e)
    json.dump(rep, open(f"{a.out}/features.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
