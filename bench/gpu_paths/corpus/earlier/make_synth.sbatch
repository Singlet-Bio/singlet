#!/bin/bash
#SBATCH --job-name=gb1-synth
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# SYNTHETIC TEST INPUT (gpu batch 1, item 3). Not real data.
# The first 20 M read pairs of GSM7064234 (SRR23617320, local .sra, 151-base cDNA reads, Illumina 4-bin qualities 2/11/25/37 with Q2 only on N).
# A deterministic pseudorandom share of the non-N cDNA bases whose quality is Q11 is lowered to Q5 ('&'), so low quality is not tied to N
# and the GPU pileup needs the per-base mask. A fixed rule adds a low base at 0-based position 128 (reads with index mod 1000 = 0) and at
# position 150 (index mod 1000 = 1), non-N bases only, so the third mask word (bases 129 to 192) carries bits in a known set of reads.
# The barcode read is cut to 28 bases. Seed and share are fixed below. Encoded with --quality full, no dedup, no trim.
# Output: $W/jobs/gpu-batch1/data/gsm7064234syn/gsm7064234syn.1fq ; script, stats and log: $W/jobs/gpu-batch1/results/synth-<job>/
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
SRA=$W/pilot/GSM7064234/sra/SRR23617320.sra; WL=$W/pipeline-benchmark/whitelists; ENV=$W/envs/pbench
BIN=$W/jobs/s15/bin/0ea48783/singlet-cpu
NPAIRS=${NPAIRS:-20000000}; SEED=${SEED:-20261105}; SHARE=${SHARE:-0.25}
OUTD=$W/jobs/gpu-batch1/data/gsm7064234syn; RES=$W/jobs/gpu-batch1/results/synth-$SLURM_JOB_ID; mkdir -p $OUTD $RES
export SINGLET_WHITELIST_DIR=$WL PATH=$ENV/bin:$PATH
SCR=/tmp/$USER-synth-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf "${SCR:?}"' EXIT
echo "SYNTHETIC job=$SLURM_JOB_ID node=$(hostname -s) npairs=$NPAIRS seed=$SEED share=$SHARE bin=$BIN $(date -Is)" | tee $RES/summary.txt
T0=$(date +%s)
fastq-dump -X $NPAIRS --split-files -O $SCR $SRA 2>&1 | tail -2
echo "fastq-dump wall $(( $(date +%s) - T0 )) s"; ls -l $SCR
cat > $RES/synth_quality.py <<'PY'
# SYNTHETIC: lowers a seeded share of non-N Q11 cDNA bases to Q5.
# Usage: synth_quality.py FILE2 FILE3 OUT_R1 OUT_R2 SEED SHARE WHITELIST
import sys, json
import numpy as np
f2, f3, o1, o2, seed, share, wlp = sys.argv[1:8]
seed = int(seed); share = float(share)
wl = set(l.strip() for l in open(wlp))
def frac(p):
    ok = n = 0
    with open(p) as f:
        for i, l in enumerate(f):
            if i % 4 == 1:
                n += 1; ok += l[:16] in wl
                if n >= 5000: break
    return ok / max(n, 1)
fa, fb = frac(f2), frac(f3)
print("barcode read test (first 16 bases on whitelist): file2 %.3f file3 %.3f" % (fa, fb), flush=True)
bc, cd = (f2, f3) if fa > fb else (f3, f2)
print("barcode+UMI read:", bc, "| cDNA read:", cd, flush=True)
rng = np.random.default_rng(seed)
CH = 500000; L = 151
st = dict(reads=0, cdna_len=L, bases=0, bases_q11_nonN=0, lowered_random=0, lowered_forced_extra=0, reads_with_lowered=0,
          reads_with_lowered_ge128=0, lowered_at_pos128=0, lowered_at_pos150=0, reads_lowq_nonN=0, reads_lowq_nonN_ge128=0, qvalues_after={})
def chunks(fp):
    with open(fp, "rb") as f:
        while True:
            lines = [x for x in (f.readline() for _ in range(4 * CH)) if x]
            if not lines: return
            yield lines
with open(o1, "wb") as w1, open(o2, "wb") as w2:
    for lb, lc in zip(chunks(bc), chunks(cd)):
        n = len(lb) // 4
        out = []
        for i in range(n):
            out += [lb[4*i], lb[4*i+1].rstrip(b"\n")[:28] + b"\n", b"+\n", lb[4*i+3].rstrip(b"\n")[:28] + b"\n"]
        w1.write(b"".join(out))
        seq = np.frombuffer(b"".join(lc[4*i+1].rstrip(b"\n") for i in range(n)), dtype=np.uint8).reshape(n, L)
        q = np.frombuffer(b"".join(lc[4*i+3].rstrip(b"\n") for i in range(n)), dtype=np.uint8).reshape(n, L).copy()
        isN = seq == ord("N")
        q11 = (q == ord(",")) & ~isN
        pick = q11 & (rng.random((n, L)) < share)
        forced = np.zeros_like(pick)
        idx = st["reads"] + np.arange(n)
        r128 = (idx % 1000 == 0); r150 = (idx % 1000 == 1)
        forced[r128, 128] = ~isN[r128, 128]
        forced[r150, 150] = ~isN[r150, 150]
        before = q.copy()
        q[pick | forced] = ord("&")
        ch = (q != before)
        low = ((q.astype(np.int16) - 33) < 10) & ~isN
        st["reads"] += n; st["bases"] += n * L; st["bases_q11_nonN"] += int(q11.sum())
        st["lowered_random"] += int(pick.sum()); st["lowered_forced_extra"] += int((forced & ~pick).sum())
        st["reads_with_lowered"] += int(ch.any(1).sum()); st["reads_with_lowered_ge128"] += int(ch[:, 128:].any(1).sum())
        st["lowered_at_pos128"] += int(ch[:, 128].sum()); st["lowered_at_pos150"] += int(ch[:, 150].sum())
        st["reads_lowq_nonN"] += int(low.any(1).sum()); st["reads_lowq_nonN_ge128"] += int(low[:, 128:].any(1).sum())
        for v, c in zip(*np.unique(q, return_counts=True)):
            st["qvalues_after"][chr(v)] = st["qvalues_after"].get(chr(v), 0) + int(c)
        out = []
        for i in range(n):
            out += [lc[4*i], lc[4*i+1], b"+\n", q[i].tobytes() + b"\n"]
        w2.write(b"".join(out))
        print("reads", st["reads"], flush=True)
st.update(seed=seed, share=share, barcode_file=bc, cdna_file=cd)
json.dump(st, open(o2 + ".stats.json", "w"), indent=1)
print(json.dumps(st, indent=1))
PY
T1=$(date +%s)
python3 $RES/synth_quality.py $SCR/SRR23617320_2.fastq $SCR/SRR23617320_3.fastq $SCR/syn_R1.fastq $SCR/syn_R2.fastq $SEED $SHARE $WL/3M-february-2018.txt 2>&1 | tee $RES/synth_quality.log | tail -45
echo "synth wall $(( $(date +%s) - T1 )) s"
cp $SCR/syn_R2.fastq.stats.json $RES/stats.json
rm -f $SCR/SRR23617320_*.fastq
T2=$(date +%s)
rm -f $OUTD/gsm7064234syn.1fq
$BIN encode --reads $SCR/syn_R1.fastq $SCR/syn_R2.fastq -o $OUTD/gsm7064234syn.1fq --quality full --no-dedup --no-trim --whitelist-dir $WL --verbose 2>&1 | cut -c1-300 | tail -40
echo "encode wall $(( $(date +%s) - T2 )) s"
ls -l $OUTD
python3 - $OUTD/gsm7064234syn.1fq <<'PY'
import struct, sys
b = open(sys.argv[1], "rb").read(96)
sl = struct.unpack_from("<4H", b, 12); nu, no = struct.unpack_from("<QQ", b, 20)
print("HEADER magic=%r n_streams=%d protocol=%d stream_lengths=%s n_unique=%d n_original=%d qual_mode=%d tier=%d" % (b[:3], b[6], b[7], sl, nu, no, b[55], b[56]))
PY
cp $0 $RES/make_synth.sbatch 2>/dev/null
echo "SYNTHETIC input ready: $OUTD/gsm7064234syn.1fq (seed $SEED share $SHARE) $(date -Is)" | tee -a $RES/summary.txt
