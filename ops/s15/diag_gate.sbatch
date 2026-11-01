#!/bin/bash
#SBATCH --job-name=s15-gatediag
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S15: why does Singlet's decode gate reject GSM7064234's SRR23617320 as splitseq? Read lengths per mate and barcode-whitelist match rates of the first 200k spots.
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
T=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT; cd $T
ACC=${ACC:-SRR23617320}; SRA=$W/pilot/GSM7064234/sra/$ACC.sra; [ -s $SRA ] && SRC=$SRA || SRC=$ACC; echo "source $SRC"
fastq-dump -X 200000 --split-files -O $T $SRC 2>&1 | tail -2; ls -l $T
python3 - <<PY
import glob, collections
W = "$W/pipeline-benchmark/whitelists/"
wl = {n: set(open(W + n).read().split()) for n in ["gex_737K-arc-v1.txt", "737K-august-2016.txt", "3M-february-2018.txt", "737K-cratac-v1.txt"]}
for f in sorted(glob.glob("$T/*.fastq")):
    L = collections.Counter(); seqs = []
    for i, l in enumerate(open(f)):
        if i % 4 == 1:
            s = l.strip(); L[len(s)] += 1; seqs.append(s)
    print(f.split("/")[-1], "lengths", L.most_common(3), "example", seqs[0][:60])
    for off in (0, 1, 8, 10):
        for n, w in wl.items():
            k = len(next(iter(w)))
            if len(seqs[0]) < off + k: continue
            m = sum(1 for s in seqs[:100000] if s[off:off + k] in w) / min(100000, len(seqs))
            if m > 0.02: print("   offset %d vs %s: %.1f%% match" % (off, n, 100 * m))
    # reverse complement of first 16 for the ATAC-style read
    comp = str.maketrans("ACGT", "TGCA")
    for n, w in wl.items():
        k = len(next(iter(w)))
        if len(seqs[0]) >= k:
            m = sum(1 for s in seqs[:100000] if s[:k].translate(comp)[::-1] in w) / min(100000, len(seqs))
            if m > 0.02: print("   revcomp offset 0 vs %s: %.1f%% match" % (n, 100 * m))
PY
