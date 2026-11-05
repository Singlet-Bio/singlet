#!/bin/bash
#SBATCH --job-name=gb1-scan1fq
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Read-only scan of our own project space: every .1fq header (qual_mode, stream lengths, reads) and every raw read file.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$W/envs/pbench/bin:$PATH
OUT=$W/jobs/gpu-batch1/results/scan1fq-$SLURM_JOB_ID; mkdir -p $OUT
find $W -xdev \( -path $W/singlet_pipeline -o -path $W/jobs/gpu-batch1/results -o -path $W/jobs/gpu-batch1/ref -o -path $W/repos -o -path $W/envs -o -path $W/venvs -o -path $W/images \) -prune -o \
     -type f -name '*.1fq' -print 2>/dev/null > $OUT/files_1fq.txt
find $W -xdev \( -path $W/singlet_pipeline -o -path $W/jobs/gpu-batch1/results -o -path $W/jobs/gpu-batch1/ref -o -path $W/repos -o -path $W/envs -o -path $W/venvs -o -path $W/images \) -prune -o \
     -type f \( -name '*.fastq' -o -name '*.fastq.gz' -o -name '*.fq' -o -name '*.fq.gz' -o -name '*.sra' -o -name '*.fastq.zst' \) -size +50M -printf '%s\t%p\n' 2>/dev/null > $OUT/raw_reads.tsv
echo "1fq files: $(wc -l < $OUT/files_1fq.txt); raw read files: $(wc -l < $OUT/raw_reads.tsv)"
python3 - $OUT <<'PY'
import struct, sys, os
out = sys.argv[1]
rows = []
for p in open(out + "/files_1fq.txt").read().split("\n"):
    if not p:
        continue
    try:
        sz = os.path.getsize(p)
        b = open(p, "rb").read(96)
    except OSError as e:
        rows.append((p, "ERR", str(e))); continue
    if len(b) < 96 or b[:3] != b"1FQ":
        rows.append((p, sz, "not a 1fq header")); continue
    ns, proto = b[6], b[7]
    sl = struct.unpack_from("<4H", b, 12)
    nu, no = struct.unpack_from("<QQ", b, 20)
    qm = b[55]
    rows.append((p, sz, ns, proto, sl, nu, no, qm, b[56]))
with open(out + "/headers.tsv", "w") as f:
    f.write("path\tbytes\tn_streams\tprotocol\tstream_lengths\tn_unique\tn_original\tqual_mode\ttier\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print("headers written:", len(rows))
PY
echo "output $OUT"
