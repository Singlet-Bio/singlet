#!/bin/bash
#SBATCH --job-name=gb1-real5
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# REAL data, QUAL_MODE 1 (binned-legacy), protocol forced to 10x-5p-v2 and --r2-maxlen 150 (the library is 5 prime; the auto label 3p-v2 makes the strand check restart the run on the sorted path) (gpu batch 1, item 3): 10x Chromium 5' v1 VDJ T-cell library vdj_v1_hs_pbmc2_t (data/vdjc/pbmc2_t), lanes L006 and L007, R2 = 150-base cDNA read
# with full per-base qualities (8 distinct values, qualities below 10 on bases that are not N). Already on Clipper: nothing is downloaded.
# R1 is cut to 26 bases (16 barcode + 10 UMI, the 5' v1 layout). At most 100 M read pairs. Encoded with --quality binned-legacy, no dedup, no trim.
# Output: $W/jobs/gpu-batch1/data/pbmc2t_vdj5/pbmc2t_vdj5.1fq
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
D=$W/data/vdjc/pbmc2_t/fastqs/vdj_v1_hs_pbmc2_t_fastqs; WL=$W/pipeline-benchmark/whitelists; ENV=$W/envs/pbench
BIN=$W/jobs/s15/bin/0ea48783/singlet-cpu; MAXPAIRS=${MAXPAIRS:-100000000}
OUTD=$W/jobs/gpu-batch1/data/pbmc2t_vdj5; RES=$W/jobs/gpu-batch1/results/real5-$SLURM_JOB_ID; mkdir -p $OUTD $RES
export SINGLET_WHITELIST_DIR=$WL PATH=$ENV/bin:$PATH
SCR=/tmp/$USER-real5-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf "${SCR:?}"' EXIT
echo "REAL job=$SLURM_JOB_ID node=$(hostname -s) src=$D lanes=L006,L007 maxpairs=$MAXPAIRS bin=$BIN $(date -Is)" | tee $RES/summary.txt
T0=$(date +%s)
# R1 cut to 26 bases (lines 2 and 4 of each record), R2 unchanged; the two lanes in order; head stops at MAXPAIRS records
( zcat $D/vdj_v1_hs_pbmc2_t_S1_L006_R1_001.fastq.gz $D/vdj_v1_hs_pbmc2_t_S1_L007_R1_001.fastq.gz | head -n $((4*MAXPAIRS)) | awk 'NR%4==2||NR%4==0{print substr($0,1,26);next}{print}' > $SCR/R1.fastq ) &
( zcat $D/vdj_v1_hs_pbmc2_t_S1_L006_R2_001.fastq.gz $D/vdj_v1_hs_pbmc2_t_S1_L007_R2_001.fastq.gz | head -n $((4*MAXPAIRS)) > $SCR/R2.fastq ) &
wait
echo "fastq prep wall $(( $(date +%s) - T0 )) s"; ls -l $SCR | tee -a $RES/summary.txt
echo "R1 records $(( $(wc -l < $SCR/R1.fastq) / 4 )), R2 records $(( $(wc -l < $SCR/R2.fastq) / 4 ))" | tee -a $RES/summary.txt
# input statistics (whole R2): reads, longest, non-N bases below Q10, reads with such a base at 0-based position 128 or later
python3 - $SCR/R2.fastq $RES/input_stats.json <<'PY'
import sys, json, collections
n = 0; mx = 0; lowN = 0; rd = 0; rd128 = 0; qs = collections.Counter()
seq = ""
with open(sys.argv[1]) as f:
    for i, l in enumerate(f):
        k = i % 4
        if k == 1: seq = l.rstrip("\n")
        elif k == 3:
            q = l.rstrip("\n"); n += 1; mx = max(mx, len(q)); hit = hit128 = False
            for j, (s, c) in enumerate(zip(seq, q)):
                if s != "N" and ord(c) - 33 < 10:
                    lowN += 1; hit = True
                    if j >= 128: hit128 = True
            rd += hit; rd128 += hit128
            if n % 2000000 == 0: print("stats reads", n, flush=True)
st = dict(reads=n, longest=mx, lowq_nonN_bases=lowN, reads_with_lowq_nonN=rd, reads_with_lowq_nonN_at_or_after_pos128=rd128)
json.dump(st, open(sys.argv[2], "w"), indent=1); print(json.dumps(st))
PY
T2=$(date +%s)
rm -f $OUTD/pbmc2t_vdj5.1fq
$BIN encode --reads $SCR/R1.fastq $SCR/R2.fastq -o $OUTD/pbmc2t_vdj5.1fq --quality binned-legacy --protocol 10x-5p-v2 --r2-maxlen 150 --no-dedup --no-trim --whitelist-dir $WL --verbose 2>&1 | cut -c1-300 | tail -40
echo "encode wall $(( $(date +%s) - T2 )) s"
ls -l $OUTD
python3 - $OUTD/pbmc2t_vdj5.1fq <<'PY'
import struct, sys
b = open(sys.argv[1], "rb").read(96)
sl = struct.unpack_from("<4H", b, 12); nu, no = struct.unpack_from("<QQ", b, 20)
print("HEADER magic=%r n_streams=%d protocol=%d stream_lengths=%s n_unique=%d n_original=%d qual_mode=%d tier=%d" % (b[:3], b[6], b[7], sl, nu, no, b[55], b[56]))
PY
cp $0 $RES/make_real.sbatch 2>/dev/null
echo "REAL input ready: $OUTD/pbmc2t_vdj5.1fq $(date -Is)" | tee -a $RES/summary.txt
