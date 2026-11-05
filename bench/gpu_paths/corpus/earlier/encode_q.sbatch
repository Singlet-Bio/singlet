#!/bin/bash
#SBATCH --job-name=gb1-encq
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# GPU batch 1, item 3: the first 100 M reads of GSM7064234 (SRR23617320, cDNA reads up to 151 bases) encoded from the local .sra
# with FULL per-base qualities (--quality full, qual_mode 2). The earlier slice (cap1fq_0ea48783) used --quality binned (Illumina 4-level:
# low quality tied to N, so the GPU pileup needs no mask). Same command as jobs/s15/pilot_rerun.sh rcap, only the quality mode differs.
# No download: the .sra is on Clipper already.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
P=$W/pilot/GSM7064234; BIN=$W/jobs/s15/bin/0ea48783/singlet-cpu; WL=$W/pipeline-benchmark/whitelists; ENV=$W/envs/pbench
OUTD=$W/jobs/gpu-batch1/data/gsm7064234q; mkdir -p $OUTD
export SINGLET_WHITELIST_DIR=$WL PATH=$ENV/bin:$PATH
echo "ENCODE job=$SLURM_JOB_ID node=$(hostname -s) bin=$BIN commit=$(cat $(dirname $BIN)/commit 2>/dev/null) sra=$P/sra/SRR23617320.sra out=$OUTD/SRR23617320.1fq $(date -Is)"
[ -f $P/sra/SRR23617320.sra ] || { echo "no .sra"; exit 2; }
rm -f $OUTD/SRR23617320.1fq $OUTD/SRR23617320.1fq.new
T0=$(date +%s)
/usr/bin/time -v $BIN download $P/sra/SRR23617320.sra --quality full --max-reads 100000000 --vdb-threads 2 --no-profile --whitelist-dir $WL -o $OUTD/SRR23617320.1fq 2>&1 | cut -c1-300 | tail -60
echo "encode wall $(( $(date +%s) - T0 )) s"
ls -l $OUTD
python3 - $OUTD/SRR23617320.1fq <<'PY'
import struct, sys
b = open(sys.argv[1], "rb").read(96)
sl = struct.unpack_from("<4H", b, 12); nu, no = struct.unpack_from("<QQ", b, 20)
print("HEADER magic=%r n_streams=%d protocol=%d stream_lengths=%s n_unique=%d n_original=%d qual_mode=%d tier=%d" % (b[:3], b[6], b[7], sl, nu, no, b[55], b[56]))
PY
echo done $(date -Is)
