#!/bin/bash
#SBATCH --job-name=s8-mem
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: device memory of one sample as the prefill chunk size (SINGLET_ENGINE_GPU_CHUNK) and the pileup arena cap (SINGLET_GPU_PILEUP_ARENA_GB) change:
# peak device memory during the prefill and during the pileup (nvidia-smi every 0.5 s), the prefill's GPU seconds, the pileup run seconds and partitions.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=${BIN:?set BIN}; s=${SAMPLE:-pbmc1k_v3}; G=$REF/GRCh38-2024-A; FQ=$PB/data/$s/$s.1fq
RES=$W/jobs/s8/results/mem-$SLURM_JOB_ID; mkdir -p $RES; SCR=/tmp/$USER-s8mem-$SLURM_JOB_ID; mkdir -p $SCR; trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=7200 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_GPU_PILEUP=on
echo "node=$(hostname) bin=$(cat $(dirname $BIN)/commit) $(date -Is)" | tee $RES/summary.txt
CFGS=${CFGS:-"16777216:0 4194304:0 1048576:0 4194304:30 4194304:24"}
for cfg in $CFGS; do
  IFS=: read ch ar nb nbb <<< "$cfg"; nb=${nb:-262144}; nbb=${nbb:-65536}; D=$RES/c${ch}_a${ar}_nb${nb}_nbb${nbb}; mkdir -p $D $SCR/o
  ( while true; do echo "$(date +%s.%N) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"; sleep 0.5; done ) > $D/mem.csv 2>/dev/null & MM=$!
  env SINGLET_ENGINE_GPU_CHUNK=$ch SINGLET_ENGINE_GPU_NB=$nb SINGLET_ENGINE_GPU_NBB=$nbb $( [ $ar != 0 ] && echo SINGLET_GPU_PILEUP_ARENA_GB=$ar ) $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
     --out-prefix $SCR/o/ --threads 18 --pipeline --output-format 1pz --engine gpu 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $D/run.log
  kill $MM 2>/dev/null
  grep -a -h "device memory (GB in use)\|upload .* s, dedup\|cache filled" $SCR/o/star_Log.out 2>/dev/null | cut -c1-300 | tee -a $RES/summary.txt
  python3 - "$D" "$ch" "$ar" "$nb" "$nbb" <<'PY' | tee -a $RES/summary.txt
import sys, re
D, ch, ar, nb = sys.argv[1:5]
rows = [tuple(map(float, l.split())) for l in open(D + "/mem.csv") if len(l.split()) == 2]
log = open(D + "/run.log", errors="replace").read()
ts = lambda pat: next((int(m.group(1)) for m in re.finditer(r"^(\d+) .*" + pat, log, re.M)), None)
t_pre = ts(r"STAR pid="); t_pil = ts(r"\[gpu_pileup\] hand-off pass: entries"); t_on = ts(r"\[gpu_pileup\] on:")
def peak(a, b): v = [m for t, m in rows if (a is None or t >= a) and (b is None or t <= b)]; return max(v) / 1024 if v else 0
print(f"chunk {int(ch)>>20}M, lane-A/B reads in flight {nb}/{sys.argv[5] if len(sys.argv) > 5 else ""}, pileup arena cap {ar} GB: peak device memory prefill phase {peak(t_pre, t_pil):.1f} GB, pileup phase {peak(t_pil, t_on):.1f} GB; "
      + "; ".join(m.group(0).strip() for m in re.finditer(r"cell partitions: \d+|peak [\d.]+ GB|GPU [\d.]+ ms, total [\d.]+ ms", log)) )
print("   " + "; ".join(re.findall(r"(?:upload|dedup|GPU mapping) [\d.]+ s", log)[:3]), "|", " ".join(re.findall(r"cache filled .*?device \d+\)", log))[:200])
PY
  rm -rf $SCR/o
done
echo done | tee -a $RES/summary.txt
