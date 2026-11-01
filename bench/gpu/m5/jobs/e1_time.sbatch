#!/bin/bash
#SBATCH --job-name=e1-time
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E1 profile (docs/gpu-engine/fable/E1-profile.md): timing runs on the standard 2 M pbmc1k_v3 dump set (exploration 12.12),
# one truth generation, then: base (m5 JSON), nsys timeline, NB sweep (A/M/B), --mono global vs --local, --blocks-per-sm,
# batch tails (m5t). STEPS selects (comma list). Read sequences stay in job /tmp; DIFF/sequence lines never reach $OUT.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
G=$REF/GRCh38-2024-A/star_2.7.11b; SD=$W/jobs/w11/bin/stardump; E=$W/jobs/fable/profile
IMG=$W/images/cuda-13-devel.sif; PY=$W/images/python-3.12-bookworm.sif; NSYS=/usr/local/bin/nsys
OUT=$E/results/time-$SLURM_JOB_ID; mkdir -p $OUT
SCR=/tmp/$USER-e1-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; export LC_ALL=C
MAXD=${MAXD:-2000000}; STEPS=,${STEPS:-base,nsys,tail,sweep,mono},
echo "job=$SLURM_JOB_ID node=$(hostname) $(date -Is) MAXD=$MAXD steps=$STEPS m5=$(cat $E/bin/m5.commit) m5t=$(cat $E/bin/m5t.commit)"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv,noheader
T0=$(date +%s)
# ---- truth, as bench/gpu/m5/jobs/m5_run.sbatch: distinct R2 sequences, every k-th (~MAXD), stardump's M5 truth
mkdir -p $SCR/idx && cp $G/Genome $G/SA $G/SAindex $G/*.txt $G/*.tab $SCR/idx/ 2>/dev/null
echo "[t] index copy $(( $(date +%s) - T0 )) s"
$PB/bin/singlet-w3 decode $PB/data/pbmc1k_v3/pbmc1k_v3.1fq --r2 /dev/stdout --threads 4 2>$SCR/decode.err | awk 'NR%4==2' \
  | sort -S 60G --parallel=16 -T $SCR | uniq -c > $SCR/distinct.txt
ND=$(wc -l < $SCR/distinct.txt); NR=$(awk '{s+=$1} END {print s}' $SCR/distinct.txt)
K=$(( (ND + MAXD - 1) / MAXD )); [ $K -lt 1 ] && K=1
awk -v k=$K '(NR-1)%k==0 {printf "@d%d_%d\n%s\n+\n", NR, $1, $2; q=$2; gsub(/./,"I",q); print q}' $SCR/distinct.txt > $SCR/distinct.fq
NS=$(( $(wc -l < $SCR/distinct.fq) / 4 )); rm -f $SCR/distinct.txt
echo "reads=$NR distinct=$ND every_k=$K selected=$NS"; echo "[t] distinct $(( $(date +%s) - T0 )) s"
mkdir -p $SCR/m5 $SCR/star
SINGLET_STAR_SEQCACHE_MB=0 STARDUMP_M5=$SCR/m5/t $SD --runMode alignReads --genomeDir $SCR/idx --readFilesIn $SCR/distinct.fq \
  --outSAMtype None --outFilterScoreMin 10 --runThreadN 24 --genomeLoad NoSharedMemory --outFileNamePrefix $SCR/star/ > $SCR/star.out 2>&1
echo "stardump exit=$?"; tail -2 $SCR/star.out; rm -f $SCR/distinct.fq; echo "[t] stardump $(( $(date +%s) - T0 )) s"
BUD="--budget-a 1024 --budget-m 2048 --budget-b 2048"
DEF="--batch-a 262144 --batch-m 262144 --batch-b 65536 $BUD"
m5run() {   # label binary args... ($PRE: profiler command in front of the binary, inside the container)
  local L=$1 B=$2; shift 2; local t=$(date +%s)
  echo "=== $L: $B $*"
  apptainer exec --nv -B $W -B $SCR $IMG env OMP_NUM_THREADS=24 ${PRE:-} $E/bin/$B --genome-dir $SCR/idx --truth $SCR/m5/t \
    --json $OUT/$L.json "$@" 2>&1 | grep -v "^  seq\|^  star\|^  ours\|^DIFF" > $OUT/$L.log
  echo "exit=${PIPESTATUS[0]} wall=$(( $(date +%s) - t )) s"
  grep -E "^M5:|GPU lane A|thread cycles|inside stitching|B/read|work [0-9]+ B/thread|overflow why|heavy lane|TAILSUM|out of memory|error" $OUT/$L.log | head -30
}
has() { [[ $STEPS == *,$1,* ]]; }
has base && m5run base m5 --reps 3 $DEF
if has nsys; then
  PRE="$NSYS profile -t cuda --cuda-memory-usage=false --stats=false -o $SCR/nsys_base -f true" m5run nsys m5 --reps 3 $DEF
  apptainer exec -B $W -B $SCR $IMG $NSYS stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_trace -f csv \
    -o $OUT/nsys_base $SCR/nsys_base.nsys-rep > $OUT/nsys_stats.log 2>&1; echo "nsys stats exit=$?"
  cp $SCR/nsys_base.nsys-rep $OUT/ 2>/dev/null; ls -la $OUT | grep nsys
fi
if has sweep; then   # reads in flight per lane (Q4): A and M together, B alongside (lane membership does not depend on batch size)
  for v in "32768 32768 8192" "65536 65536 16384" "131072 131072 32768" "524288 524288 131072" "1048576 524288 131072"; do
    set -- $v; m5run nb_a$1_m$2_b$3 m5 --reps 3 --batch-a $1 --batch-m $2 --batch-b $3 $BUD
  done
fi
if has mono; then    # one kernel per lane A: Work in global memory (array of structs) vs local memory (interleaved per thread)
  m5run mono_g_b1024 m5 --reps 3 --mono --mono-budget 1024 $DEF
  m5run mono_l_b1024 m5 --reps 3 --mono --local --mono-budget 1024 $DEF
  m5run mono_g_b0 m5 --reps 3 --mono $DEF
  m5run mono_l_b0 m5 --reps 3 --mono --local $DEF
  for b in 1 2 8 16; do m5run mono_g_b1024_bps$b m5 --reps 3 --mono --mono-budget 1024 --blocks-per-sm $b $DEF; done
  for b in 2 8; do m5run mono_l_b1024_bps$b m5 --reps 3 --mono --local --mono-budget 1024 --blocks-per-sm $b $DEF; done
fi
has tail && m5run tail m5t --reps 3 $DEF   # last: m5t may be rebuilt while the job runs
# summary (stdlib python; scripts from the branch tip)
BARE=$W/repos/singlet-py.git; mkdir -p $SCR/py
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/profile:refs/fable/profile   && git -C $BARE archive refs/fable/profile bench/fable/profile | tar -x -C $SCR/py
[ -f $SCR/py/bench/fable/profile/e1_summarize.py ] && apptainer exec -B $W -B $SCR $PY python3 $SCR/py/bench/fable/profile/e1_summarize.py $OUT > $OUT/summary.txt 2>&1
tail -80 $OUT/summary.txt 2>/dev/null
echo "[t] total $(( $(date +%s) - T0 )) s"
