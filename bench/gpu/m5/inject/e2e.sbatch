#!/bin/bash
#SBATCH --job-name=w11-e2e
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=03:59:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# W11 M5 end to end on one whole real sample (docs/gpu-engine/exploration.md §12.13). Everything with reads stays in job /tmp.
#  1. every distinct cDNA sequence of the sample (3' samples: the campaign clips nothing, so STAR maps the raw sequence)
#  2. m5 maps all of them on the GPU (lanes A, B; host lane for the rest) and writes (sequence, result) records
#  3. the campaign engine (singlet-m5inject: prod/campaign + the cache preload) runs the whole sample three times:
#       ref    no preload (as the campaign: sequence cache on, filled by STAR itself)
#       verify preload + SINGLET_STAR_SEQCACHE_VERIFY=1: STAR maps every read anyway and counts cached results that differ
#       gpu    preload: reads whose sequence is cached take the GPU's result instead of STAR's mapping
#  4. bench/star_accel/out_equiv.py (W3 harness) compares every output file of ref with verify and gpu
# Results -> $W/jobs/w11/results/e2e_<SET>*
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
SET=${SET:-GSM4006843}
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
[ "$SET" = GSM5411066 ] && G=$REF/GRCm39-2024-A
# input: FQ (a .1fq path) or the perf/benchmark copy of SET
FQ=${FQ:-}
if [ -z "$FQ" ]; then for c in $W/data/atlas-rerun/perf/$SET/sample.1fq $PB/data/$SET/$SET.1fq; do [ -e $c ] && FQ=$c && break; done; fi
[ -e "$FQ" ] || { echo "no input for $SET"; exit 2; }
D=$(dirname $FQ)
[ "${GENOME:-}" = GRCm39 ] && G=$REF/GRCm39-2024-A
BIN=$W/jobs/w11/bin/singlet-m5inject; SD=$W/jobs/w11/bin/stardump; M5=$W/jobs/w11/bin/${M5BIN:-m5w}
IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; OUT=$W/jobs/w11/results; mkdir -p $OUT
BARE=$W/repos/singlet-py.git; PY=$W/envs/pbench/bin/python
SCR=/tmp/$USER-e2e-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
echo "set=$SET node=$(hostname) job=$SLURM_JOB_ID $(date -Is) engine $(cat $BIN.commit) m5 $(sha256sum $M5 | cut -c1-16)"
T0=$(date +%s)
mkdir -p $SCR/idx && cp $G/star_2.7.11b/{Genome,SA,SAindex} $G/star_2.7.11b/*.txt $G/star_2.7.11b/*.tab $SCR/idx/ 2>/dev/null
export LC_ALL=C
$PB/bin/singlet-w3 decode $FQ --r2 /dev/stdout --threads 8 2>$SCR/decode.err | awk 'NR%4==2' | sort -S 80G --parallel=24 -T $SCR | uniq -c > $SCR/distinct.txt
ND=$(wc -l < $SCR/distinct.txt); NR=$(awk '{s+=$1} END {print s}' $SCR/distinct.txt)
echo "reads=$NR distinct=$ND"; echo "[t] distinct $(( $(date +%s) - T0 )) s"
# parameters and genome arrays as STAR sees them (stardump on 1000 sequences of the sample; the campaign's arguments)
head -1000 $SCR/distinct.txt | awk '{printf "@p%d_%d\n%s\n+\n", NR, $1, $2; q=$2; gsub(/./,"I",q); print q}' > $SCR/p.fq
mkdir -p $SCR/p $SCR/pstar
SINGLET_STAR_SEQCACHE_MB=0 STARDUMP_M5=$SCR/p/t $SD --runMode alignReads --genomeDir $SCR/idx --readFilesIn $SCR/p.fq --outSAMtype None \
  --outFilterScoreMin 10 --runThreadN 4 --genomeLoad NoSharedMemory --outFileNamePrefix $SCR/pstar/ > $SCR/pstar.out 2>&1
echo "params stardump exit=$?"
# 2. GPU
apptainer exec --nv -B $W -B $SCR $IMG env OMP_NUM_THREADS=24 $M5 --genome-dir $SCR/idx --truth $SCR/p/t --input $SCR/distinct.txt \
  --emit $SCR/blobs.bin --reps 1 --json $OUT/e2e_${SET}_m5.json ${M5ARGS:-} 2>&1 | grep -vE "^DIFF|^  seq|^  star|^  ours" | tail -22
echo "m5 exit=${PIPESTATUS[0]}"; ls -la $SCR/blobs.bin; echo "[t] gpu $(( $(date +%s) - T0 )) s"
rm -f $SCR/distinct.txt
# 3. engine runs
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
run_engine() {   # name, then env assignments
    local name=$1; shift
    mkdir -p $SCR/$name/out; ( cd $SCR/$name && env "$@" /usr/bin/time -v -o $SCR/$name/time.txt $BIN $FQ --ref-base $REF \
      --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/$name/out/ --threads 24 --pipeline --output-format 1pz \
      --bam-sort-mode disk --genome-shared off "${SCREEN[@]}" > $SCR/$name/run.log 2>&1 ); echo "engine $name exit=$?"
    grep -h "SEQCACHE" -r $SCR/$name/out --include="*Log.out" 2>/dev/null | head -5
    grep -h "SEQCACHE" $SCR/$name/run.log | head -5
    grep -E "Elapsed|Maximum resident" $SCR/$name/time.txt
    cp $SCR/$name/run.log $OUT/e2e_${SET}_$name.run.log
    echo "[t] $name $(( $(date +%s) - T0 )) s"
}
RUNS=${RUNS:-"ref verify gpu"}
for r in $RUNS; do
  case $r in
    ref|ref2) run_engine $r SINGLET_STAR_SEQCACHE_VERIFY=0 ;;
    verify)   run_engine $r SINGLET_STAR_SEQCACHE_PRELOAD=$SCR/blobs.bin SINGLET_STAR_SEQCACHE_MB=32768 SINGLET_STAR_SEQCACHE_VERIFY=1 ;;
    gpu)      run_engine $r SINGLET_STAR_SEQCACHE_PRELOAD=$SCR/blobs.bin SINGLET_STAR_SEQCACHE_MB=32768 SINGLET_STAR_SEQCACHE_VERIFY=0 ;;
  esac
done
# 4. compare every output file
git -C $BARE show refs/w11/campaign:bench/star_accel/out_equiv.py > $SCR/out_equiv.py
for v in $RUNS; do
  [ $v = ref ] && continue
  echo "=== out_equiv ref vs $v"
  $PY $SCR/out_equiv.py $SCR/ref/out $SCR/$v/out 2>&1 | tee $OUT/e2e_${SET}_equiv_$v.txt | tail -25
  # files that differ: order-independent comparison (FASTQ records sorted; other files: sorted lines), and their sizes
  for f in $(cd $SCR/ref/out && find . -type f | sort); do
    cmp -s $SCR/ref/out/$f $SCR/$v/out/$f && continue
    case $f in
      *.fq|*.fastq|*mate1|*mate2) ka=$(paste - - - - < $SCR/ref/out/$f | sort -S 20G | sha256sum | cut -c1-16)
                                  kb=$(paste - - - - < $SCR/$v/out/$f | sort -S 20G | sha256sum | cut -c1-16) ;;
      *) ka=$(sort -S 4G $SCR/ref/out/$f 2>/dev/null | sha256sum | cut -c1-16); kb=$(sort -S 4G $SCR/$v/out/$f 2>/dev/null | sha256sum | cut -c1-16) ;;
    esac
    echo "  differs: $f sizes $(stat -c %s $SCR/ref/out/$f) $(stat -c %s $SCR/$v/out/$f) order-independent: $([ $ka = $kb ] && echo SAME || echo DIFFERENT)"
  done | tee -a $OUT/e2e_${SET}_equiv_$v.txt
  diff <(cd $SCR/ref/out && find . -type f | sort) <(cd $SCR/$v/out && find . -type f | sort) | sed 's/^/  only: /'
done
for r in $RUNS; do ( cd $SCR/$r/out && find . -type f -printf '%s\t%p\n' | sort -k2 ) > $OUT/e2e_${SET}_files_$r.tsv; done
echo "[t] total $(( $(date +%s) - T0 )) s"
