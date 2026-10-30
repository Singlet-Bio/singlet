#!/bin/bash
#SBATCH --job-name=w11-m13-run
#SBATCH --account=debruinz
#SBATCH --nice=10000
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=300G
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# W11 M13, one sample end to end in ONE job (reads, dumps and replays live only in this job's /tmp):
#   1. CPU capture: dev engine (c4a1be38 + m13.patch) on the real .1fq, stops after the non-host screen (SINGLET_NH_STOP) ->
#      star_Unmapped.out.mate1/2, the engine's own nonhost/ outputs, the CPU per-read dump (dump0), the stage timers [m13t]
#   2. GPU: m13 loads the 3 indexes + Bloom into HBM, classifies the unmapped reads (timed reps), compares with dump0, writes gpu.res
#   3. replay A: `singlet nhscreen` on the same unmapped reads, CPU path (control)   4. replay B: same, GPU results injected
#   5. compare: dumps (raw.bin = hit maps in map iteration order, multi.bin = MultiHit incl. float bits) and every file of nonhost/
#   sbatch --partition=short-gpu|debug-gpu --time=HH:MM:SS --export=ALL,SET=<pbmc2m|GSM4006843|GSM5411066|GSM6470208>[,T=24][,REPS=3] m13_run.sbatch
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; M=$W/jobs/w11/m13; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A; SET=${SET:-pbmc2m}; T=${T:-24}; REPS=${REPS:-3}
NHD=$REF/nonhost
[ "$SET" = GSM5411066 ] && G=$REF/GRCm39-2024-A
if [ "$SET" = pbmc2m ]; then F=$PB/data/pbmc2m/pbmc2m.1fq; else D=$W/data/bam-stream/$SET; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$SET; F=$D/sample.1fq; fi
R=$M/results/$SET; rm -rf $R; mkdir -p $R
SCR=/tmp/$USER-m13-$SLURM_JOB_ID; mkdir -p $SCR/out $SCR/dump0 $SCR/dumpA $SCR/dumpB $SCR/repA $SCR/repB; trap 'kill $MON 2>/dev/null; rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
echo "node=$(hostname) job=$SLURM_JOB_ID set=$SET file=$F T=$T $(date -Is)"; sha256sum $M/bin/singlet-m13 | cut -c1-16; cat $M/bin/singlet-m13.commit
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
( pk=0; while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -gt $pk ] && { pk=$u; echo "gpu_mem_peak_mib=$pk" > $R/gpumem.txt; }; sleep 1; done ) &
MON=$!
# ---- 0. optional: re-stage a public sample into this job's /tmp (STAGE_SRR=SRR9900123 for GSM4006843; its temporary copy under $W/data was deleted)
if [ -n "${STAGE_SRR:-}" ]; then
  mkdir -p $SCR/in $SCR/sra; ST0=$(date +%s)
  ( export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
    prefetch --max-size u -O $SCR/sra $STAGE_SRR > $SCR/pf.log 2>&1 || { tail $SCR/pf.log; exit 1; }
    $PB/bin/singlet-campaign download $(ls $SCR/sra/$STAGE_SRR/*.sra* | head -1) --vdb-threads 4 --no-profile -o $SCR/in/$STAGE_SRR.1fq.tmp > $SCR/dl.log 2>&1 || { tail -20 $SCR/dl.log; exit 1; } )
  mv $SCR/in/$STAGE_SRR.1fq.tmp $SCR/in/$STAGE_SRR.1fq; F=$SCR/in/$STAGE_SRR.1fq; rm -rf $SCR/sra
  echo "staged $F ($(stat -c %s $F) B) in $(( $(date +%s) - ST0 )) s"
fi
# ---- 1. CPU capture
cd $SCR; S0=$(date +%s)
SINGLET_NH_DUMP=$SCR/dump0 SINGLET_NH_STOP=1 /usr/bin/time -v -o $R/engine.time.txt $M/bin/singlet-m13 $F --ref-base $REF --genome-dir $G/star_2.7.11b \
  --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads $T --pipeline --output-format 1pz --bam-sort-mode disk --genome-shared off > $R/engine.log 2>&1 || echo "engine exit $?"
echo "capture wall $(( $(date +%s) - S0 )) s"; grep -E "^\[m13t\]|\[nonhost\] (EM screening|EM converged)|\[m13\]" $R/engine.log | head -20
ls -la $SCR/out/star_Unmapped.out.mate1 $SCR/out/star_Unmapped.out.mate2 $SCR/dump0 || { echo "no unmapped reads"; tail -20 $R/engine.log; exit 1; }
# ---- 2. GPU
apptainer exec --nv -B $W -B /tmp $IMG $M/bin/m13 --reads $SCR/out/star_Unmapped.out.mate1 --index-dir $NHD --bloom $NHD/host_21mer.bloom \
  --label $SET --reps $REPS --out $SCR/gpu.res --cpu-dump $SCR/dump0/raw.bin --json $R/gpu.json > $R/gpu.log 2>&1; echo "gpu exit $?"; tail -30 $R/gpu.log
# ---- 3/4. replays
for X in A B; do
  ln -s $SCR/out/star_Unmapped.out.mate1 $SCR/rep$X/star_Unmapped.out.mate1; ln -s $SCR/out/star_Unmapped.out.mate2 $SCR/rep$X/star_Unmapped.out.mate2
done
SINGLET_NH_DUMP=$SCR/dumpA $M/bin/singlet-m13 nhscreen $SCR/repA $NHD/viral.snhskidx $NHD/bacterial.snhskidx $NHD/fungal.snhskidx $SCR/dump0 > $R/replayA.log 2>&1; echo "replay A exit $?"
SINGLET_NH_DUMP=$SCR/dumpB SINGLET_NH_GPU_RESULTS=$SCR/gpu.res $M/bin/singlet-m13 nhscreen $SCR/repB $NHD/viral.snhskidx $NHD/bacterial.snhskidx $NHD/fungal.snhskidx $SCR/dump0 > $R/replayB.log 2>&1; echo "replay B exit $?"
grep -h "^\[m13t\]" $R/replayA.log | sed 's/^/A /'; grep -h "^\[m13t\]" $R/replayB.log | sed 's/^/B /'
# ---- 5. compare
{
  echo "== per-read dumps (sha256, bytes)"
  for f in raw.bin multi.bin; do for d in dump0 dumpA dumpB; do echo "$f $d $(sha256sum $SCR/$d/$f | cut -c1-64) $(stat -c %s $SCR/$d/$f)"; done; done
  for f in raw.bin multi.bin; do
    cmp -s $SCR/dump0/$f $SCR/dumpA/$f && echo "IDENTICAL $f dump0(engine,CPU) == dumpA(replay,CPU)" || echo "DIFFERENT $f dump0 vs dumpA"
    cmp -s $SCR/dumpA/$f $SCR/dumpB/$f && echo "IDENTICAL $f dumpA(CPU) == dumpB(GPU)" || echo "DIFFERENT $f dumpA vs dumpB"
    cmp -s $SCR/dump0/$f $SCR/dumpB/$f && echo "IDENTICAL $f dump0(engine,CPU) == dumpB(GPU)" || echo "DIFFERENT $f dump0 vs dumpB"
  done
  echo "== nonhost outputs"
  ( cd $SCR/out && find nonhost -type f | sort ) | while read f; do
    a=$(sha256sum $SCR/out/$f | cut -c1-64); b=$(sha256sum $SCR/repA/$f 2>/dev/null | cut -c1-64); c=$(sha256sum $SCR/repB/$f 2>/dev/null | cut -c1-64)
    echo "FILE $f ($(stat -c %s $SCR/out/$f) B) engine==replayA:$([ "$a" = "$b" ] && echo yes || echo NO) replayA==replayB:$([ "$b" = "$c" ] && echo yes || echo NO) engine==replayB:$([ "$a" = "$c" ] && echo yes || echo NO)"
  done
  ( cd $SCR/repB && find nonhost -type f | sort ) | while read f; do [ -f $SCR/out/$f ] || echo "EXTRA file only in replayB: $f"; done
  if [ -f $SCR/out/nonhost_reads.fq ]; then
    a=$(sha256sum $SCR/out/nonhost_reads.fq | cut -c1-64); b=$(sha256sum $SCR/repA/nonhost_reads.fq | cut -c1-64); c=$(sha256sum $SCR/repB/nonhost_reads.fq | cut -c1-64)
    echo "FILE nonhost_reads.fq ($(stat -c %s $SCR/out/nonhost_reads.fq) B) engine==replayA:$([ "$a" = "$b" ] && echo yes || echo NO) replayA==replayB:$([ "$b" = "$c" ] && echo yes || echo NO) engine==replayB:$([ "$a" = "$c" ] && echo yes || echo NO)"
  else echo "FILE nonhost_reads.fq not written (no kingdom fraction > 1%)"; fi
} > $R/compare.txt 2>&1
cat $R/compare.txt
# ---- 6. clean timing replays at the campaign's 4 threads, no dumps: C = CPU path, D = GPU results injected (outputs checked again)
for X in C D; do
  mkdir -p $SCR/rep$X; ln -s $SCR/out/star_Unmapped.out.mate1 $SCR/rep$X/star_Unmapped.out.mate1; ln -s $SCR/out/star_Unmapped.out.mate2 $SCR/rep$X/star_Unmapped.out.mate2
done
NH_THREADS=4 $M/bin/singlet-m13 nhscreen $SCR/repC $NHD/viral.snhskidx $NHD/bacterial.snhskidx $NHD/fungal.snhskidx $SCR/dump0 > $R/replayC.log 2>&1; echo "replay C exit $?"
NH_THREADS=4 SINGLET_NH_GPU_RESULTS=$SCR/gpu.res $M/bin/singlet-m13 nhscreen $SCR/repD $NHD/viral.snhskidx $NHD/bacterial.snhskidx $NHD/fungal.snhskidx $SCR/dump0 > $R/replayD.log 2>&1; echo "replay D exit $?"
grep -h "^\[m13t\]" $R/replayC.log | sed 's/^/C(CPU,4thr) /'; grep -h "^\[m13t\]" $R/replayD.log | sed 's/^/D(GPU,4thr) /'
{
  for X in C D; do ( cd $SCR/out && find nonhost -type f | sort ) | while read f; do
    a=$(sha256sum $SCR/out/$f | cut -c1-64); b=$(sha256sum $SCR/rep$X/$f 2>/dev/null | cut -c1-64); echo "FILE $f engine==replay$X:$([ "$a" = "$b" ] && echo yes || echo NO)"; done
    if [ -f $SCR/out/nonhost_reads.fq ]; then a=$(sha256sum $SCR/out/nonhost_reads.fq | cut -c1-64); b=$(sha256sum $SCR/rep$X/nonhost_reads.fq | cut -c1-64); echo "FILE nonhost_reads.fq engine==replay$X:$([ "$a" = "$b" ] && echo yes || echo NO)"; fi
  done
} >> $R/compare.txt 2>&1
grep -m1 -E "\.1fq: [0-9]+ reads" $R/engine.log > $R/n_pairs.txt
cp $SCR/out/nonhost/nonhost_summary.json $SCR/out/nonhost/nonhost_em_abundance.tsv $R/ 2>/dev/null
cp $SCR/dump0/params.txt $R/ 2>/dev/null; wc -l < $SCR/dump0/barcodes.txt > $R/n_barcodes.txt
grep -E "Elapsed|Maximum resident" $R/engine.time.txt; cat $R/gpumem.txt 2>/dev/null
echo "done $(date -Is) total $(( $(date +%s) - S0 )) s"
