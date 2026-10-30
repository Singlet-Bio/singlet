#!/bin/bash
#SBATCH --job-name=w11-ing-run
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --nice=10000
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# One sample: (a) transform .1fq -> .1fg (codec / chunk-size variants), (b) GPU reader, (c) identity check against `singlet decode`.
# Env: SAMPLE_LABEL, SAMPLE_1FQ, optional BIN (default bin2), VARIANTS (default all). Reads and .1fg files live only in job /tmp.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
D=$W/jobs/w11/ingest; M8=$W/jobs/w11/m8; BIN=${BIN:-$D/bin2}
SAMPLE_LABEL=${SAMPLE_LABEL:?}; Q=${SAMPLE_1FQ:?}
OUT=$D/results/${SAMPLE_LABEL}_$SLURM_JOB_ID; mkdir -p $OUT; R=$OUT/results.jsonl
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
LIBS="--env LD_LIBRARY_PATH=$M8/lib:$D/nvcomp/lib64"
ctr() { apptainer exec --nv -B $W $LIBS $IMG "$@"; }
echo "== node $(hostname) job $SLURM_JOB_ID $(date -Is) sample $SAMPLE_LABEL $Q ($(stat -c %s $Q) B) bin $BIN"
nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv
lscpu | grep -E "Model name|^CPU\(s\)"; taskset -cp $$; df -h /tmp | tail -n 1
run() { echo "[$(date +%T)] $*" >&2; "$@" >> $R 2> $OUT/stderr.last || { echo "FAILED rc=$? $*" >&2; cat $OUT/stderr.last >&2; }; tail -n 2 $OUT/stderr.last >&2 || true; }
tag() { sed -i "\$s/^{/{\"sample\":\"$SAMPLE_LABEL\",\"run\":\"$1\",/" $R; }
echo '{"sample":"'$SAMPLE_LABEL'","info":"node","host":"'$(hostname)'","job":'$SLURM_JOB_ID',"q_bytes":'$(stat -c %s $Q)'}' >> $R

echo "== selftest (fast parser vs library decode)"
run ctr $BIN/enc_1fg --in $Q --selftest 60; tag selftest
echo "== (a) transform. Timing runs: 1 thread (core-s per M pairs), 4 and 8 threads (rate)"
ENC="ctr $BIN/enc_1fg --in $Q"
run $ENC --out $S/t1.1fg --codec zstd --level 1 --threads 1 --crc 1 --chunk-pairs 32768;  tag enc_zstd1_c32k_t1_crc1;  rm -f $S/t1.1fg
run $ENC --out $S/t1.1fg --codec zstd --level 1 --threads 1 --crc 0 --chunk-pairs 32768;  tag enc_zstd1_c32k_t1_crc0;  rm -f $S/t1.1fg
run $ENC --out $S/t1.1fg --codec zstd --level 1 --threads 1 --crc 1 --chunk-pairs 32768 --raw-sections r2; tag enc_zstd1_c32k_r2raw_t1_crc1; rm -f $S/t1.1fg
run $ENC --out $S/t1.1fg --codec raw --threads 1 --crc 1 --chunk-pairs 32768;              tag enc_raw_c32k_t1_crc1;    rm -f $S/t1.1fg
run $ENC --out $S/t1.1fg --codec zstd --level 1 --threads 4 --crc 1 --chunk-pairs 32768;  tag enc_zstd1_c32k_t4_crc1;  rm -f $S/t1.1fg
run $ENC --out $S/t1.1fg --codec zstd --level 1 --threads 8 --crc 1 --chunk-pairs 32768;  tag enc_zstd1_c32k_t8_crc1;  rm -f $S/t1.1fg
echo "== variant files (4 threads): name codec chunk-pairs extra"
mk() { local name=$1 chunk=$2; shift 2; run $ENC --out $S/$name.1fg --threads 4 --crc 1 --chunk-pairs $chunk "$@"; tag enc_$name; }
mk z1_c256k 262144 --codec zstd --level 1
mk z1_c64k   65536 --codec zstd --level 1
mk z1_c32k   32768 --codec zstd --level 1
mk z1_c16k   16384 --codec zstd --level 1
mk z3_c32k   32768 --codec zstd --level 3
mk z1_c32k_r2raw 32768 --codec zstd --level 1 --raw-sections r2
mk raw_c32k  32768 --codec raw
ls -l $S

echo "== (b) reader (batch = about 4 M pairs: B = 4 M / chunk pairs)"
RD="ctr $BIN/read_1fg"
bc() { echo $(( 4194304 / $1 )); }
declare -A CH=( [z1_c256k]=262144 [z1_c64k]=65536 [z1_c32k]=32768 [z1_c16k]=16384 [z3_c32k]=32768 [z1_c32k_r2raw]=32768 [raw_c32k]=32768 )
for f in z1_c256k z1_c64k z1_c32k z1_c16k z3_c32k z1_c32k_r2raw raw_c32k; do
  B=$(bc ${CH[$f]})
  run $RD --file $S/$f.1fg --gpu-decode-bench 2 --reps 30 --batch-chunks $B --label $f; tag gpu_decode_bench_$f
  for t in 1 2 3; do run $RD --file $S/$f.1fg --threads $t --batch-chunks $B --label ${f}_t$t; tag read_e2e_${f}_t$t; done
done
f=z1_c32k; B=$(bc 32768)
run $RD --file $S/$f.1fg --threads 2 --batch-chunks $B --crc 1 --label ${f}_t2_crc;   tag read_e2e_${f}_t2_crc1
run $RD --file $S/$f.1fg --threads 2 --batch-chunks 256 --label ${f}_b256_t2; tag read_e2e_${f}_b256_t2
run $RD --file $S/$f.1fg --gpu-decode-bench 2 --reps 30 --batch-chunks 256 --label ${f}_b256; tag gpu_decode_bench_${f}_b256
for f in z1_c32k raw_c32k z1_c32k_r2raw; do for t in 1 2; do run $RD --file $S/$f.1fg --threads $t --batch-chunks $B --no-decode --label ${f}_h2d_t$t; tag read_h2donly_${f}_t$t; done; done
run $RD --file $S/raw_c32k.1fg --threads 2 --batch-chunks $B --crc 1 --label raw_c32k_t2_crc; tag read_e2e_raw_c32k_t2_crc1
echo "== CPU-engine compatibility: host decode of .1fg into byte-numeric reads vs the .1fq library decode (hash of every read)"
DEC="ctr $BIN/dec_1fg"
tag2() { n=$(wc -l < $R); sed -i "$((n-1)),\$s/^{/{\"sample\":\"$SAMPLE_LABEL\",\"run\":\"$1\",/" $R; }
run $DEC --fg $S/z1_c32k.1fg --fq $Q --threads 4; tag2 cpu_decode_compare_t4_hash
run $DEC --fg $S/z1_c32k.1fg --fq $Q --threads 1 --nohash; tag2 cpu_decode_compare_t1_nohash
echo "== m8 baseline on the .1fq (decode + pack + H2D, no quality)"
m8() { apptainer exec --nv -B $W --env LD_LIBRARY_PATH=$M8/lib $IMG $M8/bin/m8_ingest "$@"; }
for t in 1 3; do echo "[m8 t$t]"; m8 --file $Q --threads $t --qual none --label m8_t$t >> $R 2>> $OUT/m8.err; tag m8_baseline_t$t; done

echo "== (c) identity: read_1fg --verify vs singlet decode text (decoded to files in job /tmp first: decode writes all of R1 of a batch before R2, so FIFOs deadlock)"
SW=$W/pipeline-benchmark/bin/singlet-w3
/usr/bin/time -f "DECODE cpu_user=%U cpu_sys=%S wall=%e" $SW decode $Q --r1 $S/r1.fq --r2 $S/r2.fq --threads 8 2> $OUT/decode.err
tail -n 2 $OUT/decode.err; ls -l $S/r1.fq $S/r2.fq
for f in z1_c32k raw_c32k z1_c32k_r2raw; do
  run $RD --file $S/$f.1fg --threads 2 --batch-chunks $B --verify $S/r1.fq $S/r2.fq --label verify_$f; tag verify_$f
done
echo "== done $(date -Is)"
grep -c . $R; cut -c1-300 $R | tail -n 8
