#!/bin/bash
# Smoke test of jobs/lib_run.sh with a fake binary. Needs bash with flock and /usr/bin/time (Linux or WSL).
#   wsl.exe -e bash /mnt/c/.../bench/gpu_fast/tests/smoke_lib_run.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d); trap 'rm -rf $T' EXIT
SCR=$T/scr; RES0=$T/res; REF=$T/ref; PB=$T/pb; DATA2=$T/d2; mkdir -p $SCR/data $RES0 $REF/GRCh38-2024-A/star_2.7.11b $REF/GRCh38-2024-A/genes $PB/data/s1 $DATA2
echo reads > $PB/data/s1/s1.1fq
cat > $T/fake <<'EOF'
#!/bin/bash
echo "ENV FOO=${FOO:-unset} SLOTS=${SINGLET_ENGINE_GPU_SLOTS:-unset}"
echo "ARGS $*"
out=""; while [ $# -gt 0 ]; do [ "$1" = "--out-prefix" ] && out=$2; shift; done
mkdir -p ${out}star_Solo.out/Gene
echo x > ${out}star_Solo.out/Gene/CellReads.stats
echo "Number of input reads |	100" > ${out}star_Log.final.out
echo "E11: Solo on the device" > ${out}star_Log.out; sleep 0.6
EOF
chmod +x $T/fake
BIN=$T/fake; . $HERE/../jobs/lib_run.sh
ARM_ENV=(FOO=bar); ARM_ARGS=(--no-snps --x)
ITEMS=(s1 s1 s1 s1 s1 s1); p=pt; mkdir -p $SCR/$p $RES0/$p; echo 0 > $SCR/$p/queue; : > $RES0/$p/runs.tsv
SINGLET_ENGINE_GPU_SLOTS=2
export SINGLET_ENGINE_GPU_SLOTS
worker $p 0 7 & worker $p 1 7 & sleep 0.9; touch $SCR/$p/STOP; wait
n=$(wc -l < $RES0/$p/runs.tsv)
grep -q "ENV FOO=bar SLOTS=2" $RES0/$p/c0/run.log.ts || { echo FAIL env; cat $RES0/$p/c0/run.log.ts; exit 1; }
grep -q "ARGS .*--engine gpu --no-snps --x" $RES0/$p/c0/run.log.ts || { echo FAIL args; cat $RES0/$p/c0/run.log.ts; exit 1; }
[ -f $RES0/$p/c0/CellReads.stats ] || { echo FAIL crs; exit 1; }
[ "$n" -lt 6 ] || { echo "FAIL STOP did not stop the queue ($n runs)"; exit 1; }
echo "OK: $n runs, env and args reached the binary, STOP ended the queue"
