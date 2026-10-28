#!/bin/bash
# Identity test for `singlet --engine gpu` (docs/engine-gpu.md): runs one sample three times with the same binary and compares.
#   1. cpu     --engine cpu                                   (the CPU engine; reference)
#   2. verify  --engine gpu + SINGLET_STAR_SEQCACHE_VERIFY=1  (STAR also maps every read that took a GPU result and counts any
#                                                              difference: must report verify_mismatches=0)
#   3. gpu     --engine gpu                                   (the GPU engine as used in production)
# then compares every output file of gpu (and verify) with cpu: byte for byte (bench/star_accel/out_equiv.py: JSON compared key by
# key, timing/path fields ignored), and for files that differ, order-independently (FASTQ records sorted, other files' lines
# sorted). Two CPU-only runs of the engine already differ in the order of unmapped reads and in timings; RUN_CPU2=1 adds a second
# cpu run to show that baseline.
#
# Usage: identity_test.sh SINGLET_BINARY SAMPLE.1fq OUTDIR [extra singlet args...]
# Env:   REF_BASE (default $SINGLET_REF_BASE), GENOME_DIR, GTF (defaults: GRCh38-2024-A under REF_BASE), THREADS (default 16),
#        SINGLET_ENGINE_GPU_DEVICE (default 0), RUN_CPU2=1. Needs python3 for the comparison.
# Exit 0 when verify_mismatches=0 and no file differs beyond what two CPU runs differ in.
set -uo pipefail
BIN=$1; FQ=$2; OUT=$3; shift 3
REF_BASE=${REF_BASE:-${SINGLET_REF_BASE:-}}
GENOME_DIR=${GENOME_DIR:-$REF_BASE/GRCh38-2024-A/star_2.7.11b}
GTF=${GTF:-$REF_BASE/GRCh38-2024-A/genes/genes.gtf}
THREADS=${THREADS:-16}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
mkdir -p "$OUT"
run() {   # name, engine (cpu|gpu), env...
  local name=$1 engine=$2; shift 2
  mkdir -p "$OUT/$name/out"
  local t0=$(date +%s)
  ( cd "$OUT/$name" && env "$@" "$BIN" "$FQ" --ref-base "$REF_BASE" --genome-dir "$GENOME_DIR" --exons "$GTF" --out-prefix "$OUT/$name/out/" \
      --threads "$THREADS" --pipeline --output-format 1pz --engine "$engine" "${EXTRA[@]}" > "$OUT/$name/run.log" 2>&1 )
  local rc=$?
  echo "$name: exit $rc, $(( $(date +%s) - t0 )) s wall"
  grep -h "ENGINE_GPU\|engine_gpu:\|SEQCACHE" -r "$OUT/$name/out" --include="*Log.out" 2>/dev/null | sed 's/^/    /'
}
EXTRA=("$@")
run cpu cpu SINGLET_STAR_SEQCACHE_VERIFY=0
[ "${RUN_CPU2:-0}" = 1 ] && run cpu2 cpu SINGLET_STAR_SEQCACHE_VERIFY=0
run verify gpu SINGLET_STAR_SEQCACHE_VERIFY=1
run gpu gpu SINGLET_STAR_SEQCACHE_VERIFY=0
status=0
vm=$(grep -ho "verify_mismatches=[0-9]*" -r "$OUT/verify/out" --include="*Log.out" | head -1)
echo "verify run: ${vm:-no verify line}"
[ "$vm" = "verify_mismatches=0" ] || status=1
compare() {   # a b
  python3 "$ROOT/bench/star_accel/out_equiv.py" "$1" "$2" 2>&1 | tail -15
  for f in $(cd "$1" && find . -type f | sort); do
    cmp -s "$1/$f" "$2/$f" && continue
    case $f in
      *.fq|*.fastq|*mate1|*mate2) ka=$(paste - - - - < "$1/$f" | LC_ALL=C sort | sha256sum | cut -c1-16)
                                  kb=$(paste - - - - < "$2/$f" | LC_ALL=C sort | sha256sum | cut -c1-16) ;;
      *) ka=$(LC_ALL=C sort "$1/$f" | sha256sum | cut -c1-16); kb=$(LC_ALL=C sort "$2/$f" | sha256sum | cut -c1-16) ;;
    esac
    echo "  differs: $f order-independent: $([ "$ka" = "$kb" ] && echo same || echo DIFFERENT)"
  done
}
for v in ${RUN_CPU2:+cpu2} verify gpu; do
  [ -d "$OUT/$v/out" ] || continue
  echo "=== cpu vs $v"
  compare "$OUT/cpu/out" "$OUT/$v/out" | tee "$OUT/compare_$v.txt"
done
# data files that differ even order-independently (logs, timings and run summaries excluded)
bad=$(grep "order-independent: DIFFERENT" "$OUT/compare_gpu.txt" | grep -vE "Log|read_stats|summary.json|provenance.json|pileup_stats.json|ambient_contamination.json|nonhost_summary.json" || true)
if [ -n "$bad" ]; then echo "DATA FILES DIFFER:"; echo "$bad"; status=1; fi
echo "RESULT: $([ $status = 0 ] && echo PASS || echo FAIL)"
exit $status
