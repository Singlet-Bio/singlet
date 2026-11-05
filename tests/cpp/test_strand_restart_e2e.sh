#!/bin/bash
# SPDX-License-Identifier: MIT
# tests/cpp/test_strand_restart_e2e.sh SINGLET_BIN MODE
#
# Regression test of gate 800361: the stream path's restart on the sorted path must give the outputs of a run without the GPU
# hand-off. SINGLET_TEST_FORCE_STRAND_RESTART=1 makes the stream pass restart at its end whatever the strand counts, so any small
# 10x library exercises the restart. MODE cpu runs the CPU engine; MODE gpu also runs --engine gpu with SINGLET_GPU_PILEUP=on and
# compares the two runs.
#
# Each run must: exit 0 within SINGLET_E2E_TIMEOUT seconds (default 1200; the GPU restart used to hang on the device lock); log the restart; leave no .gpu_handoff* file in the output folder; list none in provenance.json; and give a
# pileup that saw at least STAR's uniquely mapped reads. In MODE gpu the two runs must have the same pileup read count and the same
# star_Solo.out Gene matrix and barcodes.
#
# Input (skipped with exit 77 when unset, so the test passes on machines without a genome):
#   SINGLET_E2E_1FQ       a small 10x .1fq (Clipper: pipeline-benchmark/data/pbmc2m/pbmc2m.1fq)
#   SINGLET_E2E_GENOME    a reference folder with star_2.7.11b/ and genes/genes.gtf
#   SINGLET_E2E_REF_BASE  --ref-base (optional), SINGLET_E2E_THREADS (default 8), SINGLET_E2E_KEEP=1 keeps the outputs
set -uo pipefail
BIN=${1:?singlet binary}; MODE=${2:-cpu}
if [ -z "${SINGLET_E2E_1FQ:-}" ] || [ -z "${SINGLET_E2E_GENOME:-}" ]; then
  echo "SKIP: set SINGLET_E2E_1FQ and SINGLET_E2E_GENOME to run the forced strand restart end to end"; exit 77
fi
[ -f "$SINGLET_E2E_1FQ" ] || { echo "SKIP: no $SINGLET_E2E_1FQ"; exit 77; }
if [ "$MODE" = gpu ] && ! nvidia-smi -L >/dev/null 2>&1; then echo "SKIP: MODE gpu without a usable GPU"; exit 77; fi
G=$SINGLET_E2E_GENOME; T=${SINGLET_E2E_THREADS:-8}
D=$(mktemp -d "${TMPDIR:-/tmp}/singlet_restart_e2e.XXXXXX")
[ "${SINGLET_E2E_KEEP:-0}" = 1 ] || trap 'rm -rf "$D"' EXIT
FAIL=0
fail() { echo "FAIL: $*"; FAIL=1; }

run() {   # run NAME [extra args...]
  local n=$1; shift
  mkdir -p "$D/$n/out"
  ( cd "$D/$n" && SINGLET_TEST_FORCE_STRAND_RESTART=1 timeout -k 30 "${SINGLET_E2E_TIMEOUT:-1200}" "$BIN" "$SINGLET_E2E_1FQ" ${SINGLET_E2E_REF_BASE:+--ref-base "$SINGLET_E2E_REF_BASE"} \
      --genome-dir "$G/star_2.7.11b" --exons "$G/genes/genes.gtf" --out-prefix "$D/$n/out/" --threads "$T" --pipeline \
      --output-format 1pz ${SINGLET_WHITELIST_DIR:+--whitelist-dir "$SINGLET_WHITELIST_DIR"} "$@" > "$D/$n/run.log" 2>&1 )
  local rc=$?
  echo "$n: exit $rc"
  [ $rc = 124 ] && fail "$n: no exit within ${SINGLET_E2E_TIMEOUT:-1200} s (a hang)"
  [ $rc = 0 ] || { fail "$n exited $rc"; tail -20 "$D/$n/run.log"; }
  if ! grep -aq "unsorted STAR stream" "$D/$n/run.log"; then
    fail "$n: the first pass was not on the stream path, so nothing could restart"
    grep -a -E "BAM:|sorted|whitelist|--stream-unsorted ignored" "$D/$n/run.log" | head -5 | cut -c1-300 | sed 's/^/    /'
  fi
  grep -aq "restarting on the sorted path" "$D/$n/run.log" || fail "$n: the restart did not happen"
  grep -aq "pileup read .* mapped records, but STAR mapped" "$D/$n/run.log" && fail "$n: pileup shortfall ($(grep -a 'pileup read' "$D/$n/run.log" | tail -1))"
  local left; left=$(ls -A "$D/$n/out" | grep '^\.gpu_handoff' || true)
  [ -z "$left" ] || fail "$n: hand-off files left in the output folder: $left"
  if [ -f "$D/$n/out/provenance.json" ]; then
    grep -q '\.gpu_handoff' "$D/$n/out/provenance.json" && fail "$n: provenance.json lists a hand-off file"
  else fail "$n: no provenance.json"; fi
  local pil uniq
  pil=$(grep -ao "\[singlet\] Pileup: [0-9]* reads" "$D/$n/run.log" | tail -1 | grep -o "[0-9]*")
  uniq=$(awk -F'|' '/Uniquely mapped reads number/{gsub(/[ \t]/,"",$2); print $2}' "$D/$n/out/star_Log.final.out" 2>/dev/null)
  echo "$n: pileup ${pil:-?} records, STAR uniquely mapped ${uniq:-?} reads"
  if [ -z "$pil" ] || [ -z "$uniq" ] || [ "$pil" -lt "$uniq" ]; then fail "$n: the pileup saw fewer records than STAR's uniquely mapped reads"; fi
  echo "$pil" > "$D/$n/pileup_reads"
}

run cpu --engine cpu
if [ "$MODE" = gpu ]; then
  SINGLET_GPU_PILEUP=on run gpu --engine gpu
  cmp -s "$D/cpu/pileup_reads" "$D/gpu/pileup_reads" || fail "pileup reads differ: cpu $(cat "$D/cpu/pileup_reads") gpu $(cat "$D/gpu/pileup_reads")"
  for f in Gene/raw/matrix.mtx Gene/raw/barcodes.tsv Gene/filtered/barcodes.tsv Gene/filtered/matrix.mtx; do
    if [ ! -f "$D/cpu/out/star_Solo.out/$f" ]; then fail "star_Solo.out/$f: missing in the cpu run"
    elif cmp -s "$D/cpu/out/star_Solo.out/$f" "$D/gpu/out/star_Solo.out/$f"; then echo "star_Solo.out/$f: identical"
    else fail "star_Solo.out/$f differs between cpu and gpu"; fi
  done
fi
[ "${SINGLET_E2E_KEEP:-0}" = 1 ] && echo "outputs kept in $D"
[ $FAIL = 0 ] && { echo "test_strand_restart_e2e ($MODE): PASS"; exit 0; }
echo "test_strand_restart_e2e ($MODE): FAIL"; exit 1
