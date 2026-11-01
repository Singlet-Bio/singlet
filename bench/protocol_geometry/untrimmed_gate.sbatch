#!/bin/bash
#SBATCH --job-name=s14-untrimmed-gate
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Real-read gate case for the untrimmed-R1 fix (GSM7064234 / SRR23617320, 3' v3, R1 = 151 nt) on the first 2 M spots, OLD vs NEW `singlet download`.
# Env: OLD, NEW (binaries, required). Checks:
#   1. OLD with only $SINGLET_WHITELIST_DIR set REJECTs the sample (the reported bug: env ignored -> no 10x whitelist -> split-seq);
#   2. NEW with only $SINGLET_WHITELIST_DIR set passes the gate as 10x and decodes;
#   3. its decoded R1 and R2 streams are identical to the ones OLD writes when told --protocol 10x-3p-v3 --whitelist-dir (R1 trimmed to 28 nt, no R1 base reaches the cDNA; the
#      container bytes differ only in the metadata block);
#   4. NEW with no whitelist anywhere passes through with a warning (rc != 2) instead of rejecting;
#   5. an ordinary 10x sample (GSM4987144 SRR13296300) gives a byte-identical .1fq with OLD and NEW.
set -uo pipefail
OLD=${OLD:?}; NEW=${NEW:?}
W=/mnt/projects/debruinz_project/debruinz/singlet; WL=$W/pipeline-benchmark/whitelists; export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
T=$W/jobs/s14/untrimmed_gate_$SLURM_JOB_ID; mkdir -p $T
UNT=$W/pilot/GSM7064234/sra/SRR23617320.sra; NORM=$W/pilot/GSM4987144/sra/SRR13296300.sra
N="--quality binned --vdb-threads 2 --no-profile --max-reads 2000000"
echo "OLD $($OLD --version 2>&1 | head -1) $(sha256sum $OLD | cut -c1-12)   NEW $($NEW --version 2>&1 | head -1) $(sha256sum $NEW | cut -c1-12)"
RC=0; ck() { if [ "$2" = ok ]; then echo "PASS: $1"; else echo "FAIL: $1"; RC=1; fi; }

SINGLET_WHITELIST_DIR=$WL $OLD download $UNT $N -o $T/old_env.1fq > $T/old_env.log 2>&1; r=$?; grep -a "singlet-gate" $T/old_env.log | cut -c1-260
[ $r = 2 ] && ck "1. OLD + env whitelist only: rejected (rc 2) - the bug" ok || ck "1. OLD + env whitelist only: rc $r (expected 2)" bad
SINGLET_WHITELIST_DIR=$WL $NEW download $UNT $N -o $T/new_env.1fq > $T/new_env.log 2>&1; r=$?; grep -a "singlet-gate" $T/new_env.log | cut -c1-260
[ $r = 0 ] && grep -aq "PASS_10X.*tag=10x-3p-v3" $T/new_env.log && ck "2. NEW + env whitelist only: PASS_10X 10x-3p-v3, rc 0" ok || ck "2. NEW + env whitelist only: rc $r" bad
$OLD download $UNT $N --protocol 10x-3p-v3 --whitelist-dir $WL -o $T/old_manual.1fq > $T/old_manual.log 2>&1; echo "OLD manual rc $?"
dsum() { $NEW decode $1 --r1 $T/$2.r1.fq --r2 $T/$2.r2.fq > $T/$2.dec.log 2>&1; echo "$(sha256sum < $T/$2.r1.fq | cut -c1-16) $(sha256sum < $T/$2.r2.fq | cut -c1-16) $(awk 'NR==2{print length($0); exit}' $T/$2.r1.fq)"; rm -f $T/$2.r1.fq $T/$2.r2.fq; }
a=$(dsum $T/new_env.1fq A); b=$(dsum $T/old_manual.1fq B)
echo "   decoded (R1 sha, R2 sha, R1 length): NEW auto: $a   OLD --protocol: $b   (container bytes differ only in the metadata: $(stat -c %s $T/new_env.1fq) vs $(stat -c %s $T/old_manual.1fq))"
[ "$a" = "$b" ] && [ "${a##* }" = 28 ] && ck "3. NEW auto reads == OLD --protocol 10x-3p-v3 reads: R1 trimmed to 28 nt, identical barcode/UMI and cDNA streams" ok || ck "3. decoded streams differ" bad
env -u SINGLET_WHITELIST_DIR $NEW download $UNT $N --whitelist-dir /nonexistent -o $T/new_nowl.1fq > $T/new_nowl.log 2>&1; r=$?; grep -a "singlet-gate" $T/new_nowl.log | cut -c1-260
[ $r != 2 ] && grep -aq "gate cannot tell" $T/new_nowl.log && ck "4. NEW with no whitelist: warning and pass-through (rc $r)" ok || ck "4. NEW with no whitelist: rc $r, no warning" bad
SINGLET_WHITELIST_DIR=$WL $OLD download $NORM $N --whitelist-dir $WL -o $T/old_norm.1fq > $T/old_norm.log 2>&1; ro=$?
SINGLET_WHITELIST_DIR=$WL $NEW download $NORM $N --whitelist-dir $WL -o $T/new_norm.1fq > $T/new_norm.log 2>&1; rn=$?
grep -a "singlet-gate" $T/new_norm.log | cut -c1-200
a=$(sha256sum < $T/old_norm.1fq | cut -c1-16); b=$(sha256sum < $T/new_norm.1fq | cut -c1-16)
[ "$ro$rn" = 00 ] && [ "$a" = "$b" ] && ck "5. ordinary 10x sample: .1fq identical OLD/NEW ($a)" ok || ck "5. ordinary sample: rc $ro/$rn, $a vs $b" bad
echo "gate rc=$RC"
