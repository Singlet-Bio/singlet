#!/bin/bash
# Smoke test of the generic arm_load of jobs/t2.sbatch. Needs bash. wsl.exe -e bash /mnt/c/.../tests/smoke_arm_load.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
src=$(sed -n '/^arm_load() {/,/^}/p' $HERE/../jobs/t2.sbatch)
eval "$src"
BIN_A=/w/a; COMMIT_A=''; ENV_A=(); ARGS_A=()
BIN_C=/w/c; COMMIT_C=abcd1234; ENV_C=(X=1 'Y=a b'); ARGS_C=(--no-snps --x)
arm_load A
[ "$BIN" = /w/a/singlet-gpu ] && [ ${#ARM_ENV[@]} = 0 ] && [ ${#ARM_ARGS[@]} = 0 ] || { echo FAIL A; exit 1; }
arm_load C
[ "$BINDIR" = /w/c ] && [ "$COMMIT_WANT" = abcd1234 ] && [ ${#ARM_ENV[@]} = 2 ] && [ "${ARM_ENV[1]}" = 'Y=a b' ] && [ "${ARM_ARGS[0]}" = --no-snps ] || { echo FAIL C; exit 1; }
echo "OK: arm_load handles empty and non-empty arrays under set -u"
