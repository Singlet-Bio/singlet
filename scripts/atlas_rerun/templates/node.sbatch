#!/bin/bash
# Atlas re-run node packer (Anvil whole nodes: 128 cores, 257 GB, charged all 128 cores).
# One genome per reference in /dev/shm (`singlet genome load`), then the batch's staged samples on <slot_cores> cores
# (28 four-thread samples on 112 cores by default; a >1.3B-pair sample takes 16), longest first, disk BAM sort, each
# started only if its predicted finish x1.3 is before the job's end. Samples not started are released to the next node.
# The genome is unloaded on exit, including on SIGTERM (--signal=B:TERM@600 from the controller).
#   sbatch [opts] node.sbatch <orchestrator.py> <config.json> <cluster> <batch.jsonl> <slot_cores>
set -u
export PYTHONUNBUFFERED=1
ORCH=$1; CONFIG=$2; CLUSTER=$3; BATCH=$4; SLOTS=${5:-112}
exec "${AR_PYTHON:-python3}" "$ORCH" node-stage --config "$CONFIG" --cluster "$CLUSTER" --batch "$BATCH" \
  --slot-cores "$SLOTS" --time-limit "${AR_TIME:-24:00:00}"
