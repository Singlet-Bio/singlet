#!/bin/bash
set -eo pipefail
cd /mnt/home/debruinz/Singlet-AI
export PYTHONPATH="/mnt/home/debruinz/Singlet-AI/singlepress:${PYTHONPATH:-}"
python3 scripts/merge_gse.py GSE117795 2>&1
