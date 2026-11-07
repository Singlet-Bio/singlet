#!/bin/bash
#SBATCH --job-name=s15-ena
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15
python3 $D/ena_discover.py $D/excl.txt GSM5682562,GSM8385345,GSM5130155,GSM5130163,GSM5171610,GSM7884577,GSM7064234,GSM8958776,GSM8141311,GSM8183378,GSM8164511,GSM4987144,GSM4987148,GSM5456917,GSM9038586,GSM5093884,GSM5093906 $D/ena
