# sourced by the run_*.sbatch of this folder
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; X=$W/expression-sota/longread_usa
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
RAW=/mnt/projects/debruinz_project/debruinz/ground-truth/raw/PBMC-E     # read-only
# pbmc1 = ERR16108677 + ERR16108679 (ground-truth TRUTH_PBMC-E.md); R1 27 nt = 16 CB + 11 UMI, R2 89 nt
R1=$RAW/ERR16108677_1.fastq.gz,$RAW/ERR16108679_1.fastq.gz; R2=$RAW/ERR16108677_2.fastq.gz,$RAW/ERR16108679_2.fastq.gz
WL=$PB/whitelists/3M-february-2018.txt
