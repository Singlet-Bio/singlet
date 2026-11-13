#!/bin/bash
#SBATCH --job-name=identify-extras
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Extra identify runs on Clipper: endpoint survey of the 10x rows whose deposit has no barcode read, a real 10x BAM head, a staged .1fq head,
# species filters on three runs, and the route-status policy.   sbatch extras.sbatch BIN OUTNAME
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/identify; ENV=$W/envs/pbench
BIN=$1; OUT=$J/out/$2; mkdir -p $OUT
export LD_LIBRARY_PATH=$ENV/lib PATH=$ENV/bin:$PATH
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/home; trap 'rm -rf $SCR' EXIT
export HOME=$SCR/home
REF=/mnt/projects/debruinz_project/cellarium/reference
SP=$W/data/references/ve/species_filters
G=$REF/GRCh38-2024-A/star_2.7.11b
exp() { tr -d '\r' | sed "s#\$DATA#$W/data#g; s#\$REF#$REF#g"; }
echo "== endpoints"; exp < $J/nobc.tsv > $SCR/nobc.tsv; mkdir -p $OUT/endpoints
$BIN identify --list $SCR/nobc.tsv --out-dir $OUT/endpoints --whitelist-dir $J/wl --endpoints
echo "== bam and 1fq"; mkdir -p $OUT/files
printf 'bam_FCtx1_10X\tbam:%s\t\t%s\nonefq_SRR23582977\t%s\t\t%s\n' $W/data/heldout-launch/snisor_gse178175/illumina/FCtx1_10X_sc.bam $G $W/jobs/parity-v5/data/SRR23582977/SRR23582977.1fq $G > $SCR/files.tsv
$BIN identify --list $SCR/files.tsv --out-dir $OUT/files --whitelist-dir $J/wl
echo "== species"; mkdir -p $OUT/species_hint $OUT/species_all
printf 'GSM4477421\tSRR11549020\t\t%s\nGSM4593584\tSRR11938061\t\t%s\n' $G $G > $SCR/sp_h.tsv
$BIN identify --list $SCR/sp_h.tsv --out-dir $OUT/species_hint --whitelist-dir $J/wl --species-filters $SP --species-hint "Homo sapiens"
printf 'GSM4477421\tSRR11549020\t\t%s\nvisium_SRR22816622\tSRR22816622\t\t%s\n' $G $REF/GRCm39-2024-A/star_2.7.11b > $SCR/sp_a.tsv
$BIN identify --list $SCR/sp_a.tsv --out-dir $OUT/species_all --whitelist-dir $J/wl --species-filters $SP
echo "== route status"; mkdir -p $OUT/route_status
head -1 $SCR/sp_h.tsv > $SCR/rs.tsv
$BIN identify --list $SCR/rs.tsv --out-dir $OUT/route_status --whitelist-dir $J/wl --route-status $J/route_status.json
echo done
