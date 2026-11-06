#!/bin/bash
# S15 store (Slurm job): the two pilot cell-line mixtures GSM4987144 and GSM4987148 contain the mouse line EL4 (ENA: Homo sapiens + Mus musculus) but were run on GRCh38 only.
# Move the human-only STARsolo / rtools / edrops / cellbender / kraken results aside and rerun them on the human+mouse (barnyard) reference. kb, alevin, mgatk, vireo and souporcell stay as human-reference results (human reads only).
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; S=$W/benchmark-store; mkdir -p $S/_invalid_wrong_reference
for g in GSM4987144 GSM4987148; do id=pilot_$g; P=$W/pilot/$g; mkdir -p $S/_invalid_wrong_reference/$id
  for t in STARsolo rtools edrops cellbender kraken; do [ -d $S/$id/$g/$t ] && mv $S/$id/$g/$t $S/_invalid_wrong_reference/$id/$t; grep -P "^$id\t$g\t$t\t" $S/index.tsv >> $S/_invalid_wrong_reference/index_rows.tsv; grep -v -P "^$id\t$g\t$t\t" $S/index.tsv > $S/i.tmp && mv $S/i.tmp $S/index.tsv; done
  [ -L $P/starsolo ] || mv $P/starsolo $P/starsolo_human_only; rm -rf $P/starsolo_bam $P/starsolo; for t in rtools edrops cellbender kraken; do mv $P/tools/$t $P/tools/${t}_human_only 2> /dev/null; done
  SPECIES=human+mouse bash $D/submit_gap.sh $g rtools,edrops,cellbender,kraken
done
