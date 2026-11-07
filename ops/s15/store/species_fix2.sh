#!/bin/bash
# S15 store: xili_nuc_multi (human+mouse nuclei) and pilot_GSM9038586 (MDA-MB-231 xenograft, human+mouse) were run on GRCh38 only although the corpus lists human+mouse.
# Set the species, move the human-only STARsolo / rtools / edrops / cellbender results aside (index rows included, edited in place under the index lock), and put both ids at the front of the shared queue.
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; S=$W/benchmark-store; mkdir -p $S/_invalid_wrong_reference
for id in xili_nuc_multi pilot_GSM9038586; do dir=${id#pilot_}; [ "$id" = "$dir" ] && sample=$id || sample=$dir
  awk -F'\t' -v OFS='\t' -v i=$id 'NR==1{for(k=1;k<=NF;k++){if($k=="species")s=k; if($k=="fetch_note")n=k}} NR>1&&$1==i{$s="human+mouse"; $n="species corrected 2026-11-07: corpus lists human+mouse; STARsolo, rtools, edrops and cellbender rerun on the barnyard reference (earlier human-only runs in _invalid_wrong_reference)"}1' $S/datasets.tsv > $S/d.tmp && mv $S/d.tmp $S/datasets.tsv
  mkdir -p $S/_invalid_wrong_reference/$id
  for t in STARsolo rtools edrops cellbender; do [ -d $S/$id/$sample/$t ] && mv $S/$id/$sample/$t $S/_invalid_wrong_reference/$id/$t; done
  flock $S/index.tsv bash -c "awk -F'\t' -v i=$id '\$1==i && (\$3==\"STARsolo\"||\$3==\"rtools\"||\$3==\"edrops\"||\$3==\"cellbender\"){print >> \"$S/_invalid_wrong_reference/index_rows.tsv\"; next} {print}' $S/index.tsv > $S/i.tmp; cat $S/i.tmp > $S/index.tsv"
  P=$W/pilot/$dir; [ -L $P/starsolo ] && rm $P/starsolo; [ -d $P/starsolo ] && mv $P/starsolo $P/starsolo_human_only_$(date +%s); [ -d $P/starsolo_bam ] && mv $P/starsolo_bam $P/starsolo_bam_human_only_$(date +%s)
  flock $D/queue/store.queue.lock bash -c "sed -i \"/^$id\$/d\" $D/queue/store.queue; sed -i \"1i $id\" $D/queue/store.queue"
done
head -3 $D/queue/store.queue
