#!/usr/bin/env bash
# Train every Track C model (leave-one-study-out for each test study + 'none' = all studies)
cd "$(dirname "$0")/.."
for mem in exon intron; do
  for h in none GSE142444 GSE244263 GSE152522 GSE145370; do python -P tools/track_c_train.py human $mem $h | cut -c1-110; done
  for h in none GSE285936; do python -P tools/track_c_train.py mouse $mem $h | cut -c1-110; done
done
