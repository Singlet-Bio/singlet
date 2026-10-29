#!/usr/bin/env python
"""Extra Track C models for the cross-domain and within-domain tests (wraps tools/track_c_train.py without editing it).

    python -P tools/track_c_train_extra.py xfer mouse exon        # cross-species: feature-independent tables learned on HUMAN only
                                                                 #   -> results/track_c_model_mouse_exon_xhuman.bin   (eval with --holdout xhuman)
    python -P tools/track_c_train_extra.py pool human exon pbmc GSE165080[,GSE...]
                                                                 # within-domain: training pool restricted to these studies
                                                                 #   -> results/track_c_model_human_exon_pbmc.bin     (eval with --holdout pbmc)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track_c_train as tt

mode = sys.argv[1]
if mode == 'xfer':
    tt.main_xfer(sys.argv[2], sys.argv[3])
elif mode == 'pool':
    org, mem, tag, studies = sys.argv[2], sys.argv[3], sys.argv[4], set(sys.argv[5].split(','))
    keep = {k: v for k, v in tt.STUDY.items() if v in studies or tt.ORG[k] != org}
    tt.STUDY = keep                       # main() builds its pool from this dict
    tt.main(org, mem, tag)
else:
    raise SystemExit(__doc__)
