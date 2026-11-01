"""Export the ancestry label rule that python/singlet/celllabels/ships (data/ancestry_label_model.json) from ref_q.tsv.

    python anc_export.py ref_q.tsv OUT.json

Pipeline: anc_q.py (reference + test proportions on Clipper) -> anc_fit.py (cross-validation, test scoring) -> this script.
The model is a multinomial logistic regression (C = 1) on the features of anc_fit.feats(); tau = 0.8 (the existing q_min convention;
the cross-validated choice on exact-genotype q never abstains because those q are overconfident); the out-of-distribution guard is
the 99.5th percentile of the leave-one-out nearest-neighbour L1 distance between reference proportions.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

POPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
Q = ["q_" + p for p in POPS]
CLIP = 1e-4


def feats(q):
    q = np.clip(q, CLIP, 1)
    return np.hstack([q, np.sqrt(q), np.log(q[:, [0, 1, 2, 4]] / q[:, [3]])])


ref = pd.read_csv(sys.argv[1], sep="\t")
X, y = ref[Q].to_numpy(), ref["super_pop"].to_numpy()
m = LogisticRegression(C=1.0, max_iter=5000).fit(feats(X), y)
R = np.round(X, 3)
D = np.abs(R[:, None, :] - R[None, :, :]).sum(2)
np.fill_diagonal(D, 9)
out = {
    "version": "1",
    "description": "Super-population label rule on the supervised-admixture proportions q: multinomial logistic regression on [q, sqrt(q), log(q_k/q_EUR) for k in AFR,AMR,EAS,SAS]; the label is the most probable class when its probability is >= tau and q is near a reference individual, otherwise 'admixed'.",
    "classes": list(m.classes_), "coef": m.coef_.tolist(), "intercept": m.intercept_.tolist(), "tau": 0.8, "q_clip": CLIP, "C": 1.0,
    "ood_l1": round(float(np.quantile(D.min(1), 0.995)), 3),
    "ood_note": "abstain when the L1 distance from q to the nearest reference individual exceeds ood_l1 (99.5th percentile of the leave-one-out nearest-neighbour distance among the reference individuals): the rule is not trusted away from where it was trained",
    "training": {"individuals": int(len(ref)),
                 "source": "1000 Genomes phase 3 GRCh38 reference individuals that are not benchmark test individuals (leave-out), q fitted from exact genotypes on a fixed random 300,000-site subset of the panel with leave-one-out super-population frequencies",
                 "reference_cv_accuracy": 0.9983},
    "ref_q": R.tolist(),
}
json.dump(out, open(sys.argv[2], "w"), separators=(",", ":"))
print("ood_l1", out["ood_l1"], "individuals", len(ref))
