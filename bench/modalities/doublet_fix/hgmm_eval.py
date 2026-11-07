"""hgmm1k + 44 synthetic human/mouse doublets: precision / recall / AUROC of the new Singlet scorer next to the rows already measured
(old Singlet, scDblFinder, Scrublet).  Usage: hgmm_eval.py WORK_DBL_DIR OUT.csv NEW1.tsv[=label] ..."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
D, out = sys.argv[1:3]
truth = pd.read_csv(f"{D}/truth.csv"); t = truth.truth.values; syn = truth.synthetic.values
calls = {"Singlet old (header before fix)": f"{D}/singlet.tsv", "scDblFinder": f"{D}/scdblfinder.tsv",
         "Scrublet (rate 0.03)": f"{D}/scrublet_er0.03.tsv", "Scrublet (rate 0.08)": f"{D}/scrublet_er0.08.tsv"}
for a in sys.argv[3:]:
    p, _, lab = a.partition("="); calls[lab or p] = p
rows = []
for k, p in calls.items():
    d = pd.read_csv(p, sep="\t"); c = d.is_doublet.values.astype(bool)
    tp = int((c & t).sum()); fp = int((c & ~t).sum()); fn = int((~c & t).sum()); pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    rows.append(dict(tool=k, n_truth=int(t.sum()), n_called=int(c.sum()), tp=tp, fp=fp, fn=fn, precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-9),
                     auroc=roc_auc_score(t, d.score.values), auprc=average_precision_score(t, d.score.values), fp_among_natural_cells=int((c & ~t & ~syn).sum()),
                     frac_score_zero=float((d.score.values == 0).mean())))
o = pd.DataFrame(rows); o.to_csv(out, index=False); pd.set_option("display.width", 250); print(o.round(4).to_string(index=False))
