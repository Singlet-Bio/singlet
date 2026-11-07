"""Post-hoc doublet-threshold scan on a donor_assignments.tsv (uses best_singlet for cells called doublet).
Usage: thr_scan.py TRUTH.tsv.gz donor_assignments.tsv"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, roc_auc_score, average_precision_score
from common import load_truth, norm_bc, best_match_acc

T = load_truth(sys.argv[1])
d = pd.read_csv(sys.argv[2], sep="\t")
d["bc"] = [norm_bc(b) for b in d["cell"]]
d = d.drop_duplicates("bc").set_index("bc")
com = d.index.intersection(T.index)
t, p = T.loc[com], d.loc[com]
sing = (t["multiplets"] == "singlet").values
dbl = (t["multiplets"] == "doublet").values
ok = sing | dbl
print("AUROC %.4f AUPRC %.4f" % (roc_auc_score(dbl[ok], p["prob_doublet"].values[ok]),
                                 average_precision_score(dbl[ok], p["prob_doublet"].values[ok])))
print("prob_doublet distinct values: %d, == 1.0: %d of %d truth doublets" %
      (p["prob_doublet"].nunique(), int((p["prob_doublet"].values[dbl] >= 1.0).sum()), int(dbl.sum())))
print("%-8s %6s %6s %6s %7s %7s %6s" % ("thr", "ARI", "ARIasg", "sdbl", "ncall", "prec", "rec"))
for thr in [0.5, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999]:
    lab = np.where(p["prob_doublet"].values > thr, "doublet",
                   np.where(p["prob_max"].values >= 0.9, p["best_singlet"].values, "unassigned"))
    called = lab == "doublet"
    ari = adjusted_rand_score(t.loc[sing, "ind"], lab[sing])
    una = np.isin(lab, ["unassigned", "doublet"])
    ari_a = adjusted_rand_score(t.loc[sing & ~una, "ind"], lab[sing & ~una])
    print("%-8g %6.3f %6.3f %6.4f %7d %7.3f %6.3f" % (thr, ari, ari_a, called[sing].mean(), called.sum(),
          dbl[called].mean() if called.sum() else float("nan"), called[dbl].mean()))
# false doublet calls per truth donor at the default threshold
print("false-doublet rate by truth donor (prob_doublet > 0.9):")
fd = (p["prob_doublet"].values > 0.9)
for dn, g in t[sing].groupby("ind"):
    print("  %s: %.3f (n=%d)" % (dn, fd[g.index.map(lambda x: list(com).index(x))].mean() if False else pd.Series(fd, index=com)[g.index].mean(), len(g)))
