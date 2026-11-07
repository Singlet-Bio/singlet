"""Shared helpers: truth table (Kang 2018 batch 2, ctrl lane) and metrics."""
import gzip, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

def load_truth(path):
    t = pd.read_csv(path, sep="\t", index_col=0)
    t = t[t["stim"] == "ctrl"].copy()
    t["bc"] = [i.split("-")[0] for i in t.index]
    t = t.set_index("bc")
    t["ind"] = t["ind"].astype("Int64").astype(str)
    return t   # cols: ind, stim, cluster, cell (type), multiplets (singlet|doublet|ambs)

def norm_bc(b):
    return str(b).split("-")[0]

def best_match_acc(truth, pred, unassigned=("unassigned", "doublet", "NA", "nan")):
    """Hungarian matching of predicted labels to truth donors; accuracy over cells with a donor call."""
    pred = pd.Series(pred); truth = pd.Series(truth)
    called = ~pred.isin(unassigned)
    ct = pd.crosstab(truth[called], pred[called])
    if ct.empty: return 0.0, {}
    r, c = linear_sum_assignment(-ct.values)
    mp = {ct.columns[j]: ct.index[i] for i, j in zip(r, c)}
    ok = sum(ct.values[i, j] for i, j in zip(r, c))
    return ok / max(called.sum(), 1), mp

def auc_ap(y, s):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    if y.min() == y.max(): return float("nan"), float("nan")
    return roc_auc_score(y, s), average_precision_score(y, s)

def recall_at_rate(y, s, rate):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    k = max(1, int(round(rate * len(y))))
    top = np.argsort(-s, kind="stable")[:k]
    return y[top].sum() / max(y.sum(), 1), y[top].mean()
