"""Fit a label rule on the leave-out 1000G reference q vectors (ref_q.tsv) and score it ONCE on the 150 MAGE individuals (test_q.tsv)
at each depth, against the current rule (largest proportion >= 0.8 else admixed).

Reports, before and after: coverage (share given a named label), named-call precision, accuracy with abstention = wrong, and
agreement with peddy's forced single label. The abstention threshold tau is chosen on the reference only (5-fold CV: the
smallest tau with <= 0.5% wrong named calls), never on the test individuals.

    python anc_fit.py ref_q.tsv test_q.tsv ancestry_calls.tsv OUT.json
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

POPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
ref = pd.read_csv(sys.argv[1], sep="\t"); test = pd.read_csv(sys.argv[2], sep="\t"); calls = pd.read_csv(sys.argv[3], sep="\t")
Q = ["q_" + p for p in POPS]

def feats(q):  # proportions plus their sqrt and pairwise log-ratios against EUR/AFR give the model room for the admixed groups
    q = np.clip(q, 1e-4, 1)
    return np.hstack([q, np.sqrt(q), np.log(q[:, [0, 1, 2, 4]] / q[:, [3]])])

def fit(X, y, C):
    return LogisticRegression(C=C, max_iter=5000).fit(feats(X), y)

X, y = ref[Q].to_numpy(), ref["super_pop"].to_numpy()
best = None
for C in (1, 10, 100, 1000):
    skf = StratifiedKFold(5, shuffle=True, random_state=0); P = np.zeros((len(y), 5))
    for a, b in skf.split(X, y):
        m = fit(X[a], y[a], C); P[b] = m.predict_proba(feats(X[b]))
    acc = (np.array(POPS)[P.argmax(1)] == y).mean()
    if best is None or acc > best[0]: best = (acc, C, P)
acc, C, P = best
pred, conf = np.array(POPS)[P.argmax(1)], P.max(1)
tau = None
for t in np.arange(0.5, 1.0, 0.01):
    named = conf >= t
    if named.sum() and (pred[named] != y[named]).mean() <= 0.005: tau = round(float(t), 2); break
tau = tau if tau is not None else 0.9
cov = (conf >= tau).mean()
print(f"reference CV: C={C} accuracy {acc:.4f}; tau={tau} coverage {cov:.3f} named error {(pred[conf>=tau]!=y[conf>=tau]).mean():.4f}")
model = fit(X, y, C)

truth = calls[(calls.dataset == "mage150") & (calls.tool == "singlet") & (calls.level == 8.0)].set_index("individual")["super_population"]
peddy = {lv: calls[(calls.dataset == "mage150") & (calls.tool == "peddy") & (calls.level == lv)].set_index("individual")["call"] for lv in (0.5, 2.0, 8.0)}
out = {"reference_cv": {"C": C, "accuracy": float(acc), "tau": tau, "coverage": float(cov)}, "test": {}}
for lv, g in test.groupby("level"):
    g = g.dropna(subset=Q).copy(); g["truth"] = g["individual"].map(truth)
    Xt = g[Q].to_numpy(); old = g["call"].fillna("none").to_numpy()
    Pt = model.predict_proba(feats(Xt)); top = np.array(POPS)[Pt.argmax(1)]; new = np.where(Pt.max(1) >= tau, top, "admixed")
    pd_ = g["individual"].map(peddy[float(lv)]).to_numpy()
    def score(lbl):
        named = lbl != "admixed"
        return {"coverage": float(named.mean()), "named_precision": float((lbl[named] == g.truth.to_numpy()[named]).mean()) if named.any() else None,
                "wrong_named": int((lbl[named] != g.truth.to_numpy()[named]).sum()), "accuracy_abstain_wrong": float((lbl == g.truth.to_numpy()).mean()),
                "agree_with_peddy_forced": float((np.where(named, lbl, top) == pd_).mean())}
    out["test"][str(lv)] = {"n": int(len(g)), "current_rule": score(old), "new_rule": score(new), "forced_top": float((top == g.truth.to_numpy()).mean())}
    print(lv, json.dumps(out["test"][str(lv)]))
json.dump(out, open(sys.argv[4], "w"), indent=1)
