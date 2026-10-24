"""m_w4: check that the v2 read path reproduces the legacy AnnData of a whole study (X, layers, obs, var, uns keys).

    python -P tools/m_w4_equal.py <v2 root> <legacy bundle>
"""
import os
import sys
import tempfile
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
import numpy as np  # noqa: E402

os.environ["SINGLET_CACHE_DIR"] = tempfile.mkdtemp(prefix="m_w4_cache_")
from singlet.bundle import SingletBundle  # noqa: E402
from singlet.v2 import client  # noqa: E402

v2root, bundle = sys.argv[1], sys.argv[2]
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    leg = SingletBundle.open(bundle).to_anndata(verbose=False)
t_leg = time.time() - t0
t0 = time.time()
v2 = client.open_study("GSE145370", base=v2root).to_anndata()
t_v2 = time.time() - t0
print("legacy %.1fs  v2 %.1fs  shapes %s %s" % (t_leg, t_v2, leg.shape, v2.shape))
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print("%-34s %s" % (name, "OK" if cond else "DIFFERENT"))


check("obs_names identical (order)", list(leg.obs_names) == list(v2.obs_names))
check("var_names identical", list(leg.var_names) == list(v2.var_names))
check("var gene_name identical", list(leg.var["gene_name"]) == list(v2.var["gene_name"]))
check("obs columns identical", list(leg.obs.columns) == list(v2.obs.columns))
for c in leg.obs.columns:
    check("  obs[%s]" % c, leg.obs[c].astype(str).tolist() == v2.obs[c].astype(str).tolist())
check("X equal", (leg.X != v2.X).nnz == 0)
check("X dtype", leg.X.dtype == v2.X.dtype)
for k in ("spliced", "unspliced"):
    check("layer %s equal" % k, (leg.layers[k] != v2.layers[k]).nnz == 0)
check("uns keys", set(leg.uns) <= set(v2.uns) | {"singlet_bundle_path"})
check("uns study_meta equal", leg.uns["study_meta"] == v2.uns["study_meta"])
print("ALL EQUAL" if ok else "DIFFERENCES")
sys.exit(0 if ok else 1)
