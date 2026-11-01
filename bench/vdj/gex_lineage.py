# SPDX-License-Identifier: MIT
"""T / B / other labels from a GEX matrix by marker genes, a V(D)J-independent check of contig lineage.

    python -P gex_lineage.py filtered_feature_bc_matrix.h5 OUT.csv

T: >= 2 UMIs over CD3D, CD3E, CD3G, CD2 and none over MS4A1, CD79A, CD79B; B: the reverse. Everything else is
``other`` (myeloid, NK without CD3, doublets, low-count cells) and is not scored.
"""

import sys

import numpy as np
import pandas as pd

T_GENES = ("CD3D", "CD3E", "CD3G", "CD2")
B_GENES = ("MS4A1", "CD79A", "CD79B")


def main(h5, out):
    import scanpy as sc

    a = sc.read_10x_h5(h5)
    a = a[:, a.var.get("feature_types", pd.Series("Gene Expression", index=a.var_names)) == "Gene Expression"]
    a.var_names_make_unique()

    def tot(genes):
        g = [x for x in genes if x in a.var_names]
        return np.asarray(a[:, g].X.sum(axis=1)).ravel()

    t, b = tot(T_GENES), tot(B_GENES)
    lin = np.where((t >= 2) & (b == 0), "T", np.where((b >= 2) & (t == 0), "B", "other"))
    pd.DataFrame({"barcode": a.obs_names, "lineage": lin}).to_csv(out, index=False)
    print(pd.Series(lin).value_counts().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:3])
