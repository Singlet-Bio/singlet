"""hgmm1k barnyard: exon feature -> gene map (ENSG / ENSMUSG id before the first underscore) for the synthetic-doublet matrix.
Usage: hgmm_prep.py FEATURES_TSV_GZ OUT.f2g   (rows of exon.* from dbl_synth.py are the rows of exon_counts_features.tsv.gz)"""
import sys, numpy as np, pandas as pd
feat = pd.read_csv(sys.argv[1], sep="\t", header=None)[0].astype(str).values
gi = pd.factorize(pd.Series([f.split("_")[0] for f in feat]))[0].astype(np.uint32)
gi.tofile(sys.argv[2]); print("features", len(feat), "genes", int(gi.max()) + 1)
