"""Re-run universe: atlas samples + skipped 10x 'plate-based' samples, with every run's depth,
category flags (empty bundle, strand re-run, library-type screen) and cost inputs."""
import re
import sys

import numpy as np
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "."
g = pd.read_parquet(f"{d}/singlet_gsm.parquet")
plan = pd.read_csv(f"{d}/plan.tsv", sep="\t")
runs = pd.concat([pd.read_parquet(f"{d}/{f}") for f in ("runs.parquet", "runs_extra.parquet", "runs_extra2.parquet")])
runs = runs.drop_duplicates("run_accession").dropna(subset=["gsm"])
per = runs.groupby("gsm").agg(n_runs=("run_accession", "size"), reads=("read_count", "sum"),
                              bases=("base_count", "sum"), max_run_reads=("read_count", "max"))

atlas = g[(g.status.str.upper() == "DONE") & g.gse_id.isin(plan.gse)].copy()
atlas["source"] = "atlas"
plate = pd.read_parquet(f"{d}/sbc_geo_not_in_singlet.parquet")
plate = plate[plate.cat_processing_status == "skip_plate_based"]
plate = pd.DataFrame({"gsm_id": plate.geo_gsm, "gse_id": plate.cat_gse_id.fillna(plate.gse),
                      "organism": plate.organism, "n_cells": plate.obs_count, "source": "skipped_plate_10x"})
u = pd.concat([atlas[["gsm_id", "gse_id", "organism", "n_cells", "protocol", "source"]], plate], ignore_index=True)
u = u.drop_duplicates("gsm_id").merge(per, left_on="gsm_id", right_index=True, how="left")

empty = pd.read_csv(f"{d}/empty_bundle_suspects.tsv", sep="\t")
empty_gse = set(empty.iloc[:, 0])
strand = pd.read_csv(f"{d}/strand_rerun.csv")
lib = pd.read_csv(f"{d}/libtype_suspects.tsv", sep="\t")
lib = lib[lib.confidence.isin(["high", "medium"])]
u["empty_bundle"] = u.gse_id.isin(empty_gse) & (u.source == "atlas")
u["strand_rerun"] = u.gsm_id.isin(set(strand.gsm))
u["lib_class"] = u.gsm_id.map(dict(zip(lib.gsm_id, lib.lib_class)))
u["route"] = np.select(
    [u.lib_class.isin(["ADT", "HTO", "CMO / MULTI-seq", "CMO"]), u.lib_class.notna()],
    ["adt_modality", "exclude_non_gex"], "gex")
# order: (a) empty bundles, (b) strand re-runs, (c) skipped 10x, (d) rest by study cells
study_cells = u.groupby("gse_id").n_cells.sum()
u["study_cells"] = u.gse_id.map(study_cells)
u["wave"] = np.select([u.empty_bundle, u.strand_rerun, u.source == "skipped_plate_10x"], ["a_empty", "b_strand", "c_skipped10x"], "d_rest")
u.to_parquet(f"{d}/universe.parquet")
print(u.groupby(["wave", "route"]).size().unstack(fill_value=0))
print("runs known", u.n_runs.notna().sum(), "of", len(u))
