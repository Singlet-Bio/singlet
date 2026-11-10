"""Readers for the donor labels of Singlet, vireo and souporcell, and the per-pool demultiplexing scores.
Labels only: no genotype is read here."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "capabilities"))
from _gcommon import ari, contingency, hungarian  # noqa: E402

MIN_TRUTH_DONOR_CELLS = 20
NO_DONOR = {"", "<NA>", "nan", "NA"}
UNASSIGNED = {"unassigned", "doublet", "NA", "nan", "None", ""}


def norm_bc(b) -> str:
    return str(b).split("-")[0]


def read_singlet(run_dir: str | Path) -> pd.DataFrame:
    """Singlet donor labels for the called cells: index barcode, columns label, prob_doublet."""
    run_dir = Path(run_dir)
    f = run_dir / "donor" / "donor_assignments.tsv"
    if not f.exists():
        f = run_dir / "donor_assignments.tsv"
    d = pd.read_csv(f, sep="\t")
    if "cell_status" in d:
        d = d[d["cell_status"] == "cell"]
    d["bc"] = d["cell"].map(norm_bc)
    d = d.drop_duplicates("bc").set_index("bc")
    return pd.DataFrame({"label": d["donor_id"].astype(str), "prob_doublet": d.get("prob_doublet", np.nan)})


def singlet_k_from_log(run_dir: str | Path):
    """K Singlet selected, from its own log line, and the number of distinct donor labels it wrote."""
    k = None
    log = Path(run_dir) / "run.log"
    if log.exists():
        for line in log.read_text(errors="ignore").splitlines():
            m = re.search(r"auto-K: selected K=(\d+)", line)
            if m:
                k = int(m.group(1))
            m = re.search(r"\[demux\].*(?:K|donors?)\s*=\s*(\d+)", line)
    lab = read_singlet(run_dir)["label"]
    n = lab[~lab.isin(UNASSIGNED)].nunique()
    return k, int(n)


def read_vireo(d: str | Path) -> pd.DataFrame:
    x = pd.read_csv(Path(d) / "donor_ids.tsv", sep="\t")
    x["bc"] = x["cell"].map(norm_bc)
    x = x.drop_duplicates("bc").set_index("bc")
    return pd.DataFrame({"label": x["donor_id"].astype(str), "prob_doublet": x.get("prob_doublet", np.nan)})


def read_souporcell(d: str | Path) -> pd.DataFrame:
    x = pd.read_csv(Path(d) / "clusters.tsv", sep="\t")
    x["bc"] = x["barcode"].map(norm_bc)
    x = x.drop_duplicates("bc").set_index("bc")
    lab = np.where(x["status"] == "singlet", "cluster" + x["assignment"].astype(str).str.split("/").str[0],
                   np.where(x["status"] == "doublet", "doublet", "unassigned"))
    return pd.DataFrame({"label": lab}, index=x.index)


def vireo_khat(d: str | Path, frac: float = 0.01) -> int:
    """K estimated from an over-specified vireo run: donors holding at least `frac` of the cells that got a donor."""
    lab = read_vireo(d)["label"]
    lab = lab[~lab.isin(UNASSIGNED)]
    vc = lab.value_counts()
    return int((vc >= frac * len(lab)).sum())


def score(truth: pd.DataFrame, pred: pd.Series) -> dict:
    """truth: index barcode, columns ind, multiplets. pred: label per barcode (missing barcode = not called).
    Scored on the truth singlets (demuxlet class 'singlet'); 'doublet' / 'unassigned' / missing count as their own labels."""
    t = truth[(truth["multiplets"] == "singlet") & ~truth["ind"].isin(NO_DONOR)]
    # demuxlet occasionally calls a handful of cells for a donor that is not in the lane (1 to 10 cells); a donor needs at
    # least 20 truth singlets to count (the true K of every Kang lane is the number of donors above this size)
    n_by = t["ind"].map(t["ind"].value_counts())
    t = t[n_by >= MIN_TRUTH_DONOR_CELLS]
    p = pred.reindex(t.index).fillna("not_called").astype(str)
    called = p[~p.isin(UNASSIGNED | {"not_called"})]
    out = dict(n_truth_singlets=int(len(t)), n_scored=int(len(called)))
    out["ari"] = ari(t["ind"].values, p.values)
    m_ass = t.index.isin(called.index)
    out["ari_assigned"] = ari(t["ind"].values[m_ass], p.values[m_ass]) if m_ass.sum() > 1 else float("nan")
    out["coverage"] = float(m_ass.mean())
    ct = contingency(t["ind"].values[m_ass], p.values[m_ass]) if m_ass.sum() else np.zeros((0, 0))
    if ct.size:
        pairs = hungarian(ct)
        out["accuracy_assigned"] = float(sum(ct[i, j] for i, j in pairs) / ct.sum())
        tr_codes = np.unique(t["ind"].values[m_ass].astype(str))
        tsize = pd.Series(t["ind"].values).value_counts()
        rec = []
        for i, j in pairs:
            rec.append(ct[i, j] / max(int((t["ind"].values == tr_codes[i]).sum()), 1))
        rec += [0.0] * (len(tsize) - len(rec))
        out["min_donor_recall"] = float(min(rec))
    else:
        out["accuracy_assigned"] = float("nan"); out["min_donor_recall"] = 0.0
    out["k_called"] = int(called.nunique())
    return out
