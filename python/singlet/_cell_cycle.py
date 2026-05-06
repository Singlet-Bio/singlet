"""Cell cycle scoring."""

from __future__ import annotations

import pandas as pd
from anndata import AnnData

# Canonical cell cycle genes (Tirosh et al. 2016 / Scanpy default)
S_GENES_HUMAN = [
    "MCM5",
    "PCNA",
    "TYMS",
    "FEN1",
    "MCM2",
    "MCM4",
    "RRM1",
    "UNG",
    "GINS2",
    "MCM6",
    "CDCA7",
    "DTL",
    "PRIM1",
    "UHRF1",
    "MLF1IP",
    "HELLS",
    "RFC2",
    "RPA2",
    "NASP",
    "RAD51AP1",
    "GMNN",
    "WDR76",
    "SLBP",
    "CCNE2",
    "UBR7",
    "POLD3",
    "MSH2",
    "ATAD2",
    "RAD51",
    "RRM2",
    "CDC45",
    "CDC6",
    "EXO1",
    "TIPIN",
    "DSCC1",
    "BLM",
    "CASP8AP2",
    "USP1",
    "CLSPN",
    "POLA1",
    "CHAF1B",
    "BRIP1",
    "E2F8",
]

G2M_GENES_HUMAN = [
    "HMGB2",
    "CDK1",
    "NUSAP1",
    "UBE2C",
    "BIRC5",
    "TPX2",
    "TOP2A",
    "NDC80",
    "CKS2",
    "NUF2",
    "CKS1B",
    "MKI67",
    "TMPO",
    "CENPF",
    "TACC3",
    "FAM64A",
    "SMC4",
    "CCNB2",
    "CKAP2L",
    "CKAP2",
    "AURKB",
    "BUB1",
    "KIF11",
    "ANP32E",
    "TUBB4B",
    "GTSE1",
    "KIF20B",
    "HJURP",
    "CDCA3",
    "HN1",
    "CDC20",
    "TTK",
    "CDC25C",
    "KIF2C",
    "RANGAP1",
    "NCAPD2",
    "DLGAP5",
    "CDCA2",
    "CDCA8",
    "ECT2",
    "KIF23",
    "HMMR",
    "AURKA",
    "PSRC1",
    "ANLN",
    "LBR",
    "CKAP5",
    "CENPE",
    "CTCF",
    "NEK2",
    "G2E3",
    "GAS2L3",
    "CBX5",
    "CENPA",
]


def score_cell_cycle(
    adata: AnnData,
    *,
    s_genes: list[str] | None = None,
    g2m_genes: list[str] | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Score cell cycle phase for each cell.

    Uses Tirosh-style scoring (mean expression of marker genes minus
    mean of control genes) to assign S and G2/M scores, then classifies
    each cell as G1, S, or G2M.

    Parameters
    ----------
    adata
        Annotated data matrix (should be log-normalized).
    s_genes
        S-phase marker genes. Default: Tirosh 2016 human genes.
    g2m_genes
        G2/M marker genes. Default: Tirosh 2016 human genes.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Adds to .obs:
        - 'S_score': S-phase score
        - 'G2M_score': G2/M-phase score
        - 'phase': Cell cycle phase (G1, S, or G2M)
    """
    import singlet

    adata = adata.copy() if copy else adata

    if s_genes is None:
        s_genes = S_GENES_HUMAN
    if g2m_genes is None:
        g2m_genes = G2M_GENES_HUMAN

    # Filter to genes present in data
    s_genes_present = [g for g in s_genes if g in adata.var_names]
    g2m_genes_present = [g for g in g2m_genes if g in adata.var_names]

    # Score S and G2M phases
    if len(s_genes_present) > 0:
        singlet.score_genes(adata, gene_list=s_genes_present, score_name="S_score")
    else:
        adata.obs["S_score"] = 0.0

    if len(g2m_genes_present) > 0:
        singlet.score_genes(adata, gene_list=g2m_genes_present, score_name="G2M_score")
    else:
        adata.obs["G2M_score"] = 0.0

    # Classify phase
    s_scores = adata.obs["S_score"].values
    g2m_scores = adata.obs["G2M_score"].values

    phases = []
    for s, g2m in zip(s_scores, g2m_scores):
        if s > g2m and s > 0:
            phases.append("S")
        elif g2m > s and g2m > 0:
            phases.append("G2M")
        else:
            phases.append("G1")

    adata.obs["phase"] = pd.Categorical(phases, categories=["G1", "S", "G2M"])

    return adata if copy else None
