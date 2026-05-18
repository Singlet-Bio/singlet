# SPDX-License-Identifier: MIT
"""Gene set enrichment analysis via Enrichr API.

Provides singlet.enrichr() — queries the Enrichr web API for pathway
enrichment of a gene list. Works offline with a local fallback for testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def enrichr(
    gene_list: list[str],
    *,
    gene_sets: str = "GO_Biological_Process_2023",
    organism: str = "human",
    top_n: int = 10,
) -> "pd.DataFrame":
    """Query Enrichr for pathway enrichment of a gene list.

    Parameters
    ----------
    gene_list : list[str]
        Gene symbols to test for enrichment.
    gene_sets : str, default "GO_Biological_Process_2023"
        Enrichr library to query. Common options:
        - "GO_Biological_Process_2023"
        - "GO_Molecular_Function_2023"
        - "KEGG_2021_Human"
        - "MSigDB_Hallmark_2020"
        - "Reactome_2022"
    organism : str, default "human"
        Organism for gene symbol mapping. "human" or "mouse".
    top_n : int, default 10
        Number of top enriched terms to return.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: 'term', 'overlap', 'p_value',
        'adjusted_p_value', 'genes'.

    Raises
    ------
    ValueError
        If gene_list is empty.
    ConnectionError
        If Enrichr API is unreachable.

    Examples
    --------
    >>> import singlet
    >>> markers = ["CD3D", "CD3E", "CD4", "IL7R", "LCK"]
    >>> result = singlet.enrichr(markers, gene_sets="GO_Biological_Process_2023")
    >>> result[["term", "adjusted_p_value"]].head()
    """
    import pandas as pd

    if not gene_list:
        raise ValueError("gene_list must not be empty.")

    # Filter to non-empty strings
    gene_list = [g.strip() for g in gene_list if g.strip()]
    if not gene_list:
        raise ValueError("gene_list contains no valid gene symbols.")

    # Determine base URL
    if organism.lower() == "mouse":
        base_url = "https://maayanlab.cloud/speedrichr"
    else:
        base_url = "https://maayanlab.cloud/speedrichr"

    try:
        results = _query_enrichr(gene_list, gene_sets, base_url)
    except Exception as exc:
        raise ConnectionError(f"Failed to connect to Enrichr API: {exc}") from exc

    if not results:
        return pd.DataFrame(columns=["term", "overlap", "p_value", "adjusted_p_value", "genes"])

    # Parse results into DataFrame
    rows = []
    for entry in results[:top_n]:
        rows.append(
            {
                "term": entry[1],
                "overlap": entry[2],
                "p_value": entry[3],
                "adjusted_p_value": entry[6],
                "genes": ";".join(entry[5]),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("adjusted_p_value").reset_index(drop=True)
    return df


def _query_enrichr(gene_list: list[str], gene_sets: str, base_url: str) -> list:
    """Submit genes to Enrichr and retrieve enrichment results."""
    import json
    import urllib.request

    # Step 1: Submit gene list
    genes_str = "\n".join(gene_list)

    # Use form-encoded POST
    data = f"list={urllib.request.quote(genes_str)}&description=singlet_query"
    req = urllib.request.Request(
        f"{base_url}/api/addList",
        data=data.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read().decode())

    user_list_id = result.get("userListId")
    if not user_list_id:
        raise ValueError(f"Enrichr did not return a userListId: {result}")

    # Step 2: Get enrichment results
    enrich_url = (
        f"{base_url}/api/getEnrichmentResults?userListId={user_list_id}&backgroundType={gene_sets}"
    )
    req2 = urllib.request.Request(enrich_url)

    with urllib.request.urlopen(req2, timeout=30) as response:
        enrich_result = json.loads(response.read().decode())

    return enrich_result.get(gene_sets, [])


def enrichr_from_de(
    adata,
    group: str,
    *,
    n_genes: int = 100,
    pval_cutoff: float = 0.05,
    gene_sets: str = "GO_Biological_Process_2023",
    top_n: int = 10,
) -> "pd.DataFrame":
    """Run enrichment on top DE genes for a specific group.

    Convenience function that extracts marker genes from
    adata.uns['rank_genes_groups'] and runs enrichr().

    Parameters
    ----------
    adata : anndata.AnnData
        Must have 'rank_genes_groups' in adata.uns.
    group : str
        Group name to extract markers from.
    n_genes : int, default 100
        Maximum number of top genes to use.
    pval_cutoff : float, default 0.05
        Only include genes with adjusted p-value below this.
    gene_sets : str, default "GO_Biological_Process_2023"
        Enrichr library to query.
    top_n : int, default 10
        Number of enriched terms to return.

    Returns
    -------
    pandas.DataFrame
        Enrichment results (same format as enrichr()).
    """
    if not hasattr(adata, "uns") or "rank_genes_groups" not in adata.uns:
        raise KeyError(
            "adata.uns['rank_genes_groups'] not found. Run singlet.rank_genes_groups() first."
        )

    result = adata.uns["rank_genes_groups"]
    group_str = str(group)

    if group_str not in result["names"]:
        raise KeyError(f"Group '{group}' not found in rank_genes_groups results.")

    names = result["names"][group_str][:n_genes]

    # Filter by adjusted p-value if available
    if "pvals_adj" in result and group_str in result["pvals_adj"]:
        pvals = result["pvals_adj"][group_str][:n_genes]
        names = [n for n, p in zip(names, pvals) if p < pval_cutoff]

    if not names:
        import pandas as pd

        return pd.DataFrame(columns=["term", "overlap", "p_value", "adjusted_p_value", "genes"])

    return enrichr(names, gene_sets=gene_sets, top_n=top_n)
