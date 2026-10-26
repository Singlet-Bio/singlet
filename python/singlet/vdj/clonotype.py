# SPDX-License-Identifier: MIT
"""Per-cell chains, pairing and clonotypes from annotated contigs (``assemble.CONTIG_COLUMNS``).

* **Cell chains**: the kept productive contigs of a barcode, at most two per chain, by UMIs (a second alpha or light
  chain is common; more is usually a doublet, flagged ``excess_chains``).
* **Pairing**: a T cell is paired with a productive TRA and TRB (or TRG and TRD), a B cell with IGH and IGK or IGL.
* **Clonotypes**: T cells with the same set of (chain, V gene, J gene, junction nt) share an exact clonotype (Cell
  Ranger's exact subclonotype for TCRs). B cells are grouped into lineages, as Cell Ranger's enclone does: same heavy
  V and J genes, same heavy junction length, heavy junctions within 15% nucleotide Hamming distance, and the same
  light V and J genes when both cells have a light chain. Clonotype ids are ``clonotype<k>`` by decreasing size.
"""

from __future__ import annotations

import numpy as np

TCR = ("TRA", "TRB", "TRG", "TRD")
BCR = ("IGH", "IGK", "IGL")


def cell_table(contigs, max_per_chain: int = 2):
    """One row per barcode with productive kept contigs: per chain the top junctions and genes, pairing flags."""
    import pandas as pd

    d = contigs[contigs["kept"] & contigs["productive"]].sort_values(["barcode", "chain", "umis"],
                                                                      ascending=[True, True, False])
    d = d.assign(rank=d.groupby(["barcode", "chain"]).cumcount())
    excess = d[d["rank"] >= max_per_chain].groupby("barcode").size()
    d = d[d["rank"] < max_per_chain]
    rows = {}
    for r in d.itertuples(index=False):
        c = rows.setdefault(r.barcode, {"barcode": r.barcode})
        tag = r.chain if r.rank == 0 else f"{r.chain}_2"
        c[f"{tag}_cdr3"] = r.cdr3
        c[f"{tag}_cdr3_nt"] = r.cdr3_nt
        c[f"{tag}_v"] = r.v_gene
        c[f"{tag}_j"] = r.j_gene
        c[f"{tag}_c"] = r.c_gene
        c[f"{tag}_umis"] = r.umis
    out = pd.DataFrame(list(rows.values()))
    if out.empty:
        return out
    for ch in TCR + BCR:
        if f"{ch}_cdr3" not in out:
            out[f"{ch}_cdr3"] = np.nan
    has = {ch: out[f"{ch}_cdr3"].notna() for ch in TCR + BCR}
    out["receptor"] = np.where(has["IGH"] | has["IGK"] | has["IGL"],
                               np.where(has["TRA"] | has["TRB"], "mixed", "BCR"), "TCR")
    out["paired"] = ((has["TRA"] & has["TRB"]) | (has["TRG"] & has["TRD"]) | (has["IGH"] & (has["IGK"] | has["IGL"])))
    out["excess_chains"] = out["barcode"].map(excess).fillna(0).astype(int)
    return out


def _key(row, chains):
    parts = []
    for ch in chains:
        for tag in (ch, f"{ch}_2"):
            nt = row.get(f"{tag}_cdr3_nt")
            if isinstance(nt, str):
                parts.append(f"{ch}:{row.get(f'{tag}_v')}:{row.get(f'{tag}_j')}:{nt}")
    return "|".join(sorted(parts))


def assign_clonotypes(cells):
    """Adds ``clonotype_id`` and ``clonotype_size`` to a :func:`cell_table` frame."""
    import pandas as pd

    if cells.empty:
        return cells.assign(clonotype_id=pd.Series(dtype=str), clonotype_size=pd.Series(dtype=int))
    rows = cells.to_dict("records")
    key = []
    for r in rows:
        key.append(("T", _key(r, TCR)) if r["receptor"] == "TCR" else ("B", None))
    # exact keys for T cells
    group = np.full(len(rows), -1)
    seen = {}
    for i, (kind, k) in enumerate(key):
        if kind == "T":
            group[i] = seen.setdefault(k, len(seen))
    # B lineages: union-find over cells sharing heavy V/J/length bucket
    b = [i for i, (kind, _) in enumerate(key) if kind == "B"]
    parent = {i: i for i in b}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    buckets = {}
    for i in b:
        r = rows[i]
        h = r.get("IGH_cdr3_nt")
        if isinstance(h, str):
            buckets.setdefault((r.get("IGH_v"), r.get("IGH_j"), len(h)), []).append(i)
        else:
            lk = [r.get(f"{c}_cdr3_nt") for c in ("IGK", "IGL") if isinstance(r.get(f"{c}_cdr3_nt"), str)]
            buckets.setdefault(("light",) + tuple(sorted(lk)), []).append(i)

    def light(r):
        for c in ("IGK", "IGL"):
            if isinstance(r.get(f"{c}_cdr3_nt"), str):
                return (c, r.get(f"{c}_v"), r.get(f"{c}_j"))
        return None

    for bk, members in buckets.items():
        if bk[0] == "light":
            for i in members[1:]:
                parent[find(i)] = find(members[0])
            continue
        L = bk[2]
        seqs = np.array([np.frombuffer(rows[i]["IGH_cdr3_nt"].encode(), np.uint8) for i in members])
        lights = [light(rows[i]) for i in members]
        for x in range(len(members)):
            dist = (seqs[x + 1:] != seqs[x]).sum(axis=1)
            for y in np.flatnonzero(dist <= 0.15 * L) + x + 1:
                lx, ly = lights[x], lights[y]
                if lx is None or ly is None or lx == ly:
                    parent[find(members[y])] = find(members[x])
    off = len(seen)
    roots = {}
    for i in b:
        group[i] = off + roots.setdefault(find(i), len(roots))
    sizes = np.bincount(group)
    order = np.argsort(-sizes, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    out = cells.copy()
    out["clonotype_id"] = ["clonotype%d" % (rank[g] + 1) for g in group]
    out["clonotype_size"] = sizes[group]
    return out
