"""Donor genotype comparison for the donor-snps benchmark. PRIVATE: this module reads genotypes on Clipper and writes
only counts (a 3x3 table of genotype classes per donor, plus site counts). No per-site output is ever written.

Truth: pseudobulk allele counts of an independent replicate library of the same donors (a different 10x lane), pooled
over the cells demuxlet assigned to the donor (`kang2018` truth tables), read with cellsnp-lite on that lane's tagged BAM
at the shared 1000G panel. A site's truth genotype is called from the counts with a fixed binomial model
(error 0.01, flat prior) and is *confident* when its posterior is >= 0.99. The model is fixed here, not tuned.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread

ERR = 0.01
POST = 0.99
GT_NAMES = ["0/0", "0/1", "1/1"]


def genotype_from_counts(ad: np.ndarray, dp: np.ndarray, err: float = ERR, post: float = POST) -> np.ndarray:
    """int8 array: -1 not confident (or no data), 0/1/2 = number of alt alleles."""
    ad = np.asarray(ad, float); dp = np.asarray(dp, float); rf = dp - ad
    ll = np.stack([rf * np.log(1 - err) + ad * np.log(err),
                   dp * np.log(0.5),
                   ad * np.log(1 - err) + rf * np.log(err)], axis=1)
    ll -= ll.max(axis=1, keepdims=True)
    p = np.exp(ll); p /= p.sum(axis=1, keepdims=True)
    g = p.argmax(axis=1).astype(np.int8)
    g[(p.max(axis=1) < post) | (dp < 1)] = -1
    return g


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_cellsnp(d: str | Path):
    """cellSNP.base.vcf.gz + AD/DP mtx + samples -> (sites DataFrame[chrom,pos,ref,alt], AD csr, DP csr, barcodes)."""
    d = Path(d)
    rows = []
    with _open(d / "cellSNP.base.vcf.gz") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 5)
            rows.append((f[0], int(f[1]), f[3], f[4]))
    sites = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt"])
    ad = mmread(d / "cellSNP.tag.AD.mtx").tocsr()
    dp = mmread(d / "cellSNP.tag.DP.mtx").tocsr()
    bcs = [l.strip() for l in open(d / "cellSNP.samples.tsv")]
    return sites, ad, dp, bcs


def norm_bc(b: str) -> str:
    return str(b).split("-")[0]


def pseudobulk(ad, dp, bcs, labels: dict[str, str]):
    """Sum AD / DP over the cells of each label. Returns {label: (ad_vec, dp_vec)}."""
    lab = np.array([labels.get(norm_bc(b)) for b in bcs], dtype=object)
    out = {}
    for d in sorted({x for x in lab if isinstance(x, str) and x != "<NA>"}):
        idx = np.where(lab == d)[0]
        out[d] = (np.asarray(ad[:, idx].sum(axis=1)).ravel(), np.asarray(dp[:, idx].sum(axis=1)).ravel())
    return out


def site_key(chrom, pos, ref, alt):
    return f"{chrom}:{pos}:{ref}:{alt}"


def parse_gt(s: str) -> int:
    m = re.match(r"^(\d)[/|](\d)", s)
    if not m:
        return -1
    a, b = int(m.group(1)), int(m.group(2))
    return -1 if (a > 1 or b > 1) else a + b


def read_donor_vcf(path: str | Path) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """VCF with one or more sample columns -> (site keys, [GT per sample], [DP per sample]). GT in {-1,0,1,2}.
    Works for Singlet (GT:AF:AD:DP), vireo (GT:AD:DP:...), souporcell (GT:AO:RO:T:E:GO:GN)."""
    keys, gts, dps = [], [], []
    with _open(path) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            fmt = f[8].split(":")
            igt = fmt.index("GT")
            idp = fmt.index("DP") if "DP" in fmt else None
            iao = fmt.index("AO") if "AO" in fmt else None
            iro = fmt.index("RO") if "RO" in fmt else None
            g, d = [], []
            for s in f[9:]:
                p = s.split(":")
                g.append(parse_gt(p[igt]))
                if idp is not None:
                    d.append(int(float(p[idp])) if p[idp] not in (".", "") else 0)
                elif iao is not None:
                    d.append(int(float(p[iao])) + int(float(p[iro])))
                else:
                    d.append(0)
            keys.append(site_key(f[0], f[1], f[3], f[4])); gts.append(g); dps.append(d)
    G = np.array(gts, dtype=np.int8).reshape(len(keys), -1)
    D = np.array(dps, dtype=np.int64).reshape(len(keys), -1)
    return keys, [G[:, j] for j in range(G.shape[1])], [D[:, j] for j in range(D.shape[1])]


def confusion(truth_gt: np.ndarray, tool_gt: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """3x3 counts (rows truth 0/0,0/1,1/1; columns tool) over sites where the truth is confident and the tool made a call."""
    ok = (truth_gt >= 0) & (tool_gt >= 0)
    if mask is not None:
        ok &= mask
    c = np.zeros((3, 3), dtype=np.int64)
    for t in range(3):
        for u in range(3):
            c[t, u] = int(((truth_gt == t) & (tool_gt == u) & ok).sum())
    return c
