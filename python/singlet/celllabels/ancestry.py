# SPDX-License-Identifier: MIT
"""Genetic ancestry per donor from the allele depths the pipeline keeps in restricted storage.

Input per donor: reference/alternate read depths over the pipeline's SNP panel (1000 Genomes phase 3 biallelic SNPs with
AF >= 0.05), aggregated over the donor's cells. These are germline data: they are read from the restricted sidecar
(``donor/donor<k>_coverage.tsv``, or ``donor/snp_ad.1pz`` + ``donor/snp_dp.1pz`` with ``donor/donor_assignments.tsv``) on the
compute node and never written anywhere. The only things that leave this module are a super-population label, its HANCESTRO
id and a proportion per donor.

Model (supervised admixture with fixed allele frequencies, as in ADMIXTURE's projection mode): each donor has ancestry
proportions ``q`` over the five 1000 Genomes super-populations; at site ``s`` the donor's alternate-allele frequency is
``f_s = q . F_s`` (``F`` = per-super-population allele frequencies), genotypes follow Hardy-Weinberg at ``f_s``, and reads
follow a binomial with allele fraction ``eps``, 1/2 or ``1 - eps`` for the three genotypes. RNA depth is very uneven and
allelic expression is not always balanced, so each site's depth is capped at ``DP_CAP`` reads (ad scaled down with it) before
the likelihood is formed: one deeply covered site cannot outvote many shallow ones. ``q`` maximises the likelihood on the
simplex (L-BFGS on softmax logits, analytic gradient).

The call is the super-population with the largest proportion when that proportion is at least ``Q_MIN``; otherwise the donor
is reported as admixed (HANCESTRO:0306). ``AMR`` in 1000 Genomes is itself admixed (European, African and Indigenous American),
so Latin American donors are expected to land on AMR or to be reported as admixed. Labels map to HANCESTRO 2025-10-14 genetic
ancestry categories (children of HANCESTRO:0004 'ancestry category'), not to the self-identified geography-based terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

POPS = ("AFR", "AMR", "EAS", "EUR", "SAS")
HANCESTRO = {
    "AFR": ("HANCESTRO:0010", "African ancestry"),
    "AMR": ("HANCESTRO:0014", "Latin or Admixed American ancestry"),
    "EAS": ("HANCESTRO:0009", "East Asian ancestry"),
    "EUR": ("HANCESTRO:0005", "European ancestry"),
    "SAS": ("HANCESTRO:0006", "South Asian ancestry"),
    "admixed": ("HANCESTRO:0306", "admixed ancestry"),
}
HANCESTRO_VERSION = "releases/2025-10-14"
DP_CAP = 8
EPS = 0.02
Q_MIN = 0.8
MIN_SITES = 300  # sites with any depth needed before a call is made
MID_AF_DP = 20  # depth at which a site's allele fraction is informative about pooling
MAX_MID_AF = 0.15  # single donors measured 0.04-0.08 (docs/cell_annotation.md); above it the donor is treated as a pool
SCREEN_SITES = 20_000  # multi-start screening on at most this many sites
BOOT_SITES = 20_000  # bootstrap replicates of at most this many sites
ALT_READ_RANGE = (
    0.10,
    0.60,
)  # plausible alternate-read fraction over the panel for one person (observed 0.23-0.31)
MIN_MATCH_FRAC = (
    0.5  # a donor whose sites mostly miss the table is on another build or species: no call
)
F_CLIP = 1e-3


@dataclass
class AncestryCall:
    """One donor's result. ``group`` is a key of :data:`HANCESTRO` or None (not enough data)."""

    group: Optional[str]
    prob: Optional[
        float
    ]  # proportion of the called super-population (for 'admixed': the largest proportion)
    q: Dict[str, float]
    n_sites: int
    stability: Optional[float] = None  # fraction of site bootstraps that give the same group
    mid_af: Optional[float] = (
        None  # fraction of deep sites with an allele fraction far from 0, 1/2 and 1 (see infer_ancestry)
    )
    reason: Optional[str] = None  # why there is no call, when there is none despite enough sites

    @property
    def hancestro(self) -> Optional[str]:
        return HANCESTRO[self.group][0] if self.group else None

    @property
    def label(self) -> Optional[str]:
        return HANCESTRO[self.group][1] if self.group else None


def het_fraction(ad: np.ndarray, dp: np.ndarray, min_dp: int = 10) -> float:
    """Alternate-read fraction expected at a heterozygous site for this donor. RNA reads carry reference-mapping bias (the
    alternate allele is under-counted by a few points), so it is estimated as the median alternate fraction over deep sites
    that show both alleles, clipped to [0.35, 0.5]; 0.5 when there are fewer than 200 such sites."""
    m = (dp >= min_dp) & (ad > 0) & (ad < dp)
    if m.sum() < 200:
        return 0.5
    return float(np.clip(np.median(ad[m] / dp[m]), 0.35, 0.5))


def _site_terms(ad: np.ndarray, dp: np.ndarray, h: float = 0.5):
    """Read likelihoods of genotypes 0/1/2 per site, with depth capped at DP_CAP (scaled to the max, then rescaled to sum 1).
    ``h``: alternate-read fraction at a heterozygous site."""
    dp = dp.astype(np.float64)
    ad = np.minimum(ad.astype(np.float64), dp)
    scale = np.minimum(1.0, DP_CAP / np.maximum(dp, 1))
    a, r = ad * scale, (dp - ad) * scale
    ll = np.stack(
        [
            a * np.log(EPS) + r * np.log1p(-EPS),
            a * np.log(h) + r * np.log1p(-h),
            a * np.log1p(-EPS) + r * np.log(EPS),
        ],
        1,
    )
    ll -= ll.max(1, keepdims=True)
    return np.exp(ll)


def _negll(theta: np.ndarray, F: np.ndarray, L: np.ndarray):
    e = np.exp(theta - theta.max())
    q = e / e.sum()
    f = np.clip(F @ q, F_CLIP, 1 - F_CLIP)
    p0, p1, p2 = (1 - f) ** 2, 2 * f * (1 - f), f**2
    site = L[:, 0] * p0 + L[:, 1] * p1 + L[:, 2] * p2
    dsite_df = L[:, 0] * (-2 * (1 - f)) + L[:, 1] * (2 - 4 * f) + L[:, 2] * (2 * f)
    g_q = -(F * (dsite_df / site)[:, None]).sum(0)  # d(-ll)/dq
    g_theta = q * (g_q - q @ g_q)  # softmax Jacobian
    return -np.log(site).sum(), g_theta


def fit_q(
    F: np.ndarray, L: np.ndarray, start: Optional[np.ndarray] = None, seed: int = 0
) -> np.ndarray:
    """Maximum-likelihood ``q``. Without ``start`` (a proportion vector) six starts (uniform and near each vertex) are
    screened; above ``SCREEN_SITES`` sites the screening runs on a random subsample and only the best start is refined on
    every site (the validation found the same optimum at a fraction of the cost)."""
    from scipy.optimize import minimize

    def run(x0, Fs, Ls):
        return minimize(
            _negll, x0, args=(Fs, Ls), jac=True, method="L-BFGS-B", options=dict(maxiter=200)
        )

    if start is not None:
        x0 = np.log(np.clip(start, 1e-6, 1))
    else:
        K = F.shape[1]
        starts = [np.zeros(K)] + [np.eye(K)[k] * 2.0 for k in range(K)]
        if len(F) > SCREEN_SITES:
            i = np.random.default_rng(seed).choice(len(F), SCREEN_SITES, replace=False)
            Fs, Ls = F[i], L[i]
        else:
            Fs, Ls = F, L
        x0 = min((run(s0, Fs, Ls) for s0 in starts), key=lambda r: r.fun).x
        if len(F) <= SCREEN_SITES:
            e = np.exp(x0 - x0.max())
            return e / e.sum()
    r = run(x0, F, L)
    e = np.exp(r.x - r.x.max())
    return e / e.sum()


def mid_allele_fraction(ad: np.ndarray, dp: np.ndarray, min_dp: int = MID_AF_DP) -> Optional[float]:
    """Among sites with at least ``min_dp`` reads, the fraction whose alternate-allele fraction lies in (0.15, 0.35) or
    (0.65, 0.85). One person's genotypes put allele fractions near 0, 1/2 or 1 (allele-specific expression spreads them
    somewhat); cells pooled from several people put many in between. None when fewer than 100 deep sites."""
    deep = dp >= min_dp
    if deep.sum() < 100:
        return None
    f = ad[deep] / dp[deep]
    return float((((f > 0.15) & (f < 0.35)) | ((f > 0.65) & (f < 0.85))).mean())


def _group(q: np.ndarray, q_min: float) -> str:
    k = int(np.argmax(q))
    return POPS[k] if q[k] >= q_min else "admixed"


def infer_ancestry(
    ad: np.ndarray,
    dp: np.ndarray,
    freqs: np.ndarray,
    *,
    q_min: float = Q_MIN,
    min_sites: int = MIN_SITES,
    n_boot: int = 20,
    seed: int = 0,
) -> AncestryCall:
    """Ancestry of one donor. ``ad``/``dp``: alternate and total depth per site; ``freqs``: sites x 5 alternate-allele
    frequencies in :data:`POPS` order, for the same sites and the same alternate allele."""
    ad = np.asarray(ad)
    dp = np.asarray(dp)
    F = np.asarray(freqs, np.float64)
    ok = (dp > 0) & np.isfinite(F).all(1)
    n_sites = int(ok.sum())
    if n_sites < min_sites:
        return AncestryCall(None, None, {}, n_sites)
    alt = float(ad[ok].sum() / max(dp[ok].sum(), 1))
    if not ALT_READ_RANGE[0] <= alt <= ALT_READ_RANGE[1]:
        # a real person's reads over the panel are ~30% alternate; far outside that the depth table is defective
        # (a study with ad = 0 everywhere was found in validation): no call rather than a wrong one
        return AncestryCall(None, None, {}, n_sites, None, None, "alt read fraction %.3f" % alt)
    mid = mid_allele_fraction(ad[ok], dp[ok])
    if mid is not None and mid > MAX_MID_AF:
        return AncestryCall(None, None, {}, n_sites, None, mid, "mid allele fraction %.3f" % mid)
    L = _site_terms(ad[ok], dp[ok], het_fraction(ad[ok], dp[ok]))
    F = F[ok]
    q = fit_q(F, L)
    grp = _group(q, q_min)
    stab = None
    if n_boot:
        rng = np.random.default_rng(seed)
        same = 0
        m = min(n_sites, BOOT_SITES)  # stability saturates long before this many sites
        for _ in range(n_boot):
            i = rng.integers(0, n_sites, m)
            same += _group(fit_q(F[i], L[i], start=q), q_min) == grp
        stab = same / n_boot
    return AncestryCall(
        grp, float(q.max()), {p: float(v) for p, v in zip(POPS, q)}, n_sites, stab, mid
    )


class AFIndex:
    """The allele-frequency table indexed by (chrom, pos) for fast joins: built once per process, reused for every donor."""

    def __init__(self, af_table):
        chrom = af_table["chrom"].astype(str).str.replace("^chr", "", regex=True).to_numpy()
        self.key = _site_key(chrom, af_table["pos"].to_numpy(np.int64))
        order = np.argsort(self.key, kind="stable")
        self.key = self.key[order]
        self.ref = af_table["ref"].to_numpy().astype("U1")[order]
        self.alt = af_table["alt"].to_numpy().astype("U1")[order]
        self.F = af_table[list(POPS)].to_numpy(np.float32)[order]

    def align(self, sites):
        """(ad, dp, freqs) of ``sites`` (chrom, pos, ref, alt, ad, dp) joined on position with matching alleles; swapped
        ref/alt pairs are flipped (ad := dp - ad)."""
        chrom = sites["chrom"].astype(str).str.replace("^chr", "", regex=True).to_numpy()
        k = _site_key(chrom, sites["pos"].to_numpy(np.int64))
        i = np.clip(np.searchsorted(self.key, k), 0, len(self.key) - 1)
        hit = self.key[i] == k
        ref = sites["ref"].to_numpy().astype("U1")
        alt = sites["alt"].to_numpy().astype("U1")
        same = hit & (self.ref[i] == ref) & (self.alt[i] == alt)
        swap = hit & (self.ref[i] == alt) & (self.alt[i] == ref)
        use = same | swap
        ad = sites["ad"].to_numpy(np.int64)
        dp = sites["dp"].to_numpy(np.int64)
        ad = np.where(swap, dp - ad, ad)
        return ad[use], dp[use], self.F[i[use]].astype(np.float64)


_CHROM_CODE = {**{str(i): i for i in range(1, 23)}, "X": 23, "Y": 24, "MT": 25, "M": 25}


def _site_key(chrom: np.ndarray, pos: np.ndarray) -> np.ndarray:
    code = np.array([_CHROM_CODE.get(c, 0) for c in chrom], np.int64)
    return code * 1_000_000_000 + pos


def align_sites(sites, af_table):
    """Join a donor's sites to the allele-frequency table on (chrom, pos, ref, alt); a swapped ref/alt pair is flipped.

    ``sites``: DataFrame with chrom, pos, ref, alt, ad, dp. ``af_table``: DataFrame with chrom, pos, ref, alt and one column per
    entry of :data:`POPS`. Returns (ad, dp, freqs) for the joined sites."""
    import pandas as pd

    s = sites.copy()
    s["chrom"] = s["chrom"].astype(str).str.replace("^chr", "", regex=True)
    a = af_table.copy()
    a["chrom"] = a["chrom"].astype(str).str.replace("^chr", "", regex=True)
    keys = ["chrom", "pos", "ref", "alt"]
    m = s.merge(a, on=keys, how="inner")
    sw = s.rename(columns={"ref": "alt", "alt": "ref"}).merge(a, on=keys, how="inner")
    if len(sw):
        sw["ad"] = sw["dp"] - sw["ad"]
    j = pd.concat([m, sw], ignore_index=True)
    return j["ad"].to_numpy(), j["dp"].to_numpy(), j[list(POPS)].to_numpy(np.float64)
