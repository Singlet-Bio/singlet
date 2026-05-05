# SPDX-License-Identifier: GPL-2.0-or-later
"""
Cycle 52c smoke tests for the 23 new wrappers shipped in cycles 27-50 (caught up in
cycles 52a/52b).  One @pytest.mark.smoke test per feature.

Each test:
  1.  Creates a tiny synthetic input  (≤200 cells × 100 genes).
  2.  Imports the wrapper and calls it with seed=0xC0FFEE, stream=None.
  3.  Asserts the result has expected keys / shapes.
  4.  Skips gracefully when no GPU is present.

Build: SKIPPED when nvcc / cupy absent.
Gate : pending GPU dispatch.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# GPU guard
# ---------------------------------------------------------------------------


def gpu_available() -> bool:
    try:
        import cupy
        import cupy.cuda

        cupy.cuda.Device(0).use()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tiny synthetic helpers
# ---------------------------------------------------------------------------

SEED = 0xC0FFEE
N_CELLS = 200
N_GENES = 100
N_ADT = 20
N_MOTIFS = 15
N_SNPS = 50
N_FRAGS = 300
RNG = np.random.default_rng(SEED)


def _tiny_expression(n_cells: int = N_CELLS, n_genes: int = N_GENES) -> sp.csc_matrix:
    """Sparse NB-like count matrix (genes × cells) as scipy CSC."""
    rng = np.random.default_rng(SEED)
    data = rng.negative_binomial(2, 0.5, size=n_cells * n_genes).reshape(n_genes, n_cells)
    return sp.csc_matrix(data.astype(np.float32))


def _tiny_expression_cells_x_genes(n_cells: int = N_CELLS, n_genes: int = N_GENES) -> sp.csr_matrix:
    """Sparse count matrix (cells × genes) as scipy CSR."""
    return _tiny_expression(n_cells, n_genes).T.tocsr()


def _tiny_snp(n_cells: int = N_CELLS, n_snps: int = N_SNPS):
    """Pair of (n_snps × n_cells) uint32 CSC matrices for AD and DP."""
    rng = np.random.default_rng(SEED + 1)
    dp = rng.integers(5, 40, size=(n_snps, n_cells)).astype(np.uint32)
    ad = rng.integers(0, dp + 1, size=(n_snps, n_cells)).astype(np.uint32)
    return sp.csc_matrix(ad), sp.csc_matrix(dp)


def _tiny_adt(n_cells: int = N_CELLS, n_adt: int = N_ADT) -> sp.csc_matrix:
    """Sparse ADT count matrix (n_adt × n_cells)."""
    rng = np.random.default_rng(SEED + 2)
    data = rng.negative_binomial(3, 0.4, size=(n_adt, n_cells)).astype(np.float32)
    return sp.csc_matrix(data)


def _tiny_fragments(n_cells: int = N_CELLS, n_frags: int = N_FRAGS) -> sp.csc_matrix:
    """Sparse ATAC fragment-count matrix (n_frags × n_cells)."""
    rng = np.random.default_rng(SEED + 3)
    rows = rng.integers(0, n_frags, size=n_frags * 3)
    cols = rng.integers(0, n_cells, size=n_frags * 3)
    vals = np.ones(n_frags * 3, dtype=np.float32)
    return sp.csc_matrix((vals, (rows, cols)), shape=(n_frags, n_cells))


def _tiny_spatial_coords(n_spots: int = N_CELLS) -> np.ndarray:
    """Float32 (n_spots × 2) spatial coordinate array."""
    rng = np.random.default_rng(SEED + 4)
    return rng.random((n_spots, 2), dtype=np.float32) * 1000.0


def _tiny_pca(n_cells: int = N_CELLS, n_pcs: int = 20) -> np.ndarray:
    rng = np.random.default_rng(SEED + 5)
    return rng.standard_normal((n_cells, n_pcs)).astype(np.float32)


def _tiny_transition_matrix(n_cells: int = N_CELLS) -> sp.csr_matrix:
    """Row-stochastic transition matrix."""
    rng = np.random.default_rng(SEED + 6)
    data = rng.random((n_cells, n_cells)).astype(np.float32)
    row_sums = data.sum(axis=1, keepdims=True)
    data /= row_sums
    return sp.csr_matrix(data)


# ---------------------------------------------------------------------------
# Tier 1: cospar
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_cospar_smoke():
    """CoSpar fate mapping: result has fate_bias and potency_score."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.fate.cospar import run_from_csc

    mat = _tiny_expression()
    clone_ids = RNG.integers(-1, 5, size=N_CELLS).astype(np.int32)
    time_labels = RNG.integers(0, 3, size=N_CELLS).astype(np.int32)
    embedding = _tiny_pca()

    result = run_from_csc(
        mat,
        clone_ids,
        time_labels,
        embedding,
        k_neighbors=10,
        max_iters=5,
        n_fate_bias_steps=2,
        stream=None,
        seed=SEED,
    )
    assert hasattr(result, "fate_bias"), "CoSpar result missing .fate_bias"
    assert hasattr(result, "potency_score"), "CoSpar result missing .potency_score"
    assert result.fate_bias.shape[0] == N_CELLS, (
        f"fate_bias rows {result.fate_bias.shape[0]} != N_CELLS {N_CELLS}"
    )
    assert result.potency_score.shape == (N_CELLS,), (
        f"potency_score shape {result.potency_score.shape} != ({N_CELLS},)"
    )


# ---------------------------------------------------------------------------
# Tier 1: cellrank2
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_cellrank2_smoke():
    """CellRank2 absorption probabilities: result has absorption_prob."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.fate.cellrank2 import compute_absorption_probabilities

    T = _tiny_transition_matrix()
    n_terminals = 5
    terminal_idx = np.arange(n_terminals, dtype=np.int32)

    result = compute_absorption_probabilities(
        T,
        terminal_idx,
        gmres_m=10,
        gmres_max_restarts=3,
        stream=None,
        seed=SEED,
    )
    assert hasattr(result, "absorption_prob"), "CellRank2 result missing .absorption_prob"
    assert result.absorption_prob.shape == (N_CELLS, n_terminals), (
        f"absorption_prob shape {result.absorption_prob.shape} != ({N_CELLS}, {n_terminals})"
    )


# ---------------------------------------------------------------------------
# Tier 1: palantir
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_palantir_smoke():
    """Palantir: result has pseudotime and entropy."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.fate.palantir import run_from_embedding

    embedding = _tiny_pca()
    terminal_idx = np.array([0, 5, 10], dtype=np.int32)
    start_idx = np.int32(1)

    result = run_from_embedding(
        embedding,
        start_cell=start_idx,
        terminal_states=terminal_idx,
        k_neighbors=10,
        stream=None,
        seed=SEED,
    )
    assert hasattr(result, "pseudotime"), "Palantir result missing .pseudotime"
    assert hasattr(result, "entropy"), "Palantir result missing .entropy"
    assert result.pseudotime.shape == (N_CELLS,), (
        f"pseudotime shape {result.pseudotime.shape} != ({N_CELLS},)"
    )


# ---------------------------------------------------------------------------
# Tier 1: cellchat
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_cellchat_smoke():
    """CellChat: result has interaction_prob with (n_types × n_types × n_lr) shape."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.comm.cellchat import run_from_csc

    mat = _tiny_expression()
    n_types = 4
    cell_labels = RNG.integers(0, n_types, size=N_CELLS).astype(np.int32)
    n_lr = 10
    ligand_idx = RNG.integers(0, N_GENES, size=n_lr).astype(np.int32)
    receptor_idx = RNG.integers(0, N_GENES, size=n_lr).astype(np.int32)

    result = run_from_csc(
        mat,
        cell_labels,
        ligand_idx,
        receptor_idx,
        stream=None,
        seed=SEED,
    )
    assert hasattr(result, "interaction_prob"), "CellChat result missing .interaction_prob"
    iprob = np.asarray(result.interaction_prob)
    assert iprob.shape == (n_types, n_types, n_lr), (
        f"interaction_prob shape {iprob.shape} != ({n_types},{n_types},{n_lr})"
    )


# ---------------------------------------------------------------------------
# Tier 1: hdwgcna
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_hdwgcna_smoke():
    """hdWGCNA: result has module_eigengenes."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.network.hdwgcna import run_from_csc

    mat = _tiny_expression()

    result = run_from_csc(
        mat,
        k_neighbors=10,
        n_soft_power_max=5,
        stream=None,
        seed=SEED,
    )
    assert hasattr(result, "module_eigengenes"), "hdWGCNA result missing .module_eigengenes"
    me = np.asarray(result.module_eigengenes)
    assert me.shape[0] == N_CELLS, f"module_eigengenes rows {me.shape[0]} != N_CELLS {N_CELLS}"


# ---------------------------------------------------------------------------
# Tier 1: milo
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_milo_smoke():
    """Milo: result has nhoods_lfc and nhoods_fdr."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.abundance.milo import run_from_embedding

    embedding = _tiny_pca()
    n_conditions = 2
    condition_labels = RNG.integers(0, n_conditions, size=N_CELLS).astype(np.int32)

    result = run_from_embedding(
        embedding,
        condition_labels,
        k_neighbors=10,
        stream=None,
        seed=SEED,
    )
    assert hasattr(result, "nhoods_lfc"), "Milo result missing .nhoods_lfc"
    assert hasattr(result, "nhoods_fdr"), "Milo result missing .nhoods_fdr"
    assert len(result.nhoods_lfc) > 0, "Milo returned empty nhoods_lfc"


# ---------------------------------------------------------------------------
# Tier 1: scdrs
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_scdrs_smoke():
    """scDRS: result has cell_score with shape (N_CELLS,)."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.disease.scdrs import run_from_csc

    mat = _tiny_expression()
    n_disease_genes = 20
    gene_weights = RNG.random(n_disease_genes).astype(np.float32)
    gene_indices = RNG.choice(N_GENES, size=n_disease_genes, replace=False).astype(np.int32)

    result = run_from_csc(
        mat,
        gene_indices,
        gene_weights,
        n_ctrl=5,
        stream=None,
        seed=SEED,
    )
    assert hasattr(result, "cell_score"), "scDRS result missing .cell_score"
    assert result.cell_score.shape == (N_CELLS,), (
        f"cell_score shape {result.cell_score.shape} != ({N_CELLS},)"
    )


# ---------------------------------------------------------------------------
# Tier 2: granie
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_granie_smoke():
    """GrANIE: result has gene_peak_links."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.grn.granie import run_from_csc

    expr_mat = _tiny_expression()
    peak_mat = _tiny_fragments()

    result = run_from_csc(
        expr_mat,
        peak_mat,
        stream=None,
        seed=SEED,
    )
    assert hasattr(result, "gene_peak_links"), "GrANIE result missing .gene_peak_links"
    links = np.asarray(result.gene_peak_links)
    assert links.ndim == 2, f"gene_peak_links must be 2-D, got shape {links.shape}"


# ---------------------------------------------------------------------------
# Tier 2: nebula
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_nebula_smoke():
    """NEBULA eQTL: result dict has beta, se, p_value keys."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.eqtl.nebula import run

    expr_mat = _tiny_expression()  # n_genes × n_cells
    ad_mat, dp_mat = _tiny_snp()  # n_snps × n_cells
    n_donors = 5
    donor_labels = RNG.integers(0, n_donors, size=N_CELLS).astype(np.int32)

    result = run(
        expr_mat,
        ad_mat,
        dp_mat,
        donor_labels,
        stream=None,
        seed=SEED,
    )
    for key in ("beta", "se", "p_value"):
        assert key in result, f"NEBULA result missing key '{key}'"
    assert "n_snps" in result, "NEBULA result missing 'n_snps'"
    assert "n_genes" in result, "NEBULA result missing 'n_genes'"


# ---------------------------------------------------------------------------
# Tier 2: daesc
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_daesc_smoke():
    """DAESC ASE: result dict has beta, se, phi, p_value."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.ase.daesc import run

    ad_mat, dp_mat = _tiny_snp()
    n_cell_types = 3
    cell_type_labels = RNG.integers(0, n_cell_types, size=N_CELLS).astype(np.int32)

    result = run(
        ad_mat,
        dp_mat,
        cell_type_labels,
        stream=None,
        seed=SEED,
    )
    for key in ("beta", "se", "phi", "p_value"):
        assert key in result, f"DAESC result missing key '{key}'"
    assert result["n_snps"] == N_SNPS, f"DAESC n_snps {result['n_snps']} != N_SNPS {N_SNPS}"


# ---------------------------------------------------------------------------
# Tier 2: numbat
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_numbat_smoke():
    """Numbat CNA: result dict has cna_states and clone_labels."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.cna.numbat import detect

    expr_mat = _tiny_expression()
    ad_mat, dp_mat = _tiny_snp()
    chrom_labels = RNG.integers(1, 4, size=N_SNPS).astype(np.int32)

    result = detect(
        expr_mat,
        ad_mat,
        dp_mat,
        chrom_labels,
        stream=None,
        seed=SEED,
    )
    assert "cna_states" in result, "Numbat result missing 'cna_states'"
    assert "clone_labels" in result, "Numbat result missing 'clone_labels'"
    assert "n_clones" in result, "Numbat result missing 'n_clones'"


# ---------------------------------------------------------------------------
# Tier 2: monopogen
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_monopogen_smoke():
    """Monopogen variant calling: result dict has germline_calls."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.variants.monopogen import call

    ad_mat, dp_mat = _tiny_snp()
    chrom_labels = RNG.integers(1, 4, size=N_SNPS).astype(np.int32)

    result = call(
        ad_mat,
        dp_mat,
        chrom_labels,
        stream=None,
        seed=SEED,
    )
    assert "germline_calls" in result, "Monopogen result missing 'germline_calls'"
    assert "germline_genotypes" in result, "Monopogen result missing 'germline_genotypes'"
    assert "n_snps" in result, "Monopogen result missing 'n_snps'"
    assert result["n_snps"] == N_SNPS, f"n_snps {result['n_snps']} != N_SNPS {N_SNPS}"


# ---------------------------------------------------------------------------
# Tier 2: chromvar
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_chromvar_smoke():
    """chromVAR: result dict has deviation and variability."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.atac.chromvar import compute

    accessibility = _tiny_fragments()  # n_frags × n_cells (peaks × cells)
    n_peaks = N_FRAGS
    motif_in_peak = sp.csc_matrix(RNG.integers(0, 2, size=(N_MOTIFS, n_peaks)).astype(np.float32))
    peak_gc = RNG.random(n_peaks).astype(np.float32)
    peak_access = np.asarray(accessibility.mean(axis=1)).ravel().astype(np.float32)

    result = compute(
        accessibility,
        motif_in_peak,
        peak_gc,
        peak_access,
        n_background=5,
        stream=None,
        seed=SEED,
    )
    assert "deviation" in result, "chromVAR result missing 'deviation'"
    assert "variability" in result, "chromVAR result missing 'variability'"
    assert "n_motifs" in result, "chromVAR result missing 'n_motifs'"
    assert result["n_motifs"] == N_MOTIFS, f"n_motifs {result['n_motifs']} != N_MOTIFS {N_MOTIFS}"


# ---------------------------------------------------------------------------
# Tier 3: flash_deconv
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_flash_deconv_smoke():
    """FlashDeconv spatial deconvolution: result has abundance and uncertainty."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.spatial.flash_deconv import run_flash_deconv

    n_spots = 80
    n_ref_cells = 60
    n_types = 4
    spatial_mat = _tiny_expression(n_spots, N_GENES)
    ref_mat = _tiny_expression(n_ref_cells, N_GENES)
    cell_type_labels = RNG.integers(0, n_types, size=n_ref_cells).astype(np.int32)
    coords = _tiny_spatial_coords(n_spots)

    result = run_flash_deconv(
        spatial_mat,
        ref_mat,
        cell_type_labels,
        coords=coords,
        n_iter=5,
        tol=1e-3,
        stream=None,
        seed=SEED,
    )
    assert hasattr(result, "abundance_view"), "FlashDeconv result missing .abundance_view"
    assert hasattr(result, "uncertainty_view"), "FlashDeconv result missing .uncertainty_view"
    assert hasattr(result, "n_admm_iters_used"), "FlashDeconv result missing .n_admm_iters_used"


# ---------------------------------------------------------------------------
# Tier 3: stagate (fit + predict)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_stagate_smoke():
    """STAGATE spatial embedding: fit → embedding shape (n_spots, d_embed)."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.spatial.stagate import run_stagate

    n_spots = 80
    mat = _tiny_expression(n_spots, N_GENES)
    coords = _tiny_spatial_coords(n_spots)

    result = run_stagate(
        mat,
        coords,
        n_latent=8,
        n_heads=2,
        n_epochs=3,
        run_post_leiden=False,
        stream=None,
        seed=SEED,
    )
    assert hasattr(result, "n_spots"), "STAGATE result missing .n_spots"
    assert hasattr(result, "d_embed"), "STAGATE result missing .d_embed"
    assert hasattr(result, "embedding_view"), "STAGATE result missing .embedding_view"
    assert result.n_spots == n_spots, f"STAGATE n_spots {result.n_spots} != {n_spots}"


# ---------------------------------------------------------------------------
# Tier 3: cell2fate (fit + predict)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_cell2fate_smoke():
    """Cell2fate: fit model then call kinetics_dict — checks alpha/beta/gamma."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.spatial.cell2fate import fit

    expr_mat = _tiny_expression()
    intron_mat = _tiny_expression()  # unspliced proxy

    model = fit(
        expr_mat,
        intron_mat,
        K=4,
        n_epochs=3,
        n_latent=8,
        stream=None,
        seed=SEED,
    )
    # fit returns a Cell2FateModel wrapper
    assert model is not None, "cell2fate fit returned None"
    kin = model.kinetics_dict()
    for key in ("alpha", "beta", "gamma"):
        assert key in kin, f"cell2fate kinetics_dict missing '{key}'"
    assert kin["alpha"].shape[0] == N_GENES, (
        f"alpha rows {kin['alpha'].shape[0]} != N_GENES {N_GENES}"
    )
    # predict: module_loadings shape (K, N_GENES)
    loadings = model.module_loadings()
    assert loadings.shape == (4, N_GENES), (
        f"module_loadings shape {loadings.shape} != (4, {N_GENES})"
    )


# ---------------------------------------------------------------------------
# Tier 3: discrete_diffusion (train + sample)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_discrete_diffusion_smoke():
    """DiscreteDiffusion: train model then sample n=10 synthetic cells."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.generative.discrete_diffusion import sample, train

    mat = _tiny_expression()  # genes × cells

    model = train(
        mat,
        n_latent=8,
        n_steps=5,
        n_epochs=2,
        stream=None,
        seed=SEED,
    )
    assert model is not None, "discrete_diffusion train returned None"
    assert hasattr(model, "_result"), "DiscreteDiffusionWrapper missing ._result"

    synth = sample(model, n=10, stream=None, seed=SEED)
    synth_arr = np.asarray(synth)
    assert synth_arr.shape[0] == 10, f"sampled array first dim {synth_arr.shape[0]} != 10"


# ---------------------------------------------------------------------------
# Tier 3: perturb_graph (fit + predict)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_perturb_graph_smoke():
    """PerturbGraph: fit model, check pert_embeddings, then predict perturbation."""
    if not gpu_available():
        pytest.skip("GPU not available")

    from singlet.gpu.perturbation.perturb_graph import fit

    mat = _tiny_expression()
    n_perts = 5
    pert_labels = RNG.integers(-1, n_perts, size=N_CELLS).astype(np.int32)

    model = fit(
        mat,
        pert_labels,
        d_latent=8,
        n_epochs=3,
        stream=None,
        seed=SEED,
    )
    assert model is not None, "perturb_graph fit returned None"
    embs = model.pert_embeddings()
    assert embs.shape == (n_perts, 8), f"pert_embeddings shape {embs.shape} != ({n_perts}, 8)"
    # predict: shape (N_CELLS, N_GENES)
    pred = model.predict_perturbation(mat, pert_id=0, target_dose=1.0, stream=None)
    pred_arr = np.asarray(pred)
    assert pred_arr.shape == (N_CELLS, N_GENES), (
        f"predict_perturbation shape {pred_arr.shape} != ({N_CELLS}, {N_GENES})"
    )


# ---------------------------------------------------------------------------
# Tier 3: ssgsea
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_ssgsea_smoke():
    """ssGSEA: result stored in AnnData.obsm with shape (n_cells, n_sets)."""
    if not gpu_available():
        pytest.skip("GPU not available")

    try:
        import anndata
        import pandas as pd
    except ImportError:
        pytest.skip("anndata or pandas not installed")

    from singlet.gpu.enrich.ssgsea import run_ssgsea

    mat_np = _tiny_expression().T.toarray()  # cells × genes
    gene_names = [f"Gene{i}" for i in range(N_GENES)]
    adata = anndata.AnnData(X=mat_np)
    adata.var_names = gene_names

    n_sets = 5
    net = pd.DataFrame(
        {
            "source": [f"Set{j}" for j in range(n_sets) for _ in range(8)],
            "target": [f"Gene{RNG.integers(0, N_GENES)}" for _ in range(n_sets * 8)],
        }
    )

    run_ssgsea(adata, net, stream=None, seed=SEED)
    assert "ssgsea_scores" in adata.obsm, "ssGSEA did not write 'ssgsea_scores' to obsm"
    assert adata.obsm["ssgsea_scores"].shape == (N_CELLS, n_sets), (
        f"ssgsea_scores shape {adata.obsm['ssgsea_scores'].shape} != ({N_CELLS}, {n_sets})"
    )


# ---------------------------------------------------------------------------
# Tier 3: progeny
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_progeny_smoke():
    """PROGENy: result stored in AnnData.obsm['progeny_scores'] (n_cells × n_pathways)."""
    if not gpu_available():
        pytest.skip("GPU not available")

    try:
        import anndata
        import pandas as pd
    except ImportError:
        pytest.skip("anndata or pandas not installed")

    from singlet.gpu.enrich.progeny import run_progeny

    mat_np = _tiny_expression().T.toarray()  # cells × genes
    gene_names = [f"Gene{i}" for i in range(N_GENES)]
    adata = anndata.AnnData(X=mat_np)
    adata.var_names = gene_names

    n_pathways = 4
    weights = pd.DataFrame(
        {
            "pathway": [f"Path{j}" for j in range(n_pathways) for _ in range(10)],
            "gene": [f"Gene{RNG.integers(0, N_GENES)}" for _ in range(n_pathways * 10)],
            "weight": RNG.standard_normal(n_pathways * 10).astype(np.float32),
        }
    )

    run_progeny(adata, weights, stream=None, seed=SEED)
    assert "progeny_scores" in adata.obsm, "PROGENy did not write 'progeny_scores' to obsm"
    assert adata.obsm["progeny_scores"].shape == (N_CELLS, n_pathways), (
        f"progeny_scores shape {adata.obsm['progeny_scores'].shape} != ({N_CELLS}, {n_pathways})"
    )


# ---------------------------------------------------------------------------
# Tier 3: omnidoublet
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_omnidoublet_smoke():
    """OmniDoublet: writes obs keys omni_doublet_score and omni_doublet_call."""
    if not gpu_available():
        pytest.skip("GPU not available")

    try:
        import anndata
    except ImportError:
        pytest.skip("anndata not installed")

    from singlet.gpu.qc.omnidoublet import run_omni_doublet

    rna_np = _tiny_expression().T.toarray()  # cells × genes
    adt_np = _tiny_adt().T.toarray()  # cells × adt
    adata = anndata.AnnData(X=rna_np)
    adata.obsm["adt"] = adt_np

    run_omni_doublet(
        adata, adt_key="adt", n_pcs_rna=10, n_pcs_adt=5, n_hvg=50, k=10, stream=None, seed=SEED
    )

    assert "omni_doublet_score" in adata.obs.columns, (
        "OmniDoublet did not write 'omni_doublet_score' to obs"
    )
    assert "omni_doublet_call" in adata.obs.columns, (
        "OmniDoublet did not write 'omni_doublet_call' to obs"
    )
    scores = adata.obs["omni_doublet_score"].to_numpy(dtype=float)
    assert np.all(np.isfinite(scores)), "omni_doublet_score contains non-finite values"


# ---------------------------------------------------------------------------
# Tier 3: doublet_score
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_doublet_score_smoke():
    """Scrublet-style doublet_score: writes obs keys doublet_score and doublet_call."""
    if not gpu_available():
        pytest.skip("GPU not available")

    try:
        import anndata
    except ImportError:
        pytest.skip("anndata not installed")

    from singlet.gpu.qc.doublet_score import run_doublet_score

    pca_mat = _tiny_pca()
    adata = anndata.AnnData(X=_tiny_expression().T.toarray())
    adata.obsm["X_pca"] = pca_mat

    run_doublet_score(adata, embedding_key="X_pca", k=10, stream=None, seed=SEED)

    assert "doublet_score" in adata.obs.columns, (
        "doublet_score did not write 'doublet_score' to obs"
    )
    assert "doublet_call" in adata.obs.columns, "doublet_score did not write 'doublet_call' to obs"


# ---------------------------------------------------------------------------
# Tier 3: csi_gep
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_csi_gep_smoke():
    """CSI-GEP consensus NMF: writes varm and obsm keys."""
    if not gpu_available():
        pytest.skip("GPU not available")

    try:
        import anndata
    except ImportError:
        pytest.skip("anndata not installed")

    from singlet.gpu.reduce.nmf.csi_gep import run_csi_gep

    mat_np = _tiny_expression().T.toarray()  # cells × genes
    adata = anndata.AnnData(X=mat_np)

    run_csi_gep(
        adata,
        k_range=[3, 5],
        n_runs=5,
        subsample_frac=0.8,
        nmf_max_iter=10,
        stream=None,
        seed=SEED,
    )

    assert "csi_gep_programs" in adata.varm, "CSI-GEP did not write 'csi_gep_programs' to varm"
    assert "csi_gep_usage" in adata.obsm, "CSI-GEP did not write 'csi_gep_usage' to obsm"
    k_chosen = adata.varm["csi_gep_programs"].shape[1]
    assert k_chosen in (3, 5), f"CSI-GEP chose k={k_chosen} not in [3, 5]"
    assert adata.obsm["csi_gep_usage"].shape == (N_CELLS, k_chosen), (
        f"csi_gep_usage shape {adata.obsm['csi_gep_usage'].shape} != ({N_CELLS}, {k_chosen})"
    )
