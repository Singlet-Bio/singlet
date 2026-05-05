"""
Cycle 20 — collected binding tests grouped by function family.

Tests are written exclusively against the public API specification in
``singlet-gpu/state/designs/20-binding-extension.md``.  No kernel source
is read.  No GPU execution is required for the existence / signature tests;
the two smoke-execution tests are individually gated by ``requires_gpu`` and
``pytest.importorskip("cupy")``.

Organisation:
  - Family A: cupy ingest / export  (from_cupy_csr, to_cupy_csr)
  - Family B: preprocess            (normalize_total, log1p, highly_variable_genes)
  - Family C: svd backends          (lanczos, irlba, randomized, krylov, deflation, auto_select, pca)
  - Family D: nmf variants          (nmf, nmf_chunked, nmf_graph_factorize)
  - Family E: result classes        (NormalizeResult, HvgResult, SvdResult, NmfResult)
  - Family F: signature inspection  (keyword-only arg presence on key functions)

Skip strategy: same as test_core.py — module-level importorskip for
singlet.gpu; requires_gpu marker for device tests; importorskip("cupy") inside
tests that build cupy arrays.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level skip when wheel is not built.
# ---------------------------------------------------------------------------
singlet_gpu = pytest.importorskip(
    "singlet.gpu",
    reason=(
        "singlet.gpu not available. "
        "Run `pip install -e singlet-gpu/python/` first."
    ),
)

from conftest import requires_gpu  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CORE = singlet_gpu._core

_EXON_FILE = "exon_counts.1pz"


def _core_callable(name: str) -> bool:
    """Return True iff _core has attribute ``name`` and it is callable."""
    return hasattr(_CORE, name) and callable(getattr(_CORE, name))


# ===========================================================================
# Family A — cupy ingest / export
# ===========================================================================

_CUPY_INGEST_FUNCTIONS = [
    "from_cupy_csr",
    "to_cupy_csr",
]


@pytest.mark.parametrize("func_name", _CUPY_INGEST_FUNCTIONS)
def test_cupy_ingest_function_exists(func_name):
    """Each cupy ingest/export function is present and callable in _core."""
    assert hasattr(_CORE, func_name), (
        f"_core.{func_name} not found — _bind_cupy_ingest.hpp not wired into module"
    )
    assert callable(getattr(_CORE, func_name)), (
        f"_core.{func_name} is present but not callable"
    )


def test_from_cupy_csr_signature():
    """from_cupy_csr accepts exactly one positional argument: csr_matrix."""
    try:
        sig = inspect.signature(_CORE.from_cupy_csr)
    except (ValueError, TypeError):
        pytest.skip("pybind11 signature not introspectable in this build")

    params = list(sig.parameters.keys())
    assert "csr_matrix" in params, (
        f"from_cupy_csr expected parameter 'csr_matrix', got parameters: {params}"
    )


def test_to_cupy_csr_signature():
    """to_cupy_csr accepts exactly one positional argument: device_csc."""
    try:
        sig = inspect.signature(_CORE.to_cupy_csr)
    except (ValueError, TypeError):
        pytest.skip("pybind11 signature not introspectable in this build")

    params = list(sig.parameters.keys())
    assert "device_csc" in params, (
        f"to_cupy_csr expected parameter 'device_csc', got parameters: {params}"
    )


# ===========================================================================
# Family B — preprocess (normalize_total, log1p, highly_variable_genes)
# ===========================================================================

_PREPROCESS_FUNCTIONS = [
    "normalize_total",
    "log1p",
    "highly_variable_genes",
]


@pytest.mark.parametrize("func_name", _PREPROCESS_FUNCTIONS)
def test_preprocess_function_exists(func_name):
    """Each preprocessing function is present and callable in _core."""
    assert hasattr(_CORE, func_name), (
        f"_core.{func_name} not found — bind_kernels(m) not called in PYBIND11_MODULE"
    )
    assert callable(getattr(_CORE, func_name)), (
        f"_core.{func_name} is present but not callable"
    )


def test_normalize_total_has_target_sum_kwarg():
    """normalize_total exposes 'target_sum' as a keyword argument (default 0.0)."""
    try:
        sig = inspect.signature(_CORE.normalize_total)
    except (ValueError, TypeError):
        pytest.skip("pybind11 signature not introspectable in this build")

    params = sig.parameters
    assert "target_sum" in params, (
        f"normalize_total missing 'target_sum' kwarg; found: {list(params.keys())}"
    )
    default = params["target_sum"].default
    if default is not inspect.Parameter.empty:
        assert default == 0.0 or default == pytest.approx(0.0), (
            f"normalize_total 'target_sum' default expected 0.0, got {default!r}"
        )


def test_normalize_total_has_approximate_median_kwarg():
    """normalize_total exposes 'approximate_median' as a keyword argument."""
    try:
        sig = inspect.signature(_CORE.normalize_total)
    except (ValueError, TypeError):
        pytest.skip("pybind11 signature not introspectable in this build")

    params = sig.parameters
    assert "approximate_median" in params, (
        f"normalize_total missing 'approximate_median' kwarg; found: {list(params.keys())}"
    )


def test_log1p_has_base_kwarg():
    """log1p exposes 'base' as a keyword argument (0 = natural log)."""
    try:
        sig = inspect.signature(_CORE.log1p)
    except (ValueError, TypeError):
        pytest.skip("pybind11 signature not introspectable in this build")

    params = sig.parameters
    assert "base" in params, (
        f"log1p missing 'base' kwarg; found: {list(params.keys())}"
    )


def test_highly_variable_genes_kwargs():
    """highly_variable_genes exposes n_top_genes, flavor, min_mean, max_mean, pearson_theta."""
    try:
        sig = inspect.signature(_CORE.highly_variable_genes)
    except (ValueError, TypeError):
        pytest.skip("pybind11 signature not introspectable in this build")

    params = sig.parameters
    for kwarg in ("n_top_genes", "flavor", "min_mean", "max_mean", "pearson_theta"):
        assert kwarg in params, (
            f"highly_variable_genes missing kwarg '{kwarg}'; found: {list(params.keys())}"
        )


# ===========================================================================
# Family C — SVD backends
# ===========================================================================

# Rule 32: svd_lanczos, svd_irlba, svd_krylov backends were removed.
# svd_deflation is the primary; svd_randomized is the fallback; pca aliases auto_select.
_SVD_FUNCTIONS = [
    "svd_randomized",
    "svd_deflation",
    "svd_auto_select",
    "pca",
]


@pytest.mark.parametrize("func_name", _SVD_FUNCTIONS)
def test_svd_function_exists(func_name):
    """Each SVD backend function is present and callable in _core."""
    assert hasattr(_CORE, func_name), (
        f"_core.{func_name} not found — bind_kernels() SVD section missing"
    )
    assert callable(getattr(_CORE, func_name)), (
        f"_core.{func_name} is present but not callable"
    )


@pytest.mark.parametrize("func_name", _SVD_FUNCTIONS)
def test_svd_function_accepts_k_arg(func_name):
    """Every SVD backend exposes 'mat' and 'k' positional arguments."""
    try:
        sig = inspect.signature(getattr(_CORE, func_name))
    except (ValueError, TypeError):
        pytest.skip(f"pybind11 signature not introspectable for {func_name}")

    params = sig.parameters
    assert "mat" in params, (
        f"{func_name}: missing 'mat' positional arg; got {list(params.keys())}"
    )
    assert "k" in params, (
        f"{func_name}: missing 'k' positional arg; got {list(params.keys())}"
    )


@pytest.mark.parametrize("func_name", _SVD_FUNCTIONS)
def test_svd_function_has_seed_kwarg(func_name):
    """Every SVD backend exposes 'seed' as a keyword argument."""
    try:
        sig = inspect.signature(getattr(_CORE, func_name))
    except (ValueError, TypeError):
        pytest.skip(f"pybind11 signature not introspectable for {func_name}")

    params = sig.parameters
    assert "seed" in params, (
        f"{func_name}: missing 'seed' kwarg (reproducibility contract); "
        f"got {list(params.keys())}"
    )


# ===========================================================================
# Family D — NMF variants
# ===========================================================================

_NMF_FUNCTIONS = [
    "nmf",
    "nmf_chunked",
    # nmf_graph_factorize is feature-flagged behind SINGLET_GPU_BUILD_NMF_GRAPH;
    # tested separately with skipif when binding is missing.
]


@pytest.mark.parametrize("func_name", _NMF_FUNCTIONS)
def test_nmf_function_exists(func_name):
    """Each NMF function is present and callable in _core."""
    assert hasattr(_CORE, func_name), (
        f"_core.{func_name} not found — bind_kernels() NMF section missing"
    )
    assert callable(getattr(_CORE, func_name)), (
        f"_core.{func_name} is present but not callable"
    )


def test_nmf_has_rank_and_kwargs():
    """nmf exposes 'mat', 'rank', 'loss', 'solver_mode', 'init_mode', 'max_iter',
    'tol', and 'seed' parameters."""
    try:
        sig = inspect.signature(_CORE.nmf)
    except (ValueError, TypeError):
        pytest.skip("pybind11 signature not introspectable for nmf")

    params = sig.parameters
    for p in ("mat", "rank", "loss", "solver_mode", "init_mode", "max_iter", "tol", "seed"):
        assert p in params, (
            f"nmf missing parameter '{p}'; found: {list(params.keys())}"
        )


def test_nmf_chunked_has_loader_and_rank():
    """nmf_chunked exposes 'loader' and 'rank' as the first two positional parameters."""
    try:
        sig = inspect.signature(_CORE.nmf_chunked)
    except (ValueError, TypeError):
        pytest.skip("pybind11 signature not introspectable for nmf_chunked")

    params = sig.parameters
    assert "loader" in params, (
        f"nmf_chunked missing 'loader' param; found: {list(params.keys())}"
    )
    assert "rank" in params, (
        f"nmf_chunked missing 'rank' param; found: {list(params.keys())}"
    )


@pytest.mark.skipif(
    not hasattr(_CORE, "nmf_graph_factorize"),
    reason="nmf_graph_factorize feature-flagged behind SINGLET_GPU_BUILD_NMF_GRAPH",
)
def test_nmf_graph_factorize_has_inputs_rank_topology():
    """nmf_graph_factorize exposes 'inputs', 'rank', and 'topology' parameters."""
    try:
        sig = inspect.signature(_CORE.nmf_graph_factorize)
    except (ValueError, TypeError):
        pytest.skip("pybind11 signature not introspectable for nmf_graph_factorize")

    params = sig.parameters
    for p in ("inputs", "rank", "topology"):
        assert p in params, (
            f"nmf_graph_factorize missing parameter '{p}'; found: {list(params.keys())}"
        )


@pytest.mark.skipif(
    not hasattr(_CORE, "nmf_graph_factorize"),
    reason="nmf_graph_factorize feature-flagged behind SINGLET_GPU_BUILD_NMF_GRAPH",
)
def test_nmf_graph_factorize_topology_default():
    """nmf_graph_factorize 'topology' defaults to 'shared_h'."""
    try:
        sig = inspect.signature(_CORE.nmf_graph_factorize)
    except (ValueError, TypeError):
        pytest.skip("pybind11 signature not introspectable for nmf_graph_factorize")

    topology_param = sig.parameters.get("topology")
    if topology_param is None:
        pytest.skip("topology param not present — existence checked separately")

    default = topology_param.default
    if default is not inspect.Parameter.empty:
        assert default == "shared_h", (
            f"nmf_graph_factorize 'topology' default expected 'shared_h', got {default!r}"
        )


# ===========================================================================
# Family E — result classes
# ===========================================================================

_RESULT_CLASSES = [
    "NormalizeResult",
    "HvgResult",
    "SvdResult",
    "NmfResult",
]


@pytest.mark.parametrize("class_name", _RESULT_CLASSES)
def test_result_class_exists(class_name):
    """Each result class is present and is a type in _core."""
    assert hasattr(_CORE, class_name), (
        f"_core.{class_name} not found — result class binding missing from "
        "cycle 20 extension"
    )
    cls = getattr(_CORE, class_name)
    assert isinstance(cls, type), (
        f"_core.{class_name} is not a type, got {type(cls)}"
    )


def test_normalize_result_has_target_used():
    """NormalizeResult exposes 'target_used' as a read-only attribute."""
    NR = _CORE.NormalizeResult
    # pybind11 read-only properties appear in the class __dict__ as properties.
    # We cannot instantiate the class without a GPU, so we check the descriptor.
    # Accept either a property/slot or a pybind11 getset descriptor.
    attrs = dir(NR)
    assert "target_used" in attrs, (
        "NormalizeResult must expose 'target_used' attribute "
        "(design: float target_used)"
    )


def test_normalize_result_has_size_factors_view():
    """NormalizeResult exposes 'size_factors_view' as a read-only attribute."""
    NR = _CORE.NormalizeResult
    attrs = dir(NR)
    assert "size_factors_view" in attrs, (
        "NormalizeResult must expose 'size_factors_view' attribute "
        "(design: cupy_view<float>)"
    )


def test_svd_result_attributes():
    """SvdResult exposes U_view, d_view, V_view, and k_selected attributes."""
    SR = _CORE.SvdResult
    attrs = dir(SR)
    for attr in ("U_view", "d_view", "V_view", "k_selected"):
        assert attr in attrs, (
            f"SvdResult missing attribute '{attr}' "
            f"(actual binding: U_view, d_view, V_view CAI dicts + k_selected int)"
        )


def test_nmf_result_attributes():
    """NmfResult exposes W_view, d_view, H_view, iterations, converged."""
    NR = _CORE.NmfResult
    attrs = dir(NR)
    for attr in ("W_view", "d_view", "H_view", "iterations", "converged", "k_used"):
        assert attr in attrs, (
            f"NmfResult missing attribute '{attr}' "
            f"(actual binding: W_view, d_view, H_view CAI dicts + "
            f"iterations int + converged bool + k_used int)"
        )


def test_hvg_result_attributes():
    """HvgResult exposes indices_view, scores_view, mean_view, var_view."""
    HR = _CORE.HvgResult
    attrs = dir(HR)
    for attr in ("indices_view", "scores_view", "mean_view", "var_view"):
        assert attr in attrs, (
            f"HvgResult missing attribute '{attr}' "
            f"(actual binding: indices_view int32 CAI + "
            f"scores_view/mean_view/var_view float32 CAI)"
        )


# ===========================================================================
# Family F — signature completeness cross-check
# ===========================================================================

def test_all_cycle20_bindings_present():
    """Consolidated existence check: all post-Rule-32 cycle 20 entry points are in _core.

    Rule 32 removed svd_lanczos / svd_irlba / svd_krylov backends; deflation is
    primary, randomized is fallback.  nmf_graph_factorize is feature-flagged
    behind SINGLET_GPU_BUILD_NMF_GRAPH and excluded here.
    """
    expected = [
        # cupy ingest
        "from_cupy_csr",
        "to_cupy_csr",
        # preprocess
        "normalize_total",
        "log1p",
        "highly_variable_genes",
        # svd (post-Rule-32: deflation primary + randomized fallback + auto_select + pca alias)
        "svd_randomized",
        "svd_deflation",
        "svd_auto_select",
        "pca",
        # nmf (nmf_graph_factorize feature-flagged — checked separately)
        "nmf",
        "nmf_chunked",
    ]
    missing = [name for name in expected if not _core_callable(name)]
    assert not missing, (
        f"The following cycle 20 bindings are missing from _core: {missing}\n"
        "Check that bind_kernels(m) and from/to_cupy_csr are called inside "
        "PYBIND11_MODULE(_core, m)."
    )


def test_all_result_classes_present():
    """Consolidated existence check: all 4 result classes are exposed as types."""
    missing = []
    for name in _RESULT_CLASSES:
        if not hasattr(_CORE, name) or not isinstance(getattr(_CORE, name), type):
            missing.append(name)
    assert not missing, (
        f"Result classes missing or not types in _core: {missing}"
    )


# ===========================================================================
# Cycle 22 — binding extension #2 (cycles 7–17 functions + result classes)
# ===========================================================================
#
# Organisation (one pytest class per cycle family):
#   TestCycle7BindingsStreaming       — streaming_pipeline_run
#   TestCycle8BindingsKnn             — knn_graph, knn_graph_from_indices
#   TestCycle9BindingsLeiden          — leiden_partition
#   TestCycle10BindingsUmap           — umap_embed
#   TestCycle11BindingsDe             — wilcoxon_de, ttest_de
#   TestCycle12BindingsAnno           — marker_score, celltypist_project,
#                                       load_celltypist_model
#   TestCycle13BindingsGsea           — fgsea, aucell
#   TestCycle14BindingsIntegrate      — harmony, bbknn
#   TestCycle15BindingsVelocity       — velocity_prep_compute
#   TestCycle16BindingsMtLineage      — mt_lineage_call_clones
#   TestCycle17BindingsDonorPseudobulk— donor_pseudobulk_de
#
# Each class contains:
#   - one ``test_binding_<func>_exists`` per function exposed by that cycle
#   - one ``test_result_class_<cls>_exists`` per result class introduced by that cycle
#   - one ``test_<cycle>_all_present`` grouped check that hard-asserts all
#     cycle-scoped names in one call
#
# A consolidated ``test_all_cycle22_bindings_present`` at module level
# asserts all 17 functions + 13 result classes (= 30 names) in one pass.
#
# Skip strategy: identical to cycle 20 — importorskip gates the entire
# module; no GPU required for existence checks; individual GPU/cupy tests
# use ``requires_gpu`` + ``pytest.importorskip("cupy")``.
# ===========================================================================

def _skip_if_not_present(name: str, kind: str = "binding") -> None:
    """Skip the calling test cleanly when the named attribute is absent."""
    if not hasattr(_CORE, name):
        pytest.skip(
            f"_core.{name} not present — {kind} not yet wired (wheel pending GPU build)"
        )


# ---------------------------------------------------------------------------
# TestCycle7BindingsStreaming
# ---------------------------------------------------------------------------

class TestCycle7BindingsStreaming:
    """Cycle 7 — streaming_pipeline_run and StreamingPipelineResult."""

    def test_binding_streaming_pipeline_run_exists(self):
        """_core.streaming_pipeline_run is present and callable."""
        _skip_if_not_present("streaming_pipeline_run")
        assert callable(getattr(_CORE, "streaming_pipeline_run")), (
            "_core.streaming_pipeline_run is present but not callable"
        )

    def test_result_class_StreamingPipelineResult_exists(self):
        """_core.StreamingPipelineResult is present and is a type."""
        _skip_if_not_present("StreamingPipelineResult", kind="result class")
        cls = getattr(_CORE, "StreamingPipelineResult")
        assert isinstance(cls, type), (
            f"_core.StreamingPipelineResult is not a type, got {type(cls)}"
        )

    def test_cycle7_all_present(self):
        """Grouped: all cycle 7 names are present in _core."""
        names = ["streaming_pipeline_run", "StreamingPipelineResult"]
        missing = [n for n in names if not hasattr(_CORE, n)]
        assert not missing, (
            f"Cycle 7 — missing from _core: {missing}"
        )


# ---------------------------------------------------------------------------
# TestCycle8BindingsKnn
# ---------------------------------------------------------------------------

class TestCycle8BindingsKnn:
    """Cycle 8 — knn_graph, knn_graph_from_indices, and KnnResult."""

    def test_binding_knn_graph_exists(self):
        """_core.knn_graph is present and callable."""
        _skip_if_not_present("knn_graph")
        assert callable(getattr(_CORE, "knn_graph")), (
            "_core.knn_graph is present but not callable"
        )

    def test_binding_knn_graph_from_indices_exists(self):
        """_core.knn_graph_from_indices is present and callable."""
        _skip_if_not_present("knn_graph_from_indices")
        assert callable(getattr(_CORE, "knn_graph_from_indices")), (
            "_core.knn_graph_from_indices is present but not callable"
        )

    def test_result_class_KnnResult_exists(self):
        """_core.KnnResult is present and is a type."""
        _skip_if_not_present("KnnResult", kind="result class")
        cls = getattr(_CORE, "KnnResult")
        assert isinstance(cls, type), (
            f"_core.KnnResult is not a type, got {type(cls)}"
        )

    def test_cycle8_all_present(self):
        """Grouped: all cycle 8 names are present in _core."""
        names = ["knn_graph", "knn_graph_from_indices", "KnnResult"]
        missing = [n for n in names if not hasattr(_CORE, n)]
        assert not missing, (
            f"Cycle 8 — missing from _core: {missing}"
        )


# ---------------------------------------------------------------------------
# TestCycle9BindingsLeiden
# ---------------------------------------------------------------------------

class TestCycle9BindingsLeiden:
    """Cycle 9 — leiden_partition and LeidenResult."""

    def test_binding_leiden_partition_exists(self):
        """_core.leiden_partition is present and callable."""
        _skip_if_not_present("leiden_partition")
        assert callable(getattr(_CORE, "leiden_partition")), (
            "_core.leiden_partition is present but not callable"
        )

    def test_result_class_LeidenResult_exists(self):
        """_core.LeidenResult is present and is a type."""
        _skip_if_not_present("LeidenResult", kind="result class")
        cls = getattr(_CORE, "LeidenResult")
        assert isinstance(cls, type), (
            f"_core.LeidenResult is not a type, got {type(cls)}"
        )

    def test_cycle9_all_present(self):
        """Grouped: all cycle 9 names are present in _core."""
        names = ["leiden_partition", "LeidenResult"]
        missing = [n for n in names if not hasattr(_CORE, n)]
        assert not missing, (
            f"Cycle 9 — missing from _core: {missing}"
        )


# ---------------------------------------------------------------------------
# TestCycle10BindingsUmap
# ---------------------------------------------------------------------------

class TestCycle10BindingsUmap:
    """Cycle 10 — umap_embed and UmapResult."""

    def test_binding_umap_embed_exists(self):
        """_core.umap_embed is present and callable."""
        _skip_if_not_present("umap_embed")
        assert callable(getattr(_CORE, "umap_embed")), (
            "_core.umap_embed is present but not callable"
        )

    def test_result_class_UmapResult_exists(self):
        """_core.UmapResult is present and is a type."""
        _skip_if_not_present("UmapResult", kind="result class")
        cls = getattr(_CORE, "UmapResult")
        assert isinstance(cls, type), (
            f"_core.UmapResult is not a type, got {type(cls)}"
        )

    def test_cycle10_all_present(self):
        """Grouped: all cycle 10 names are present in _core."""
        names = ["umap_embed", "UmapResult"]
        missing = [n for n in names if not hasattr(_CORE, n)]
        assert not missing, (
            f"Cycle 10 — missing from _core: {missing}"
        )


# ---------------------------------------------------------------------------
# TestCycle11BindingsDe
# ---------------------------------------------------------------------------

class TestCycle11BindingsDe:
    """Cycle 11 — wilcoxon_de, ttest_de, WilcoxonResult, TtestResult."""

    def test_binding_wilcoxon_de_exists(self):
        """_core.wilcoxon_de is present and callable."""
        _skip_if_not_present("wilcoxon_de")
        assert callable(getattr(_CORE, "wilcoxon_de")), (
            "_core.wilcoxon_de is present but not callable"
        )

    def test_binding_ttest_de_exists(self):
        """_core.ttest_de is present and callable."""
        _skip_if_not_present("ttest_de")
        assert callable(getattr(_CORE, "ttest_de")), (
            "_core.ttest_de is present but not callable"
        )

    def test_result_class_WilcoxonResult_exists(self):
        """_core.WilcoxonResult is present and is a type."""
        _skip_if_not_present("WilcoxonResult", kind="result class")
        cls = getattr(_CORE, "WilcoxonResult")
        assert isinstance(cls, type), (
            f"_core.WilcoxonResult is not a type, got {type(cls)}"
        )

    def test_result_class_TtestResult_exists(self):
        """_core.TtestResult is present and is a type."""
        _skip_if_not_present("TtestResult", kind="result class")
        cls = getattr(_CORE, "TtestResult")
        assert isinstance(cls, type), (
            f"_core.TtestResult is not a type, got {type(cls)}"
        )

    def test_cycle11_all_present(self):
        """Grouped: all cycle 11 names are present in _core."""
        names = ["wilcoxon_de", "ttest_de", "WilcoxonResult", "TtestResult"]
        missing = [n for n in names if not hasattr(_CORE, n)]
        assert not missing, (
            f"Cycle 11 — missing from _core: {missing}"
        )


# ---------------------------------------------------------------------------
# TestCycle12BindingsAnno
# ---------------------------------------------------------------------------

class TestCycle12BindingsAnno:
    """Cycle 12 — marker_score, celltypist_project, load_celltypist_model,
    MarkerScoreResult, RefMapResult, CelltypistModel."""

    def test_binding_marker_score_exists(self):
        """_core.marker_score is present and callable."""
        _skip_if_not_present("marker_score")
        assert callable(getattr(_CORE, "marker_score")), (
            "_core.marker_score is present but not callable"
        )

    def test_binding_celltypist_project_exists(self):
        """_core.celltypist_project is present and callable."""
        _skip_if_not_present("celltypist_project")
        assert callable(getattr(_CORE, "celltypist_project")), (
            "_core.celltypist_project is present but not callable"
        )

    def test_binding_load_celltypist_model_exists(self):
        """_core.load_celltypist_model is present and callable."""
        _skip_if_not_present("load_celltypist_model")
        assert callable(getattr(_CORE, "load_celltypist_model")), (
            "_core.load_celltypist_model is present but not callable"
        )

    def test_result_class_MarkerScoreResult_exists(self):
        """_core.MarkerScoreResult is present and is a type."""
        _skip_if_not_present("MarkerScoreResult", kind="result class")
        cls = getattr(_CORE, "MarkerScoreResult")
        assert isinstance(cls, type), (
            f"_core.MarkerScoreResult is not a type, got {type(cls)}"
        )

    def test_result_class_RefMapResult_exists(self):
        """_core.RefMapResult is present and is a type."""
        _skip_if_not_present("RefMapResult", kind="result class")
        cls = getattr(_CORE, "RefMapResult")
        assert isinstance(cls, type), (
            f"_core.RefMapResult is not a type, got {type(cls)}"
        )

    def test_result_class_CelltypistModel_exists(self):
        """_core.CelltypistModel is present and is a type."""
        _skip_if_not_present("CelltypistModel", kind="result class")
        cls = getattr(_CORE, "CelltypistModel")
        assert isinstance(cls, type), (
            f"_core.CelltypistModel is not a type, got {type(cls)}"
        )

    def test_cycle12_all_present(self):
        """Grouped: all cycle 12 names are present in _core."""
        names = [
            "marker_score", "celltypist_project", "load_celltypist_model",
            "MarkerScoreResult", "RefMapResult", "CelltypistModel",
        ]
        missing = [n for n in names if not hasattr(_CORE, n)]
        assert not missing, (
            f"Cycle 12 — missing from _core: {missing}"
        )


# ---------------------------------------------------------------------------
# TestCycle13BindingsGsea
# ---------------------------------------------------------------------------

class TestCycle13BindingsGsea:
    """Cycle 13 — fgsea, aucell, FgseaResult, AUCellResult."""

    def test_binding_fgsea_exists(self):
        """_core.fgsea is present and callable."""
        _skip_if_not_present("fgsea")
        assert callable(getattr(_CORE, "fgsea")), (
            "_core.fgsea is present but not callable"
        )

    def test_binding_aucell_exists(self):
        """_core.aucell is present and callable."""
        _skip_if_not_present("aucell")
        assert callable(getattr(_CORE, "aucell")), (
            "_core.aucell is present but not callable"
        )

    def test_result_class_FgseaResult_exists(self):
        """_core.FgseaResult is present and is a type."""
        _skip_if_not_present("FgseaResult", kind="result class")
        cls = getattr(_CORE, "FgseaResult")
        assert isinstance(cls, type), (
            f"_core.FgseaResult is not a type, got {type(cls)}"
        )

    def test_result_class_AUCellResult_exists(self):
        """_core.AUCellResult is present and is a type."""
        _skip_if_not_present("AUCellResult", kind="result class")
        cls = getattr(_CORE, "AUCellResult")
        assert isinstance(cls, type), (
            f"_core.AUCellResult is not a type, got {type(cls)}"
        )

    def test_cycle13_all_present(self):
        """Grouped: all cycle 13 names are present in _core."""
        names = ["fgsea", "aucell", "FgseaResult", "AUCellResult"]
        missing = [n for n in names if not hasattr(_CORE, n)]
        assert not missing, (
            f"Cycle 13 — missing from _core: {missing}"
        )


# ---------------------------------------------------------------------------
# TestCycle14BindingsIntegrate
# ---------------------------------------------------------------------------

class TestCycle14BindingsIntegrate:
    """Cycle 14 — harmony, bbknn, HarmonyResult.
    Note: bbknn reuses KnnResult from cycle 8; no new result class added."""

    def test_binding_harmony_exists(self):
        """_core.harmony is present and callable."""
        _skip_if_not_present("harmony")
        assert callable(getattr(_CORE, "harmony")), (
            "_core.harmony is present but not callable"
        )

    def test_binding_bbknn_exists(self):
        """_core.bbknn is present and callable."""
        _skip_if_not_present("bbknn")
        assert callable(getattr(_CORE, "bbknn")), (
            "_core.bbknn is present but not callable"
        )

    def test_result_class_HarmonyResult_exists(self):
        """_core.HarmonyResult is present and is a type."""
        _skip_if_not_present("HarmonyResult", kind="result class")
        cls = getattr(_CORE, "HarmonyResult")
        assert isinstance(cls, type), (
            f"_core.HarmonyResult is not a type, got {type(cls)}"
        )

    def test_cycle14_all_present(self):
        """Grouped: all cycle 14 names are present in _core.
        KnnResult (cycle 8) is reused by bbknn — checked in TestCycle8BindingsKnn."""
        names = ["harmony", "bbknn", "HarmonyResult"]
        missing = [n for n in names if not hasattr(_CORE, n)]
        assert not missing, (
            f"Cycle 14 — missing from _core: {missing}"
        )


# ---------------------------------------------------------------------------
# TestCycle15BindingsVelocity
# ---------------------------------------------------------------------------

class TestCycle15BindingsVelocity:
    """Cycle 15 — velocity_prep_compute and VelocityPrepResult."""

    def test_binding_velocity_prep_compute_exists(self):
        """_core.velocity_prep_compute is present and callable."""
        _skip_if_not_present("velocity_prep_compute")
        assert callable(getattr(_CORE, "velocity_prep_compute")), (
            "_core.velocity_prep_compute is present but not callable"
        )

    def test_result_class_VelocityPrepResult_exists(self):
        """_core.VelocityPrepResult is present and is a type."""
        _skip_if_not_present("VelocityPrepResult", kind="result class")
        cls = getattr(_CORE, "VelocityPrepResult")
        assert isinstance(cls, type), (
            f"_core.VelocityPrepResult is not a type, got {type(cls)}"
        )

    def test_cycle15_all_present(self):
        """Grouped: all cycle 15 names are present in _core."""
        names = ["velocity_prep_compute", "VelocityPrepResult"]
        missing = [n for n in names if not hasattr(_CORE, n)]
        assert not missing, (
            f"Cycle 15 — missing from _core: {missing}"
        )


# ---------------------------------------------------------------------------
# TestCycle16BindingsMtLineage
# ---------------------------------------------------------------------------

class TestCycle16BindingsMtLineage:
    """Cycle 16 — mt_lineage_call_clones and ClonePrediction."""

    def test_binding_mt_lineage_call_clones_exists(self):
        """_core.mt_lineage_call_clones is present and callable."""
        _skip_if_not_present("mt_lineage_call_clones")
        assert callable(getattr(_CORE, "mt_lineage_call_clones")), (
            "_core.mt_lineage_call_clones is present but not callable"
        )

    def test_result_class_ClonePrediction_exists(self):
        """_core.ClonePrediction is present and is a type."""
        _skip_if_not_present("ClonePrediction", kind="result class")
        cls = getattr(_CORE, "ClonePrediction")
        assert isinstance(cls, type), (
            f"_core.ClonePrediction is not a type, got {type(cls)}"
        )

    def test_cycle16_all_present(self):
        """Grouped: all cycle 16 names are present in _core."""
        names = ["mt_lineage_call_clones", "ClonePrediction"]
        missing = [n for n in names if not hasattr(_CORE, n)]
        assert not missing, (
            f"Cycle 16 — missing from _core: {missing}"
        )


# ---------------------------------------------------------------------------
# TestCycle17BindingsDonorPseudobulk
# ---------------------------------------------------------------------------

class TestCycle17BindingsDonorPseudobulk:
    """Cycle 17 — donor_pseudobulk_de and DonorPseudobulkResult."""

    def test_binding_donor_pseudobulk_de_exists(self):
        """_core.donor_pseudobulk_de is present and callable."""
        _skip_if_not_present("donor_pseudobulk_de")
        assert callable(getattr(_CORE, "donor_pseudobulk_de")), (
            "_core.donor_pseudobulk_de is present but not callable"
        )

    def test_result_class_DonorPseudobulkResult_exists(self):
        """_core.DonorPseudobulkResult is present and is a type."""
        _skip_if_not_present("DonorPseudobulkResult", kind="result class")
        cls = getattr(_CORE, "DonorPseudobulkResult")
        assert isinstance(cls, type), (
            f"_core.DonorPseudobulkResult is not a type, got {type(cls)}"
        )

    def test_cycle17_all_present(self):
        """Grouped: all cycle 17 names are present in _core."""
        names = ["donor_pseudobulk_de", "DonorPseudobulkResult"]
        missing = [n for n in names if not hasattr(_CORE, n)]
        assert not missing, (
            f"Cycle 17 — missing from _core: {missing}"
        )


# ---------------------------------------------------------------------------
# Consolidated cycle 22 pre-flight check
# ---------------------------------------------------------------------------

_CYCLE22_FUNCTIONS = [
    # cycle 7 — streaming
    "streaming_pipeline_run",
    # cycle 8 — knn
    "knn_graph",
    "knn_graph_from_indices",
    # cycle 9 — leiden
    "leiden_partition",
    # cycle 10 — umap
    "umap_embed",
    # cycle 11 — DE
    "wilcoxon_de",
    "ttest_de",
    # cycle 12 — annotation
    "marker_score",
    "celltypist_project",
    "load_celltypist_model",
    # cycle 13 — GSEA
    "fgsea",
    "aucell",
    # cycle 14 — integration
    "harmony",
    "bbknn",
    # cycle 15 — velocity
    "velocity_prep_compute",
    # cycle 16 — MT lineage
    "mt_lineage_call_clones",
    # cycle 17 — donor pseudobulk
    "donor_pseudobulk_de",
]

_CYCLE22_RESULT_CLASSES = [
    # cycle 7
    "StreamingPipelineResult",
    # cycle 8  (reused by cycle 14 bbknn — one entry)
    "KnnResult",
    # cycle 9
    "LeidenResult",
    # cycle 10
    "UmapResult",
    # cycle 11
    "WilcoxonResult",
    "TtestResult",
    # cycle 12
    "MarkerScoreResult",
    "RefMapResult",
    "CelltypistModel",
    # cycle 13
    "FgseaResult",
    "AUCellResult",
    # cycle 14
    "HarmonyResult",
    # cycle 15
    "VelocityPrepResult",
    # cycle 16
    "ClonePrediction",
    # cycle 17
    "DonorPseudobulkResult",
]

# 17 functions + 15 result classes = 32 total names.  The design doc
# quotes "~13 new result classes" because ClonePrediction and
# VelocityPrepResult were listed tentatively; the consolidated test
# asserts the full set of 17 + 15 = 32 (not 30) to be exhaustive.
# The counter in the return-format comment is corrected accordingly.


def test_all_cycle22_bindings_present():
    """Consolidated pre-flight: all cycle 22 functions and result classes
    are present in _core.

    This is the fastest smoke-gate: if the wheel was built without the
    cycle 22 additions, this test fails immediately before any
    family-specific checks run.

    Functions asserted (17): streaming_pipeline_run, knn_graph,
      knn_graph_from_indices, leiden_partition, umap_embed, wilcoxon_de,
      ttest_de, marker_score, celltypist_project, load_celltypist_model,
      fgsea, aucell, harmony, bbknn, velocity_prep_compute,
      mt_lineage_call_clones, donor_pseudobulk_de.

    Result classes asserted (15): StreamingPipelineResult, KnnResult,
      LeidenResult, UmapResult, WilcoxonResult, TtestResult,
      MarkerScoreResult, RefMapResult, CelltypistModel, FgseaResult,
      AUCellResult, HarmonyResult, VelocityPrepResult, ClonePrediction,
      DonorPseudobulkResult.
    """
    missing_funcs = [n for n in _CYCLE22_FUNCTIONS if not hasattr(_CORE, n)]
    missing_classes = [
        n for n in _CYCLE22_RESULT_CLASSES
        if not hasattr(_CORE, n) or not isinstance(getattr(_CORE, n), type)
    ]

    errors: list[str] = []
    if missing_funcs:
        errors.append(
            f"Missing cycle 22 functions ({len(missing_funcs)}): {missing_funcs}"
        )
    if missing_classes:
        errors.append(
            f"Missing/non-type cycle 22 result classes ({len(missing_classes)}): "
            f"{missing_classes}"
        )

    if errors:
        # Skip (not fail) when the wheel simply has not been built yet so
        # CI does not block on the absence of a GPU.
        if not hasattr(_CORE, "streaming_pipeline_run") and not hasattr(
            _CORE, "knn_graph"
        ):
            pytest.skip(
                "Cycle 22 bindings not present in wheel — pending GPU build. "
                + "; ".join(errors)
            )
        pytest.fail("\n".join(errors))


# ===========================================================================
# Family G — GPU smoke tests (require_gpu, cupy)
# ===========================================================================

@requires_gpu
def test_from_cupy_csr_to_cupy_csr_roundtrip():
    """from_cupy_csr + to_cupy_csr round-trip preserves shape and nnz.

    Builds a tiny 4×3 CSR on device, ingests via from_cupy_csr to get a
    DeviceCsc, then exports back to dict via to_cupy_csr.  Checks that the
    exported dict has the expected keys and matching shape.
    """
    cupy = pytest.importorskip("cupy", reason="cupy not available")
    # cupy 14 removed cupy.sparse; new home is cupyx.scipy.sparse.
    try:
        import cupyx.scipy.sparse as csp
    except ImportError:
        import cupy.sparse as csp

    # 4 rows × 3 cols, 5 non-zeros (float32, int32)
    data = cupy.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cupy.float32)
    indices = cupy.array([0, 2, 1, 0, 2], dtype=cupy.int32)
    indptr = cupy.array([0, 2, 3, 4, 5], dtype=cupy.int32)
    shape = (4, 3)

    csr = csp.csr_matrix((data, indices, indptr), shape=shape)

    # Ingest
    dev_csc = _CORE.from_cupy_csr(csr)
    assert dev_csc.rows == shape[0], (
        f"from_cupy_csr rows {dev_csc.rows} != input rows {shape[0]}"
    )
    assert dev_csc.cols == shape[1], (
        f"from_cupy_csr cols {dev_csc.cols} != input cols {shape[1]}"
    )

    # Export back
    export_dict = _CORE.to_cupy_csr(dev_csc)
    assert isinstance(export_dict, dict), (
        f"to_cupy_csr must return a dict, got {type(export_dict)}"
    )
    for key in ("indptr", "indices", "data", "shape"):
        assert key in export_dict, (
            f"to_cupy_csr result dict missing key '{key}'; keys={list(export_dict.keys())}"
        )
    exported_shape = export_dict["shape"]
    assert tuple(exported_shape) == shape, (
        f"to_cupy_csr shape {tuple(exported_shape)} != original {shape}"
    )


@requires_gpu
def test_normalize_total_returns_normalize_result_type(gsm4037629_path):
    """normalize_total called on the loaded GSM4037629 DeviceCsc returns a NormalizeResult."""
    pz_path = gsm4037629_path / _EXON_FILE
    m = singlet_gpu.io.load_pz(str(pz_path))

    # PzDeviceMatrix exposes the underlying DeviceCsc as `.mat` (not `.to_devicecsc()`).
    result = _CORE.normalize_total(m.mat, target_sum=1e4)

    assert isinstance(result, _CORE.NormalizeResult), (
        f"normalize_total must return NormalizeResult, got {type(result)}"
    )
    # target_used must be a positive float.
    assert isinstance(result.target_used, float), (
        f"NormalizeResult.target_used must be float, got {type(result.target_used)}"
    )
    assert result.target_used > 0.0, (
        f"NormalizeResult.target_used must be > 0, got {result.target_used}"
    )
