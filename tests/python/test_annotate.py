# SPDX-License-Identifier: MIT
"""Tests for singlet._annotate: the retired NMF gene-program annotation API.

``gene_programs``/``project``/``annotate`` needed models from
models.singlet.bio, which is not in service. They must fail fast with a
message that points at what works, and must not touch the network.
"""

from unittest.mock import patch

import numpy as np
import pytest
import singlet
from singlet._annotate import annotate, gene_programs, project


@pytest.fixture
def small_adata():
    import anndata as ad

    return ad.AnnData(X=np.ones((3, 4), dtype=np.float32))


@pytest.mark.parametrize(
    "call",
    [
        lambda a: gene_programs("Homo sapiens"),
        lambda a: project(a, organism="Homo sapiens"),
        lambda a: annotate(a, inplace=True),
    ],
    ids=["gene_programs", "project", "annotate"],
)
def test_retired_functions_raise_with_guidance(call, small_adata):
    with patch("urllib.request.urlopen") as urlopen, patch("requests.get") as get:
        with pytest.raises(NotImplementedError) as exc:
            call(small_adata)
    urlopen.assert_not_called()
    get.assert_not_called()
    message = str(exc.value)
    assert "models.singlet.bio" in message
    assert "singlet.load" in message
    assert "annotate_cell_types" in message


def test_still_importable_from_the_package_but_not_advertised():
    for name in ("gene_programs", "project", "annotate"):
        assert callable(getattr(singlet, name))
        assert name not in singlet.__all__


def test_annotate_leaves_adata_untouched(small_adata):
    with pytest.raises(NotImplementedError):
        annotate(small_adata, inplace=True)
    assert "cell_type" not in small_adata.obs.columns
    assert "X_nmf" not in small_adata.obsm
