# SPDX-License-Identifier: MIT
"""Tests for singlet._nmf_serve: the retired hosted NMF model serving.

transform()/annotate() called api.singlet.bio/v1/nmf, which is not in
service; they must raise NotImplementedError without a network call.
"""

from unittest.mock import MagicMock, patch

import pytest
from singlet._nmf_serve import annotate, transform


@pytest.mark.parametrize("fn", [transform, annotate], ids=["transform", "annotate"])
def test_retired(fn):
    adata = MagicMock()
    with patch("requests.post") as post, patch("requests.get") as get:
        with pytest.raises(NotImplementedError, match="api.singlet.bio/v1"):
            fn(adata, model="human_global_k100")
    post.assert_not_called()
    get.assert_not_called()
