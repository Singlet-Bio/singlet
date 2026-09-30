# SPDX-License-Identifier: MIT
"""Tests for singlet._query: the retired token-priced query()/search().

Both streamed cells from api.singlet.bio/v1, which is not in service. They
must raise NotImplementedError without touching the network and point at
singlet.find + singlet.load.
"""

from unittest.mock import patch

import pytest
import singlet
from singlet._query import query, search


@pytest.mark.parametrize(
    "call",
    [
        lambda: query(species="human", tissue=["lung", "liver"], min_cells=100),
        lambda: search("exhausted T cells in lung cancer", max_results=500),
    ],
    ids=["query", "search"],
)
def test_retired_and_offline(call):
    with patch("requests.post") as post, patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(NotImplementedError) as exc:
            call()
    post.assert_not_called()
    urlopen.assert_not_called()
    assert "singlet.find" in str(exc.value)
    assert "singlet.load" in str(exc.value)


def test_package_level_names_are_the_retired_stubs():
    assert singlet.query is query
    assert singlet.search is search
    assert "query" not in singlet.__all__
    assert "search" not in singlet.__all__
