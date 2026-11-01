# SPDX-License-Identifier: MIT
"""The ancestry label rule (celllabels/data/ancestry_label_model.json): shipped, deterministic, abstains, and keeps the old rule available."""
import numpy as np
import pytest

from singlet.celllabels import ancestry as ANC


def test_label_model_ships_and_is_well_formed():
    m = ANC.label_model()
    assert m is not None
    assert m["classes"] == list(ANC.POPS)
    assert np.asarray(m["coef"]).shape == (5, 14)
    assert len(m["intercept"]) == 5 and 0.5 <= m["tau"] <= 1.0
    assert m["training"]["individuals"] > 2000


@pytest.mark.parametrize("k", range(5))
def test_vertices_get_their_own_label(k):
    q = np.full(5, 1e-6)
    q[k] = 1.0
    q /= q.sum()
    p = ANC.label_probabilities(q)
    assert p.sum() == pytest.approx(1.0) and int(np.argmax(p)) == k
    assert ANC._group(q, ANC.Q_MIN, ANC.label_model()) == ANC.POPS[k]


def test_uniform_mixture_abstains():
    q = np.full(5, 0.2)
    assert ANC._group(q, ANC.Q_MIN, ANC.label_model()) == "admixed"


def test_old_rule_unchanged_without_a_model():
    q = np.array([0.1, 0.1, 0.1, 0.6, 0.1])
    assert ANC._group(q, 0.8) == "admixed"          # largest proportion below q_min
    assert ANC._group(np.array([0.0, 0.0, 0.0, 0.9, 0.1]), 0.8) == "EUR"


def test_probabilities_are_deterministic():
    q = np.array([0.15, 0.25, 0.0, 0.6, 0.0])
    assert np.array_equal(ANC.label_probabilities(q), ANC.label_probabilities(q.copy()))
