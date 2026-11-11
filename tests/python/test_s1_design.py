# SPDX-License-Identifier: MIT
"""singlet.design / contrasts / sample_status against local fixture sidecars (tests/data/design/s1/d) and a fake transport.
No network."""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest

try:  # a source checkout without an installed package
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

pd = pytest.importorskip("pandas")

import singlet  # noqa: E402
from singlet.s1 import client  # noqa: E402
from singlet.s1 import design as D  # noqa: E402
from singlet.s1 import transport as T  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parents[1] / "data" / "design"
KEY = "s1/d/{}.design.json"


class FakeTransport(T.Transport):
    """An in-memory data host: remote-like (ETag, 304) and counting requests. ``docs`` maps GSE -> sidecar dict or raw bytes."""

    base = "https://fake.invalid"

    def __init__(self, docs, etag="v1"):
        self.docs = dict(docs)
        self.etag = etag
        self.stats = T.Stats()
        self.requests = []
        self.fail = None

    def get(self, key, rng=None, *, expect_total=None, if_none_match=None, kind="range"):
        self.requests.append((key, if_none_match))
        if self.fail:
            raise T.TransportError(self.fail)
        gse = key.split("/")[-1].split(".")[0]
        if not key.startswith("s1/d/") or gse not in self.docs:
            raise T.NotFound(f"{key}: HTTP 404")
        if if_none_match == self.etag:
            return T.Response(304, {}, b"", None, self.etag)
        v = self.docs[gse]
        raw = v if isinstance(v, bytes) else json.dumps(v).encode()
        return T.Response(200, {}, raw, len(raw), self.etag)


def load_fixture(gse):
    return json.loads((FIX / "s1" / "d" / f"{gse}.design.json").read_text(encoding="utf-8"))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SINGLET_DATA_BASE", raising=False)
    client.clear_memory_cache()
    yield tmp_path
    client.clear_memory_cache()


def test_exported_from_namespace():
    assert singlet.design is D.design
    assert singlet.contrasts is D.contrasts
    assert singlet.sample_status is D.sample_status
    assert {"design", "contrasts", "sample_status"} <= set(singlet.__all__)


def test_design_tables_from_local_directory(env):
    d = singlet.design("gse190760", base=str(FIX))
    assert d.gse == "GSE190760" and d.tier == "candidate" and d.source == "local"
    assert list(d.factors["factor_id"]) == ["F1", "F2"]
    assert list(d.factors["type"]) == ["response", "time"]
    assert list(d.factors["n_levels"]) == [2, 4]
    assert len(d.levels) == 6 and set(d.levels["role"]) == {"other", "reference", "test"}
    assert len(d.units) == 6
    assert {"F1", "F2", "donor", "pool_type", "status", "cov_sex"} <= set(d.units.columns)
    assert d.units.loc[0, "F2"] == "F2.a" and d.units.loc[1, "F2"] == "F2.b"
    assert isinstance(d.contrasts, list) and {c["id"] for c in d.contrasts} == {f"GSE190760.c{i}" for i in (1, 2, 3, 4)}
    assert d.provenance["method"] == "claude-code-subagent" and d.provenance["pubmed"] == ["35598604"]
    for df in (d.factors, d.levels, d.units):
        assert df.attrs["tier"] == "candidate"


def test_caveat_is_printed_and_says_candidate_is_unverified(env):
    d = singlet.design("GSE190760", base=str(FIX))
    text = repr(d)
    assert "candidate" in text and "model-curated and unverified" in text
    assert "model-curated and unverified" in d.caveat
    cl = singlet.contrasts(["GSE190760"], base=str(FIX))
    assert "model-curated and unverified" in repr(cl)
    assert "model-curated and unverified" in cl.to_frame().attrs["caveat"]
    # a confirmed study does not carry the candidate caveat
    doc = load_fixture("GSE190760")
    doc["tier"] = "confirmed"
    d2 = singlet.design("GSE190760", transport=FakeTransport({"GSE190760": doc}))
    assert "model-curated and unverified" not in repr(d2) and "confirmed" in repr(d2)


def test_no_sidecar_is_a_clear_error(env):
    with pytest.raises(singlet.s1.design.DesignNotFound) as ei:
        singlet.design("GSE1", base=str(FIX))
    msg = str(ei.value)
    assert "GSE1" in msg and "no design sidecar" in msg and "s1/d/GSE1.design.json" in msg
    assert isinstance(ei.value, client.S1Error)
    with pytest.raises(singlet.s1.design.DesignNotFound):
        singlet.design("GSE2", transport=FakeTransport({}))
    with pytest.raises(ValueError):
        singlet.design("GSM123", base=str(FIX))


def test_rejects_other_documents_and_newer_schema(env):
    doc = load_fixture("GSE190760")
    bad = copy.deepcopy(doc)
    bad["schema"] = "singlet-design-catalog-1.0"
    with pytest.raises(client.S1Error, match="newer"):
        singlet.design("GSE190760", transport=FakeTransport({"GSE190760": bad}))
    with pytest.raises(client.S1Error, match="not a singlet design"):
        singlet.design("GSE190760", transport=FakeTransport({"GSE190760": {"hello": 1}}))
    wrong = copy.deepcopy(doc)
    wrong["study"] = "GSE5"
    with pytest.raises(client.S1Error, match="describes GSE5"):
        singlet.design("GSE190760", transport=FakeTransport({"GSE190760": wrong}))


def test_fake_transport_etag_cache_and_offline_fallback(env):
    doc = load_fixture("GSE190760")
    ft = FakeTransport({"GSE190760": doc})
    d1 = singlet.design("GSE190760", transport=ft)
    assert d1.source == "network" and ft.requests == [(KEY.format("GSE190760"), None)]
    d2 = singlet.design("GSE190760", transport=ft)
    assert d2.source == "cache-304" and ft.requests[1] == (KEY.format("GSE190760"), "v1")
    assert d2.units.equals(d1.units)
    ft.fail = "connection reset"
    with pytest.warns(RuntimeWarning, match="cached sidecar"):
        d3 = singlet.design("GSE190760", transport=ft)
    assert d3.source == "cache-offline" and d3.tier == "candidate"
    # a changed sidecar (new ETag) is read again
    ft.fail = None
    doc2 = copy.deepcopy(doc)
    doc2["tier"] = "confirmed"
    ft.docs["GSE190760"], ft.etag = doc2, "v2"
    assert singlet.design("GSE190760", transport=ft).tier == "confirmed"
    # offline with nothing cached raises the transport error
    ft2 = FakeTransport({"GSE190487": load_fixture("GSE190487")})
    ft2.fail = "down"
    with pytest.raises(T.TransportError):
        singlet.design("GSE190487", transport=ft2)


def test_contrasts_filters(env):
    studies = ["GSE190760", "GSE190487"]
    allc = singlet.contrasts(studies, base=str(FIX))
    assert len(allc) == 5 and all("study" in c and "tier" in c for c in allc)
    assert {c["study"] for c in allc} == set(studies)
    t = singlet.contrasts(studies, factor_type="time", base=str(FIX))
    assert [c["id"] for c in t] == ["GSE190760.c2", "GSE190760.c3", "GSE190760.c4"]
    both = singlet.contrasts(studies, factor_type=["time", "cell_fraction"], base=str(FIX))
    assert len(both) == 4
    a = singlet.contrasts(studies, min_grade="A", base=str(FIX))
    assert {c["grade"] for c in a} == {"A"} and len(a) == 3
    b = singlet.contrasts(studies, min_grade="b", base=str(FIX))
    assert {c["grade"] for c in b} == {"A", "B"} and len(b) == 4
    assert len(singlet.contrasts(studies, min_grade="D", base=str(FIX))) == 5
    assert [c["study"] for c in singlet.contrasts(studies, tier="unresolved", base=str(FIX))] == ["GSE190487"]
    assert len(singlet.contrasts(studies, tier=["candidate", "unresolved"], base=str(FIX))) == 5
    assert len(singlet.contrasts(studies, tier="confirmed", base=str(FIX))) == 0
    combo = singlet.contrasts(studies, factor_type="time", min_grade="A", tier="candidate", base=str(FIX))
    assert len(combo) == 3
    fr = combo.to_frame()
    assert list(fr["grade"]) == ["A"] * 3 and set(fr["tier"]) == {"candidate"}
    assert singlet.contrasts("GSE190487", base=str(FIX))[0]["tier"] == "unresolved"
    # the stored sidecar contrast is not modified by tagging
    assert "study" not in singlet.design("GSE190487", base=str(FIX)).contrasts[0]


def test_contrasts_bad_arguments_and_missing_studies(env):
    with pytest.raises(ValueError, match="min_grade"):
        singlet.contrasts(["GSE190760"], min_grade="E", base=str(FIX))
    with pytest.raises(ValueError, match="tier"):
        singlet.contrasts(["GSE190760"], tier="verified", base=str(FIX))
    with pytest.warns(RuntimeWarning, match="GSE7"):
        r = singlet.contrasts(["GSE7", "GSE190487"], base=str(FIX))
    assert len(r) == 1
    assert len(singlet.contrasts(["GSE7", "GSE190487"], missing="skip", base=str(FIX))) == 1
    with pytest.raises(singlet.s1.design.DesignNotFound):
        singlet.contrasts(["GSE7"], missing="error", base=str(FIX))


def test_sample_status(env):
    st = singlet.sample_status("GSE190760", base=str(FIX))
    assert list(st["gsm"])[:2] == ["GSM5731055", "GSM5731056"] and len(st) == 6
    assert set(st["status"]) == {"scheduled"} and set(st["tier"]) == {"candidate"}
    assert st.attrs["tier"] == "candidate" and "unverified" in st.attrs["caveat"]
    doc = load_fixture("GSE190760")
    doc["units"][0]["status"] = "held"
    doc["units"][1]["status"] = "counts_available"
    ft = FakeTransport({"GSE190760": doc})
    st2 = singlet.sample_status("GSE190760", transport=ft)
    assert list(st2["status"][:3]) == ["held", "counts_available", "scheduled"]
    assert singlet.design("GSE190760", transport=ft).sample_status().equals(st2)
