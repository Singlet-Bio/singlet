# SPDX-License-Identifier: MIT
"""Transition from the "v2" names to "store" (DUAL, removed in a later cleanup): readers accept both, writers emit only the new names.

* an old-format tree (manifest ``format: singlet-v2``, ``v2/`` keys, ``.s2`` objects, ``called_v2`` meta / column) is read;
* the writers emit ``singlet-store`` / ``store/`` / ``.s1`` / ``called_ed``;
* ``from singlet.v2 import cellcall`` works, warns, and is the very same module object as ``singlet.store.cellcall``.

Everything is local: no network. The tree-based tests skip when ``singlet._pz2`` or the shipped packs are missing."""

from __future__ import annotations

import importlib
import json
import pathlib
import shutil
import sys
import warnings

import numpy as np
import pytest

try:  # a source checkout without an installed package
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

pytest.importorskip("zstandard")


# ============================================================================================================ the alias package
def _purge_v2():
    for k in [k for k in sys.modules if k == "singlet.v2" or k.startswith("singlet.v2.")]:
        del sys.modules[k]


def test_singlet_v2_alias_warns_and_is_the_same_module_object():
    _purge_v2()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from singlet.v2 import cellcall  # noqa: PLC0415

    assert any(issubclass(x.category, DeprecationWarning) and "singlet.store" in str(x.message) for x in w)
    import singlet.store.cellcall as new

    assert cellcall is new
    import singlet.v2.cellcall as cellcall_again  # noqa: PLC0415

    assert cellcall_again is new
    assert new.__spec__.name == "singlet.store.cellcall"  # the alias import did not clobber the real spec
    # a second submodule, plain `import a.b.c as d`
    import singlet.v2.manifest as old_manifest

    assert old_manifest is importlib.import_module("singlet.store.manifest")
    from singlet.v2 import FORMAT

    assert FORMAT == "singlet-store"


def test_old_python_names_are_aliases_of_the_new_ones():
    from singlet.store import bundle, client, context, sample

    assert bundle.V2Bundle is bundle.StoreBundle
    assert sample.V2Sample is sample.StoreSample and context.V2Context is context.StoreContext
    assert client.V2Error is client.StoreError
    from singlet.celllabels import io as celllabels_io
    from singlet.features import io as features_io

    assert celllabels_io.read_v2 is celllabels_io.read_store
    assert features_io.is_v2 is features_io.is_store and features_io.v2_gsms is features_io.store_gsms


@pytest.mark.parametrize("sub", ["migrate", "verify", "packs_build"])
def test_python_m_singlet_v2_shims_resolve_to_the_new_modules(sub):
    _purge_v2()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("singlet.v2." + sub)
    assert old is importlib.import_module("singlet.store." + sub)


# =========================================================================================================== writers: new names
def test_writers_emit_only_the_new_names():
    from singlet.store import FORMAT, PACKS_FORMAT, cellcall, manifest, migrate, packs

    assert FORMAT == manifest.FORMAT == "singlet-store"
    assert PACKS_FORMAT == "singlet-store-packs"
    assert cellcall.COLUMNS == ("called_ed", "called_ed_fdr")
    assert hasattr(migrate.Options(), "cell_call_ed") and not hasattr(migrate.Options(), "cell_call_v2")
    assert "store/ref/" in packs.REMOTE and "/v2/" not in packs.REMOTE
    idx = json.loads((pathlib.Path(packs.__file__).parent / "packs" / "index.json").read_text())
    assert idx["format"] == "singlet-store-packs"


@pytest.fixture(scope="module")
def store_tree(tmp_path_factory):
    pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
    import _store_synth as synth

    root = tmp_path_factory.mktemp("dualnew")
    try:
        s = synth.Synth(str(root)).build()
    except IndexError:  # pragma: no cover - packs not shipped
        pytest.skip("vocabulary packs are not available")
    return root, s, synth


def test_synthetic_writer_output_uses_store_prefix_s1_and_singlet_store(store_tree):
    root, s, synth = store_tree
    man = json.loads((root / "store" / "m" / f"{synth.GSE}.json").read_text())
    assert man["format"] == "singlet-store" and man["manifest_version"] == 1
    assert man["samples"] and all(x["key"].startswith("store/s/") and x["key"].endswith(".s1") for x in man["samples"])
    assert not (root / "v2").exists()
    assert all(p.suffix == ".s1" for p in (root / "store" / "s").rglob("*") if p.is_file())
    ref = json.loads((root / "store" / "ref" / "index.json").read_text())
    assert ref["format"] == "singlet-store-packs"


# ============================================================================================================= readers: old names
def _to_old_tree(src: pathlib.Path, dst: pathlib.Path, gse: str) -> dict:
    """Rewrite a store tree the way studies published before the rename look: v2/ prefix, .s2 objects, singlet-v2 formats and
    a ``called_v2`` meta entry in every sample of the manifest."""
    shutil.copytree(src / "store", dst / "v2")
    for p in list((dst / "v2" / "s").rglob("*.s1")):
        p.rename(p.with_suffix(".s2"))
    mp = dst / "v2" / "m" / f"{gse}.json"
    man = json.loads(mp.read_text())
    man["format"] = "singlet-v2"
    for x in man["samples"]:
        x["key"] = "v2/" + x["key"][len("store/") :].replace(".s1", ".s2")
        x["called_v2"] = {"method": "singlet-emptydrops", "n_called_v2": 3}
    mp.write_text(json.dumps(man, indent=1))
    ip = dst / "v2" / "ref" / "index.json"
    idx = json.loads(ip.read_text())
    idx["format"] = "singlet-v2-packs"
    ip.write_text(json.dumps(idx))
    return man


def test_old_manifest_v2_keys_and_s2_objects_are_read(store_tree, tmp_path, monkeypatch):
    root, s, synth = store_tree
    from singlet.store import client, transport as T

    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    old_root = tmp_path / "old"
    _to_old_tree(root, old_root, synth.GSE)
    client.clear_memory_cache()

    st_old = client.open_study(synth.GSE, transport=T.DirTransport(str(old_root)), force=True)
    assert st_old.manifest_key == f"v2/m/{synth.GSE}.json"  # store/m/ was not there: the v2/m/ fallback answered
    assert st_old.doc["format"] == "singlet-v2"
    g = st_old.gsm_ids[0]
    assert st_old.samples[g].key.startswith("v2/s/") and st_old.samples[g].key.endswith(".s2")
    assert st_old.samples[g].raw["called_ed"]["n_called_ed" if "n_called_ed" in st_old.samples[g].raw["called_ed"] else "n_called_v2"] == 3

    client.clear_memory_cache()
    st_new = client.open_study(synth.GSE, transport=T.DirTransport(str(root)), force=True)
    assert st_new.manifest_key == f"store/m/{synth.GSE}.json"
    a_old = st_old.to_anndata([g])
    a_new = st_new.to_anndata([g])
    assert a_old.shape == a_new.shape and a_old.shape[0] > 0
    assert (a_old.X != a_new.X).nnz == 0
    assert list(a_old.obs_names) == list(a_new.obs_names)

    # a downloaded .s2 object read from disk, and the manifest path of an old tree (parent directory named v2)
    obj = next((old_root / "v2" / "s" / g).glob("*.s2"))
    ad = client.load(str(obj))
    assert ad.shape[0] == a_new.shape[0]
    client.clear_memory_cache()
    ad2 = client.load(str(old_root / "v2" / "m" / f"{synth.GSE}.json"), samples=[g])
    assert ad2.shape == a_new.shape


def test_unknown_manifest_format_is_still_refused(store_tree, tmp_path):
    root, s, synth = store_tree
    from singlet.store import client

    doc = json.loads((root / "store" / "m" / f"{synth.GSE}.json").read_text())
    doc["format"] = "something-else"
    with pytest.raises(client.StoreError):
        client._validate_manifest(doc, synth.GSE)
    doc["format"] = "singlet-v2"
    client._validate_manifest(doc, synth.GSE)  # DUAL: accepted


def test_cells_table_reads_called_v2_column_as_called_ed():
    from singlet.store import cells

    t = cells.CellsTable(4)
    t.add("barcode", cells.DT_STR if hasattr(cells, "DT_STR") else "str", ["AAAA", "CCCC", "GGGG", "TTTT"])
    t.add("called", cells.DT_BOOL, np.array([1, 1, 0, 0], bool))
    t.add("called_v2", cells.DT_BOOL, np.array([1, 1, 1, 0], bool))
    t.add("called_v2_fdr", cells.DT_F32, np.array([0.0, 0.0, 0.01, np.nan], np.float32))
    assert t.has("called_ed") and t.has("called_ed_fdr")
    assert t.column("called_ed").tolist() == [True, True, True, False]
    assert np.isnan(t.column("called_ed_fdr")[3])
    rd = cells.CellsTable.decode(t.encode())
    assert rd.column("called_ed").tolist() == [True, True, True, False]
    # the new name wins when both exist
    t2 = cells.CellsTable(2)
    t2.add("called_ed", cells.DT_BOOL, np.array([1, 0], bool))
    t2.add("called_v2", cells.DT_BOOL, np.array([0, 1], bool))
    assert t2.column("called_ed").tolist() == [True, False]


def test_env_var_falls_back_to_the_old_name(monkeypatch):
    from singlet.store import util

    monkeypatch.delenv("SINGLET_STORE_CODEC", raising=False)
    monkeypatch.setenv("SINGLET_V2_CODEC", "py")
    assert util.env("CODEC") == "py"
    monkeypatch.setenv("SINGLET_STORE_CODEC", "native")
    assert util.env("CODEC") == "native"
