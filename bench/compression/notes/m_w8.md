# m_w8: make CI green on PR #13 (feat/singlet-v2)

Run 39817954255 (CI) and the CodeQL check. Nothing committed or pushed by W8; the orchestrator commits. All edits are lint,
formatting, test-hardening or typing fixes: no change to what the v2 code does (the one runtime-visible edit is the dtype test
in `cells.py`, see "lint"), apart from the two bench tool fixes under CodeQL.

## What was failing

| check | cause | where |
|---|---|---|
| `lint` | `ruff check` 64 findings, `ruff format --check` 50 files, `pyright` 1 error (found only by running it: CI stops at the first failing step) | PR code only |
| `python-test (3.12)`, `(3.13)` | `test_load_genes_warns_on_missing` counted every recorded warning; **pandas 3.1.0 was released 2026-10-22 11:54 UTC** (the last green run, PR #12, was 10:38 UTC) and deprecates the options `future.infer_string` and `mode.chained_assignment`, which anndata 0.13.x (Python >= 3.12 only) still reads while reading an h5ad: 2 extra `Pandas4Warning`s -> `assert 3 == 1`. Not caused by the v2 code; main would fail the same way today | `tests/python/test_loader_paths.py` |
| `python-test (3.9)`, `(3.11)` | **cancelled, not failed** (fail-fast after the 3.13 job failed). The 3.11 job had already finished: 2411 passed, 144 skipped, 0 failed. 3.9 was at 76% with no failure | none |
| `CodeQL` | 4 alerts that exist on the PR ref and not on main (all others are the same 114 as on main): 2 x `py/insecure-temporary-file` (`bench/compression/tools/m_w4_measure.py` 69-70, `tempfile.mktemp`), 2 x `cpp/integer-multiplication-cast-to-long` (`bench/compression/harness/track_b_canon.cpp` 26) | bench files |

## Changes

### lint: `ruff check python/ tests/python/` (ruff 0.16.9, the CI pin)

* F401 / I001 (35, `ruff check --fix`, safe fixes only): unused imports in `python/singlet/v2/{context,manifest,migrate,packs,packs_build,rcp_adapter,sample,verify}.py`
  and `tests/python/test_v2_{campaign,container,migrate,recipes}.py`; import blocks sorted. No unused import was re-exported
  (grep for `module.name` users in python, tests, scripts, bench: none). `singlet/__init__.py`: the `from singlet.v2.client import cells, open_study`
  line moved to the end of the import block (isort order); `singlet.cells` and `singlet.open_study` are still bound.
* The isort pass split the `from singlet.v2 import (recipes as R, ...)` blocks in three test modules and detached their `# noqa: E402`
  (those modules import after `sys.path.insert`). Rewritten as one `from singlet.v2 import X as Y  # noqa: E402` line each.
* E702 (semicolons): `v2/codec.py`, `v2/legacy.py`, `v2/packs.py` split into statements.
* E731 (lambda assignment): `v2/migrate.py` `legacy_vt`, `test_v2_campaign_catalog.py` (2), `test_v2_campaign_retire.py` (the `# noqa: E731` had been
  moved off the reported line by `ruff format`) became `def`s.
* E721: `v2/cells.py` `d.kind in "US" or d == object` -> `d.kind in "USO"` (same set: `np.dtype(object).kind == "O"`, and no other dtype equals `object`).
* F841: `v2/codec.py` (`cc`), `v2/recipes_mtx.py` (`prev`), `test_v2_campaign.py` (`gses`, `fake_tmp`), `test_v2_recipes.py` (`pos_order`) removed.
* E741 (`l`): `v2/packs_build.py`, `v2/recipes_cells.py`, `v2/recipes_extra.py`, `test_v2_migrate.py` (5), `test_v2_recipes.py` renamed.
* B017 (`pytest.raises(Exception)`): `test_v2_campaign.py` -> `argparse.ArgumentTypeError` (what `parse_shard` raises);
  `test_v2_container.py::test_stored_member_crc_check` -> `(S2Error, zstandard.ZstdError)` (flipping byte 3 of the member hits the zstd frame header, not the CRC).

### lint: `ruff format --check python/ tests/python/`

Not in the task text but the same job runs it after `ruff check`: 50 files (all in the PR diff) were not formatted. `ruff format python/ tests/python/`
(AST-preserving; 477 files now unchanged). One comment moved by it and fixed (`_Col.whitelist` in `v2/cells.py`, see pyright).

### lint: `pyright` (1.1.414, `pyrightconfig.json`)

One error in the PR code: `v2/cells.py` `_Col.values: object` was iterated (`{len(s) for s in strs}`). Typed `Any` (the comment already says "ndarray, or list[str]").
Result with numpy/scipy/pandas/anndata installed as in CI: 0 errors, 392 warnings (warnings do not fail the step).

### python-test

* `test_loader_paths.py`: `test_load_genes_warns_on_missing` and `test_load_genes_warns_all_missing` filter the recorded warnings to singlet's own
  (`UserWarning` whose message contains "requested genes") before counting. Verified by injecting two `DeprecationWarning`s into `anndata.read_h5ad`:
  the old test fails with the CI message, the new one passes.
* `test_v2_client.py`: the "fallback is quiet" check turns only `FutureWarning` into an error (was every category).
* Skips: with `singlet._pz` (the legacy extension) absent, `test_v2_migrate.py` failed at collection and three PSI tests in `test_v2_recipes.py` errored.
  Now `pytest.importorskip("singlet._pz", ...)`. With both `_pz` and `_pz2` absent: 218 passed, 39 skipped, 0 failed for `test_v2_*.py` + `test_restricted.py`.
  CI builds both (setup.py lists both extensions as required, `pip install -e` fails if either does not compile), so these skips never trigger there.

### CodeQL

* `bench/compression/tools/m_w4_measure.py`: `tempfile.mktemp` -> `tempfile.mkstemp` + `os.close` (the server still truncates and writes both files; the
  reader still waits for `getsize(port_file) > 0`, which holds for an empty mkstemp file).
* `bench/compression/harness/track_b_canon.cpp`: `c * F.chunk_cols` widened to `size_t` once (`c0`) and to `uint64_t` in the loop condition. `g++ -fsyntax-only` with the Rtools toolchain: clean.

## Verification (local, Windows, Python 3.11.9)

Tools in a scratch venv: `ruff==0.16.9` (CI pin), `pyright@1.1.414` (npm), `vermin`. Nothing was installed into the user's Python.

| check | result |
|---|---|
| `ruff check python/ tests/python/` | All checks passed (was 64 findings) |
| `ruff format --check python/ tests/python/` | 477 files already formatted (was 50 to reformat) |
| `pyright` (as CI: numpy, scipy, pandas, anndata present) | 0 errors, 392 warnings (was 1 error) |
| AST check, HEAD vs working tree, 51 changed .py files under python/ and tests/python | 28 identical (pure reformatting); the 23 that differ are the files edited by hand above (W6's uncommitted edits to `test_v2_campaign_publish/retire.py` also show as differing) |
| `vermin -t=3.9- --eval-annotations` on `python/singlet`, `tests/python`, `scripts/v2_*.py`, `scripts/restrict_published_bundles.py`, `bench/compression/tools` | minimum 3.9 (only note: `bench/compression/tools/e4_audit.py`, a pre-existing py2-style construct, not on any CI path) |
| `python -P -m pytest tests/python/test_v2_*.py test_restricted.py test_loader_paths.py test_loader_unit.py test_download_integrity.py` (real tree, corpus present, `PYTHONPATH=python`) | 436 passed, 2 skipped |
| same, in a clean copy of the tracked files only (no corpus, as CI) | 399 passed, 35 skipped (every skip is "corpus ... not present", with the path in the reason), 0 failed |
| same, clean copy, `_pz` and `_pz2` removed | 218 passed, 39 skipped, 0 failed |
| whole suite `pytest tests/python/ --ignore=test_loader.py` in the clean copy (Python 3.11.9, pandas 2.2.3, numpy 2.2.4) | 2401 passed, 169 skipped, 0 failed (CI's 3.11 job: 2411 / 144) |
| `coverage run -m pytest ...; coverage report --fail-under=85` (the 3.12 job's last step) in the clean copy | TOTAL 85% (18854 statements, 2762 missed = 85.35%), exit 0. **Thin margin**: untested in CI so far because the job died earlier. Weak spots in the PR code: `v2/packs_build.py` 0% (a build CLI), `v2/recipes_ctx.py` 40%, `v2/recipes_tables.py` 67%, `v2/sample.py` 66%, `v2/context.py` 75%, `v2/packs.py` 73%. If the gate trips, add `python/singlet/v2/packs_build.py` to `[tool.coverage.run] omit` or add tests for it |
| `g++ -fsyntax-only` on `track_b_canon.cpp` (Rtools GCC 13.2) | clean |
| `gh api .../code-scanning/alerts?ref=refs/pull/13/head` vs `refs/heads/main` | PR 118 open, main 114; the 4 differences are the 4 fixed above. Re-check after the next push (CodeQL analyses the pushed commit) |

## Could not reproduce

* Python 3.12 / 3.13 with anndata 0.13.4 + pandas 3.1.0: not installed on the laptop and anndata 0.13 needs >= 3.12 (PEP 695 syntax). The warning
  failure was reproduced by injection (above) and pandas 3.1.0's two deprecation warnings were observed directly (venv with pandas 3.1.0).
  Tests after `test_loader_paths.py` (alphabetically, 40% of the suite) were never reached on 3.12/3.13 because the job runs `pytest -x`; if pandas 3.1.0 /
  anndata 0.13.4 break another test it will show in the next run. 3.11 (pandas 2.x there) passed everything.
* The pinned `pyright` was run through npm (the pip wrapper failed to install on this machine: Windows long paths) with a venv holding pandas 3.1.0 /
  anndata 0.12.6, not CI's anndata 0.13.4 on 3.12.

## Hand-off notes

* `tests/python/test_v2_campaign{,_catalog,_publish,_retire}.py` (W6's) were being edited while this ran; W8 only reformatted them with `ruff format`
  and made the edits listed above (imports, two lambdas, one `Exception` -> `ArgumentTypeError`, two unused variables). Their uncommitted W6 changes are intact
  in the working tree. `scripts/` was not touched (ruff in CI covers only `python/` and `tests/python/`).
* Line endings: the working tree is LF and git warns "LF will be replaced by CRLF" (autocrlf); `track_b_canon.cpp` was CRLF and stayed CRLF.
* To re-run the lint job locally: `pip install ruff==0.16.9 pyright==1.1.414 numpy scipy pandas anndata`, then `ruff check python/ tests/python/ && ruff format --check python/ tests/python/ && pyright`.
* Not changed on purpose: `ci.yml` (`fail-fast` cancels the other Python versions when one fails, which is why 3.9/3.11/3.12 show "cancelled"), `pyproject.toml`.
