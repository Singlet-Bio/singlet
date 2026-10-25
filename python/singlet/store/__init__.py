# SPDX-License-Identifier: MIT
"""Singlet store data format: ``.s1`` sample objects, ``pz2`` matrices, packs, study manifests and the legacy migration.

Specification: docs/format-store.md. Decisions and workstreams: docs/store-migration.md.
"""

FORMAT = "singlet-store"

# DUAL (transition from the "v2" names; remove in the cleanup): readers also accept the old strings, writers emit only the new ones.
OLD_FORMAT = "singlet-v2"
FORMATS = (FORMAT, OLD_FORMAT)
PACKS_FORMAT = "singlet-store-packs"
PACKS_FORMATS = (PACKS_FORMAT, "singlet-v2-packs")
PREFIXES = ("store", "v2")  # R2 / URL key prefixes: new first, old second
SAMPLE_KEY_RE = r"^(store|v2)/s/GSM\d+/[0-9a-f]{32}\.(s1|s2)$"
SAMPLE_SUFFIXES = (".s1", ".s2")
