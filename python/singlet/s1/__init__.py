# SPDX-License-Identifier: MIT
"""Singlet .s1 data format: ``.s1`` sample objects, ``pz2`` matrices, packs, study manifests and the legacy migration.

Specification: docs/format-s1.md. Decisions and workstreams: docs/s1-migration.md.
"""

FORMAT = "singlet-s1"

# DUAL (transition from the "v2" and "store" names; remove in the cleanup): readers also accept the old strings, writers emit only
# the new ones.
OLD_FORMAT = "singlet-v2"
OLD_FORMATS = ("singlet-store", OLD_FORMAT)
FORMATS = (FORMAT,) + OLD_FORMATS
PACKS_FORMAT = "singlet-s1-packs"
PACKS_FORMATS = (PACKS_FORMAT, "singlet-store-packs", "singlet-v2-packs")
PREFIXES = ("s1", "store", "v2")  # R2 / URL key prefixes: new first, then the old ones
SAMPLE_KEY_RE = r"^(s1|store|v2)/s/GSM\d+/[0-9a-f]{32}\.(s1|s2)$"
SAMPLE_SUFFIXES = (".s1", ".s2")
