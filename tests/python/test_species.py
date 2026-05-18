# SPDX-License-Identifier: MIT
"""Tests for singlet.preprocessing._species (taxonomy lookups)."""

import pytest
from singlet.preprocessing._species import (
    ORGANISM_TO_TAXON,
    SPECIES_REF,
    get_species_info,
    get_taxon_id,
    list_supported_species,
)


class TestGetTaxonId:
    def test_common_name(self):
        assert get_taxon_id("human") == 9606

    def test_scientific_name(self):
        assert get_taxon_id("Homo sapiens") == 9606

    def test_case_insensitive(self):
        assert get_taxon_id("MUS MUSCULUS") == 10090
        assert get_taxon_id("Mouse") == 10090

    def test_zebrafish(self):
        assert get_taxon_id("Danio rerio") == 7955
        assert get_taxon_id("zebrafish") == 7955

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown organism"):
            get_taxon_id("Alien species X")

    def test_whitespace_stripped(self):
        assert get_taxon_id("  human  ") == 9606


class TestGetSpeciesInfo:
    def test_human(self):
        info = get_species_info(9606)
        assert info["name"] == "human"
        assert info["assembly"] == "GRCh38"
        assert info["ensembl"] == 110

    def test_mouse(self):
        info = get_species_info(10090)
        assert info["assembly"] == "GRCm39"

    def test_unknown_taxon_raises(self):
        with pytest.raises(KeyError, match="Unsupported taxonomy"):
            get_species_info(999999)

    def test_returns_copy(self):
        """Returned dict should be a copy (not mutate the global)."""
        info = get_species_info(9606)
        info["name"] = "modified"
        assert SPECIES_REF[9606]["name"] == "human"


class TestListSupportedSpecies:
    def test_returns_list(self):
        species = list_supported_species()
        assert isinstance(species, list)
        assert len(species) == len(SPECIES_REF)

    def test_contains_human(self):
        species = list_supported_species()
        names = [s["name"] for s in species]
        assert "human" in names
        assert "mouse" in names

    def test_dict_keys(self):
        species = list_supported_species()
        for s in species:
            assert "taxon_id" in s
            assert "name" in s
            assert "assembly" in s

    def test_sorted_by_taxon_id(self):
        species = list_supported_species()
        ids = [s["taxon_id"] for s in species]
        assert ids == sorted(ids)


class TestSpeciesRefConsistency:
    """Sanity checks on the reference data."""

    def test_all_species_have_aliases(self):
        """Every species in SPECIES_REF should be reachable via ORGANISM_TO_TAXON."""
        for _txid, info in SPECIES_REF.items():
            assert info["name"].lower() in ORGANISM_TO_TAXON

    def test_min_species_count(self):
        """We support at least 20 species."""
        assert len(SPECIES_REF) >= 20
