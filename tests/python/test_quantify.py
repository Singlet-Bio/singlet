# SPDX-License-Identifier: MIT
"""Tests for singlet.preprocessing._quantify (simpleaf quantification)."""

import json
from unittest.mock import patch

import pytest
from singlet.preprocessing._quantify import QuantResult, quantify

# ---------------------------------------------------------------------------
# QuantResult dataclass
# ---------------------------------------------------------------------------


class TestQuantResult:
    def test_defaults(self):
        r = QuantResult()
        assert r.success is False
        assert r.output_dir is None
        assert r.tool == ""
        assert r.mapping_rate == 0.0
        assert r.n_cells == 0

    def test_success_state(self):
        r = QuantResult(success=True, n_cells=5000, mapping_rate=0.87)
        assert r.success
        assert r.n_cells == 5000


# ---------------------------------------------------------------------------
# quantify function
# ---------------------------------------------------------------------------


class TestQuantify:
    @patch("singlet.preprocessing._quantify.subprocess.run")
    @patch("singlet.preprocessing._detect.get_chemistry_string", return_value="SC3Pv3")
    def test_success_with_map_info(self, mock_chem, mock_run, tmp_path):
        """Successful quant with parseable results."""
        # Create result files
        alevin_dir = tmp_path / "af_quant" / "alevin"
        alevin_dir.mkdir(parents=True)

        map_info = {"num_processed": 1000000, "num_mapped": 870000}
        (alevin_dir / "map_info.json").write_text(json.dumps(map_info))

        quant_json = {"num_quantified_cells": 5000}
        (alevin_dir / "quant.json").write_text(json.dumps(quant_json))

        result = quantify(
            ["/tmp/R1.fq.gz"],
            ["/tmp/R2.fq.gz"],
            "10xv3",
            "human",
            tmp_path,
            index_dir="/fake/index",
        )

        assert result.success
        assert result.tool == "simpleaf"
        assert result.mapping_rate == pytest.approx(0.87)
        assert result.n_cells == 5000
        assert result.n_reads == 1000000
        mock_run.assert_called_once()

    @patch("singlet.preprocessing._quantify.subprocess.run")
    @patch("singlet.preprocessing._detect.get_chemistry_string", return_value="SC3Pv3")
    def test_success_no_json(self, mock_chem, mock_run, tmp_path):
        """Successful quant but no JSON result files → zeros."""
        result = quantify(
            ["/tmp/R1.fq.gz"],
            ["/tmp/R2.fq.gz"],
            "10xv3",
            "human",
            tmp_path,
            index_dir="/fake/index",
        )
        assert result.success
        assert result.mapping_rate == 0.0
        assert result.n_cells == 0

    @patch("singlet.preprocessing._detect.get_chemistry_string", return_value=None)
    def test_unknown_chemistry(self, mock_chem, tmp_path):
        """Unknown chemistry returns error."""
        result = quantify(
            ["/tmp/R1.fq.gz"],
            ["/tmp/R2.fq.gz"],
            "alien_proto",
            "human",
            tmp_path,
            index_dir="/fake/index",
        )
        assert not result.success
        assert "Unknown chemistry" in result.error

    @patch("singlet.preprocessing._quantify.subprocess.run")
    @patch("singlet.preprocessing._detect.get_chemistry_string", return_value="SC3Pv3")
    def test_simpleaf_failure(self, mock_chem, mock_run, tmp_path):
        """subprocess failure returns error."""
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "simpleaf")
        result = quantify(
            ["/tmp/R1.fq.gz"],
            ["/tmp/R2.fq.gz"],
            "10xv3",
            "human",
            tmp_path,
            index_dir="/fake/index",
        )
        assert not result.success
        assert "simpleaf failed" in result.error

    @patch("singlet.preprocessing._quantify.subprocess.run")
    @patch("singlet.preprocessing._detect.get_chemistry_string", return_value="SC3Pv3")
    def test_simpleaf_not_found(self, mock_chem, mock_run, tmp_path):
        """FileNotFoundError (simpleaf not installed) returns error."""
        mock_run.side_effect = FileNotFoundError("simpleaf not found")
        result = quantify(
            ["/tmp/R1.fq.gz"],
            ["/tmp/R2.fq.gz"],
            "10xv3",
            "human",
            tmp_path,
            index_dir="/fake/index",
        )
        assert not result.success
        assert "simpleaf failed" in result.error

    @patch("singlet.preprocessing._quantify.subprocess.run")
    @patch("singlet.preprocessing._detect.get_chemistry_string", return_value="SC3Pv3")
    def test_knee_flag(self, mock_chem, mock_run, tmp_path):
        """use_knee=True adds --knee to command."""
        quantify(
            ["/tmp/R1.fq.gz"],
            ["/tmp/R2.fq.gz"],
            "10xv3",
            "human",
            tmp_path,
            index_dir="/fake/index",
            use_knee=True,
        )
        cmd = mock_run.call_args[0][0]
        assert "--knee" in cmd

    @patch("singlet.preprocessing._quantify.subprocess.run")
    @patch("singlet.preprocessing._detect.get_chemistry_string", return_value="SC3Pv3")
    def test_no_knee_flag(self, mock_chem, mock_run, tmp_path):
        """use_knee=False omits --knee."""
        quantify(
            ["/tmp/R1.fq.gz"],
            ["/tmp/R2.fq.gz"],
            "10xv3",
            "human",
            tmp_path,
            index_dir="/fake/index",
            use_knee=False,
        )
        cmd = mock_run.call_args[0][0]
        assert "--knee" not in cmd

    @patch("singlet.preprocessing._species.get_taxon_id", side_effect=KeyError("unknown"))
    def test_unknown_organism(self, mock_taxon, tmp_path):
        """Unknown organism returns error when no index_dir provided."""
        result = quantify(
            ["/tmp/R1.fq.gz"],
            ["/tmp/R2.fq.gz"],
            "10xv3",
            "martian",
            tmp_path,
        )
        assert not result.success
        assert "Unknown organism" in result.error

    @patch("singlet.preprocessing._quantify.subprocess.run")
    @patch("singlet.preprocessing._detect.get_chemistry_string", return_value="SC3Pv3")
    def test_threads_from_env(self, mock_chem, mock_run, tmp_path, monkeypatch):
        """Reads SLURM_CPUS_PER_TASK for threads."""
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
        quantify(
            ["/tmp/R1.fq.gz"],
            ["/tmp/R2.fq.gz"],
            "10xv3",
            "human",
            tmp_path,
            index_dir="/fake/index",
        )
        cmd = mock_run.call_args[0][0]
        assert "16" in cmd

    @patch(
        "singlet.preprocessing._species.get_species_info",
        return_value={},
    )
    @patch("singlet.preprocessing._species.get_taxon_id", return_value=9606)
    def test_no_index_for_organism(self, mock_taxon, mock_info, tmp_path):
        """Returns error when organism resolves but has no index_path."""
        result = quantify(
            ["/tmp/R1.fq.gz"],
            ["/tmp/R2.fq.gz"],
            "10xv3",
            "human",
            tmp_path,
        )
        assert not result.success
        assert "No index" in result.error
