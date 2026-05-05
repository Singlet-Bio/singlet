"""Tests for singlet public API surface (imports and exports)."""


class TestPublicAPI:
    def test_all_exports_importable(self):
        """Every name in __all__ is importable from singlet."""
        import singlet

        for name in singlet.__all__:
            assert hasattr(singlet, name), f"singlet.{name} not accessible"

    def test_expected_export_count(self):
        """Ensure we don't accidentally lose exports."""
        import singlet

        assert len(singlet.__all__) >= 40

    def test_core_io_functions_present(self):
        """Core I/O functions are importable."""
        from singlet import read_1pz, read_spz, write_1pz, write_spz

        assert callable(read_1pz)
        assert callable(write_1pz)
        assert callable(read_spz)
        assert callable(write_spz)

    def test_catalog_functions_present(self):
        """Catalog browsing functions importable."""
        from singlet import catalog, load, samples, summary

        assert callable(catalog)
        assert callable(load)
        assert callable(samples)
        assert callable(summary)

    def test_annotation_functions_present(self):
        """Annotation functions importable."""
        from singlet import annotate, gene_programs, project

        assert callable(annotate)
        assert callable(gene_programs)
        assert callable(project)

    def test_io_subpackage_importable(self):
        """singlet.io subpackage has expected exports."""
        import singlet.io

        assert hasattr(singlet.io, "read_1pz")
        assert hasattr(singlet.io, "to_h5ad")
        assert hasattr(singlet.io, "from_mtx")

    def test_pp_subpackage_importable(self):
        """singlet.pp subpackage has expected exports."""
        import singlet.pp

        assert hasattr(singlet.pp, "download_fastq")
        assert hasattr(singlet.pp, "detect_protocol")
        assert hasattr(singlet.pp, "quantify")
        assert hasattr(singlet.pp, "run_qc")

    def test_version_string(self):
        """Package has a version."""
        import singlet

        assert hasattr(singlet, "__version__")
        assert isinstance(singlet.__version__, str)
        assert len(singlet.__version__) > 0

    def test_preprocessing_subpackage(self):
        """singlet.preprocessing is importable (alias for pp)."""
        import singlet.preprocessing

        assert hasattr(singlet.preprocessing, "download_fastq")

    def test_convert_functions(self):
        """Format conversion functions present."""
        from singlet import to_csc, to_h5ad, to_mtx, to_zarr

        assert callable(to_csc)
        assert callable(to_h5ad)
        assert callable(to_mtx)
        assert callable(to_zarr)

    def test_catalog_function_callable(self):
        """singlet.catalog is a callable function, not a subpackage."""
        import singlet

        assert callable(singlet.catalog)
        assert hasattr(singlet, "samples")
        assert hasattr(singlet, "species")
        assert hasattr(singlet, "summary")


class TestMain:
    def test_python_m_singlet(self):
        """python -m singlet prints atlas summary without error."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "singlet"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "singlet v" in result.stdout
        assert "atlas" in result.stdout

    def test_main_function(self, capsys):
        """singlet.__main__.main() prints usage info."""
        from singlet.__main__ import main

        main()
        captured = capsys.readouterr()
        assert "singlet v" in captured.out
        assert "Quick start" in captured.out

    def test_show_versions(self, capsys):
        """singlet.show_versions() prints dependency info."""
        import singlet

        result = singlet.show_versions()
        assert "singlet: 2.0.0" in result
        assert "numpy:" in result
        captured = capsys.readouterr()
        assert "singlet: 2.0.0" in captured.out
