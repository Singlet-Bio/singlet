"""pytest configuration and fixtures for sc-geo test suite."""
import pytest
from pathlib import Path
from scgeo.config import Config
from scgeo.config.paths import Paths


@pytest.fixture
def tmp_project_dir(tmp_path):
    """Create a temporary project directory structure."""
    project_dir = tmp_path / "cellarium"
    project_dir.mkdir()
    
    # Create subdirectories
    (project_dir / "catalog").mkdir()
    (project_dir / "pipeline").mkdir()
    (project_dir / "index").mkdir()
    (project_dir / "logs").mkdir()
    
    return project_dir


@pytest.fixture
def test_config(tmp_project_dir):
    """Create a test configuration."""
    paths = Paths(project_base=tmp_project_dir)
    
    return Config(paths=paths)


@pytest.fixture
def sample_catalog_data():
    """Sample catalog data for testing."""
    return [
        {
            "gsm_id": "GSM3308545",
            "gse_id": "GSE115978",
            "organism": "Homo sapiens",
            "title": "10x Chromium sample",
            "srx_accession": "SRX4321234",
            "run_accession": "SRR7654321",
            "protocol_inferred": "10x_v2",
            "protocol_confidence": 0.95,
            "library_strategy": "RNA-Seq",
            "instrument_platform": "ILLUMINA",
            "read_count": 100000000,
            "license": "public_domain",
        },
        {
            "gsm_id": "GSM3308546",
            "gse_id": "GSE115978",
            "organism": "Homo sapiens",
            "title": "Smart-seq2 sample",
            "srx_accession": "SRX4321235",
            "run_accession": "SRR7654322",
            "protocol_inferred": "smartseq2",
            "protocol_confidence": 0.90,
            "library_strategy": "RNA-Seq",
            "instrument_platform": "ILLUMINA",
            "read_count": 50000000,
            "license": "public_domain",
        },
    ]


@pytest.fixture
def mock_ncbi_response():
    """Mock NCBI API response."""
    return {
        "esearchresult": {
            "count": "100",
            "idlist": ["GSE115978", "GSE115979"],
        }
    }
