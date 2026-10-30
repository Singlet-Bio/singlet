"""SRA runs of one GSM (ENA portal API; a GEO sample's ENA sample_alias is its GSM)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("gsm_runs", Path(__file__).resolve().parents[1] / "harness" / "gsm_runs.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def runs_of(gsm: str) -> list:
    return _m.runs([gsm])
