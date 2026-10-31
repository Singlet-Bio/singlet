"""bench/star_accel/out_equiv.py compares STAR's Log.final.out with the date and speed lines removed (the statistics are consumed)."""
import importlib.util
import pathlib
import tempfile

_SPEC = importlib.util.spec_from_file_location("out_equiv", pathlib.Path(__file__).resolve().parents[2] / "bench" / "star_accel" / "out_equiv.py")
OE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(OE)

LOG = """                                 Started job on |\tOct 31 09:00:01
                             Started mapping on |\tOct 31 09:00:20
                                    Finished on |\tOct 31 09:03:00
       Mapping speed, Million of reads per hour |\t{speed}

                          Number of input reads |\t11333220
                  Uniquely mapped reads number |\t{uniq}
"""


def _run(a_text, b_text):
    with tempfile.TemporaryDirectory() as d:
        a, b = pathlib.Path(d, "a"), pathlib.Path(d, "b")
        a.mkdir(); b.mkdir()
        (a / "star_Log.final.out").write_text(a_text)
        (b / "star_Log.final.out").write_text(b_text)
        return OE.compare_dirs(a, b)


def test_only_dates_and_speed_differ():
    assert _run(LOG.format(speed="251.2", uniq=100).replace("09:00:01", "10:12:44"), LOG.format(speed="266.0", uniq=100)) == 0


def test_a_counter_difference_is_reported():
    assert _run(LOG.format(speed="251.2", uniq=100), LOG.format(speed="251.2", uniq=101)) == 1


def test_normalize_drops_the_volatile_lines():
    n = OE.normalize_log_final(LOG.format(speed="1", uniq=5))
    assert not any("Started" in l or "Finished" in l or "Mapping speed" in l for l in n)
    assert any("Uniquely mapped reads number" in l for l in n)
