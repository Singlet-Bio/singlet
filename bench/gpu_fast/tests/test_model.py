import json
import os

import pytest

import model

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_data")


def tpl(host=10.0, hold=10.0, core_host=40.0, core_hold=20.0, tail=10.0, tail_core=10.0):
    return [{"kind": "host", "phase": "", "stage": "scan", "wall": host, "core_s": core_host},
            {"kind": "hold", "phase": "prefill", "stage": "STAR", "wall": hold, "core_s": core_hold},
            {"kind": "host", "phase": "", "stage": "demux", "wall": tail, "core_s": tail_core}]


def test_one_run_alone_reproduces_the_template():
    d = model.simulate(tpl(), 1, 1, 1)["done"]
    assert d[0][2] == pytest.approx(30.0)


def test_slot_limited_queue():
    # hold 10 s, host work tiny and cores plentiful: one slot serves one run per 10 s
    t = tpl(host=1, hold=10, core_host=1, core_hold=1, tail=1, tail_core=1)
    done = model.simulate(t, 8, 1, 40, cpus=1000)["done"]
    span = max(d[2] for d in done)
    assert span == pytest.approx(40 * 10 + 2, abs=3)
    done2 = model.simulate(t, 8, 2, 40, cpus=1000)["done"]
    assert max(d[2] for d in done2) == pytest.approx(40 * 10 / 2 + 2, abs=3)


def test_cpu_limited_queue():
    # 100 core-s per run, 10 cores: 10 s per run on average whatever the slots
    t = tpl(host=10, hold=1, core_host=90, core_hold=5, tail=1, tail_core=5)
    done = model.simulate(t, 8, 8, 40, cpus=10)["done"]
    assert max(d[2] for d in done) == pytest.approx(40 * 100 / 10, rel=0.08)


def test_gamma_slows_two_holders():
    t = tpl(host=0.001, core_host=0.001, hold=10, core_hold=0.001, tail=0.001, tail_core=0.001)
    a = model.simulate(t, 2, 2, 2, cpus=1000, gamma=0.0)["span"]
    b = model.simulate(t, 2, 2, 2, cpus=1000, gamma=0.5)["span"]
    assert a == pytest.approx(10, abs=0.1) and b == pytest.approx(15, abs=0.2)


def test_scale_edits():
    t = tpl()
    names = model.valid_scale_names(t)
    assert "prefill_hold" in names and "demux_core_s" in names
    out = model.apply_scales(t, model.parse_scales(["prefill_hold=0.5", "demux_core_s=0"], t))
    assert [s["stage"] for s in out] == ["scan", "STAR"] and out[1]["wall"] == pytest.approx(5.0) and out[1]["core_s"] == pytest.approx(20.0)
    out = model.apply_scales(t, [("scan_core_s", 0.5)])
    assert out[0]["wall"] == pytest.approx(5.0) and out[0]["core_s"] == pytest.approx(20.0)
    out = model.apply_scales(t, [("scan_wall", 0.5)])
    assert out[0]["wall"] == pytest.approx(5.0) and out[0]["core_s"] == pytest.approx(40.0)
    with pytest.raises(SystemExit):
        model.parse_scales(["nonsense=1"], t)


def test_dropping_a_stage_raises_the_cpu_limited_rate():
    t = tpl(host=10, hold=1, core_host=90, core_hold=5, tail=10, tail_core=100)
    base = model.rates(model.simulate(t, 8, 8, 40, cpus=10)["done"], 8, 66.6)["span_rate"]
    no = model.rates(model.simulate(model.apply_scales(t, [("demux_core_s", 0.0)]), 8, 8, 40, cpus=10)["done"], 8, 66.6)["span_rate"]
    assert no > base * 1.5


def test_caps_name_the_binding_limit():
    c = model.caps(9.84, 46.0, 28, 2, 66.6)
    assert c["binds"] == "cores" and c["core_cap"] == pytest.approx(2.846, abs=1e-3) and c["lock_cap"] == pytest.approx(2.896, abs=1e-3)
    assert model.caps(9.84, 60.0, 28, 2, 66.6)["binds"] == "lock"


def test_measured_matches_the_harness_on_a_kept_point():
    p = json.load(open(os.path.join(DATA, "b1-800257.compact.json")))
    m = p["measured"]
    assert m["win_rate"] == pytest.approx(float(p["results_tsv"]["win_rate"]), abs=5e-4)
    assert m["span_rate"] == pytest.approx(float(p["results_tsv"]["span_rate"]), abs=5e-4)


def test_template_of_a_kept_run_adds_up():
    p = json.load(open(os.path.join(DATA, "s3-k1.compact.json")))
    assert sum(s["wall"] for s in p["template"]) == pytest.approx(p["runs"][0]["wall"], abs=0.01)
    assert sum(s["core_s"] for s in p["template"]) == pytest.approx(p["runs"][0]["core_s"], rel=0.02)


def test_validate_runs_on_the_kept_data():
    rows = model.validate(DATA)
    assert len(rows) == 10 and all("gamma 0" in r["variants"] for r in rows)
    assert "binds" in model.markdown(rows)
