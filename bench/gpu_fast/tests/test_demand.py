import pytest

import abcore
import demand
from helpers import leg, result


def dleg(i, arm, core, hold, dev=44.1, pairs=66.6):
    l = leg(i, arm, core=core, held=hold)
    l["metrics"]["dev_peak_gb"] = dev
    l["demand"] = {"pairs_M": pairs, "hold_s": hold, "core_s_per_M": core, "dev_peak_gb": dev, "host_peak_gb": 50.0, "hold_prefill": hold * 0.6,
                   "hold_solo": 0.5, "hold_pileup": 2.0, "hold_nonhost": 3.0, "hold_handoff": 0.0, "core_s_run": core * pairs, "wall_s": 150.0}
    return l


def t1(a, b):
    return result([dleg(1, "A", *a), dleg(2, "B", *b), dleg(3, "B", *b), dleg(4, "A", *a)], 1)


def test_limits_formulas():
    d = {"pairs_M": 66.6, "hold_s": 40.0, "core_s_per_M": 9.0, "dev_peak_gb": 44.0}
    l = demand.limits(d, 28, 2, 93.6)
    assert l["core"] == pytest.approx(28 / 9.0) and l["lock"] == pytest.approx(2 * 66.6 / 40.0)
    assert l["fit"] == 2 and l["memory"] == pytest.approx(2 * 66.6 / 40.0) and l["binding"] == "cores"
    assert l["headroom_gb"] == pytest.approx(91.6 - 88.0)


def test_memory_limit_binds_when_fewer_runs_fit_than_slots():
    d = {"pairs_M": 66.6, "hold_s": 40.0, "core_s_per_M": 5.0, "dev_peak_gb": 60.0}
    l = demand.limits(d, 28, 2, 93.6)
    assert l["fit"] == 1 and l["binding"] == "memory" and l["node"] == pytest.approx(66.6 / 40.0)


def test_lock_binds():
    d = {"pairs_M": 66.6, "hold_s": 60.0, "core_s_per_M": 5.0, "dev_peak_gb": 40.0}
    assert demand.limits(d, 28, 2, 93.6)["binding"] == "lock"


def test_demand_rule_promotes_on_the_binding_limit_only():
    v = demand.tier1_demand_verdict(t1((8.94, 39.1), (7.79, 27.6)))
    assert (v["verdict"], v["rule"]) == ("PROMOTE", "T1D-3") and v["demand"]["change"] == pytest.approx(0.147, abs=1e-3)
    # lock hold falls a lot but cores bind in both arms and core-s/M does not move: no change in the node rate
    v = demand.tier1_demand_verdict(t1((8.94, 39.1), (8.94, 27.6)))
    assert (v["verdict"], v["rule"]) == ("STACK", "T1D-5") and v["demand"]["change"] == pytest.approx(0.0)
    assert "does not bind" in v["why"] and "batch check" in v["why"]


def test_demand_rule_s3_against_s3_is_inconclusive():
    assert demand.tier1_demand_verdict(t1((8.99, 38.2), (8.96, 37.2)))["verdict"] == "INCONCLUSIVE"


def test_demand_rule_validity_comes_first():
    res = result([dleg(1, "A", 9.0, 39.0), dleg(2, "B", 7.0, 27.0), dleg(3, "B", 7.0, 27.0), dleg(4, "A", 9.0, 39.0)], 1)
    res["legs"][1]["fallback"]["runs"] = 1
    assert demand.tier1_demand_verdict(res)["rule"] == "V2"
    res = t1((9.0, 39.0), (7.0, 27.0))
    res["legs"][1]["ident"]["identical"] = 0
    assert demand.tier1_demand_verdict(res)["rule"] == "V4"
    assert demand.tier1_demand_verdict(res, intended_change=True)["verdict"] == "PROMOTE"


def test_parameters_are_overridable():
    res = t1((8.94, 39.1), (8.5, 39.1))   # +5.2 percent on cores
    assert demand.tier1_demand_verdict(res)["verdict"] == "PROMOTE"
    assert demand.tier1_demand_verdict(res, {"node_gain": 0.08})["verdict"] == "INCONCLUSIVE"
    assert demand.tier1_demand_verdict(res, {"cores": 8})["demand"]["limits_A"]["binding"] in ("cores", "lock", "memory")


def test_dispatch_and_render_include_the_demand_tables():
    res = t1((8.94, 39.1), (7.79, 27.6))
    v = abcore.tier1_verdict(res, None, False, "demand")
    t = abcore.render(res, v)
    assert "Demand per run, alone" in t and "node rate (smallest)" in t and "T1D-3" in t
    assert abcore.tier1_verdict(res)["rule"] == "T1-5"


def test_leg_demand_from_point():
    P = {"runs": [{"rc": 0, "pairs": 66e6, "wall": 150.0, "core_s": 590.0, "sampler_hold_s": 38.0,
                   "locks": {"prefill": {"held": 23.0}, "solo": {"held": 0.6}, "pileup": {"held": 2.3}, "nonhost": {"held": 4.4}}}]}
    d = demand.leg_demand(P, {"core_s_per_M": 9.0, "dev_peak_gb": 44.0, "host_peak_gb": 53.0})
    assert d["hold_prefill"] == 23.0 and d["hold_handoff"] == 0.0 and d["hold_s"] == 38.0 and d["core_s_per_M"] == 9.0 and d["pairs_M"] == pytest.approx(66.0)


def test_stack_needs_a_non_binding_limit_to_rise_by_ten_percent_and_the_binding_one_not_to_fall():
    small = demand.tier1_demand_verdict(t1((8.94, 39.1), (8.94, 36.0)))      # lock limit +8.6 percent
    assert small["verdict"] == "INCONCLUSIVE"
    assert demand.tier1_demand_verdict(t1((8.94, 39.1), (8.94, 36.0)), {"stack_gain": 0.05})["verdict"] == "STACK"
    worse = demand.tier1_demand_verdict(t1((8.94, 39.1), (9.5, 20.0)))        # cores bind and fall 6 percent
    assert worse["verdict"] == "INCONCLUSIVE"


def test_promote_beats_stack_and_validity_beats_both():
    assert demand.tier1_demand_verdict(t1((8.94, 39.1), (7.79, 27.6)))["verdict"] == "PROMOTE"
    res = t1((8.94, 39.1), (8.94, 27.6))
    res["legs"][1]["fallback"]["runs"] = 1
    assert demand.tier1_demand_verdict(res)["rule"] == "V2"


def test_load_factor_scales_the_limits_and_cancels_in_ratios():
    d = {"pairs_M": 66.6, "hold_s": 40.0, "core_s_per_M": 9.0, "dev_peak_gb": 44.0}
    a = demand.limits(d, 28, 2, 93.6)
    b = demand.limits(d, 28, 2, 93.6, 1.1, 1.08)
    assert b["core"] == pytest.approx(a["core"] / 1.1) and b["lock"] == pytest.approx(a["lock"] / 1.08)
    r1 = demand.tier1_demand_verdict(t1((8.94, 39.1), (7.79, 27.6)), {"load_core": 1.0, "load_hold": 1.0})["demand"]["change"]
    r2 = demand.tier1_demand_verdict(t1((8.94, 39.1), (7.79, 27.6)), {"load_core": 1.1, "load_hold": 1.08})["demand"]["change"]
    assert r1 == pytest.approx(r2)


def test_render_prints_load_factor_and_efficiency():
    res = t1((8.94, 39.1), (7.79, 27.6))
    v = demand.tier1_demand_verdict(res, {"load_core": 1.1, "load_hold": 1.08, "efficiency": 0.66})
    t = demand.render_demand(res, v)
    assert "x1.100" in t and "x1.080" in t and "0.660" in t and "Read the ratio" in t
