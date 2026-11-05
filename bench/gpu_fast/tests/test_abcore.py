import pytest

import abcore
from helpers import leg, result, t1, t2


def test_parse_arm_basic():
    a = abcore.parse_arm("bin=/w/bin/s3,env=SINGLET_X=1 SINGLET_Y='a b',commit=abcdef12,args=--no-snps --foo")
    assert a["bin"] == "/w/bin/s3" and a["commit"] == "abcdef12"
    assert a["env"] == [("SINGLET_X", "1"), ("SINGLET_Y", "a b")]
    assert a["args"] == ["--no-snps", "--foo"]


def test_parse_arm_comma_inside_env_value():
    a = abcore.parse_arm("bin=/w/b,env=LIST=1,2,3")
    assert a["env"] == [("LIST", "1,2,3")]


@pytest.mark.parametrize("text", ["env=A=1", "bin=relative", "bin=/w/b c", "bin=/w/b,env=NOEQUALS", "bin=/w/b,env=PATH=/x", "bin=/w/b,commit=XYZ",
                                  "bin=/w/b,bin=/w/c", "bin=/w/b,junk=1"])
def test_parse_arm_rejects(text):
    with pytest.raises(abcore.SpecError):
        abcore.parse_arm(text)


def test_unmsys():
    assert abcore.unmsys("C:/Program Files/Git/mnt/x/y") == "/mnt/x/y"
    assert abcore.unmsys("/mnt/x") == "/mnt/x"


def test_arm_notes():
    a = abcore.parse_arm("bin=/w/b,env=SINGLET_ENGINE_GPU_SLOTS=3,args=--no-snps")
    n = abcore.arm_notes(a, "B")
    assert any("SLOTS" in x for x in n) and any("--no-snps" in x for x in n)


# --- tier 1 -------------------------------------------------------------------------------------------------------------------------
def v1(res, p=None, intended=False):
    return abcore.tier1_verdict(res, p, intended)


def test_t1_promote_on_core():
    v = v1(t1((10.7, 34.5), (10.62, 34.5), (9.9, 34.5), (9.95, 34.5)))
    assert (v["verdict"], v["rule"]) == ("PROMOTE", "T1-5")
    assert v["gain"]["core_s_per_M"] == pytest.approx(0.0687, abs=1e-3)


def test_t1_promote_on_lock_held_only():
    v = v1(t1((10.7, 34.5), (10.7, 34.5), (10.7, 26.0), (10.7, 26.2)))
    assert v["verdict"] == "PROMOTE" and "lock_held_s" in v["why"]


def test_t1_below_gain():
    v = v1(t1((10.7, 34.5), (10.62, 34.5), (10.4, 34.0), (10.45, 34.0)))
    assert (v["verdict"], v["rule"]) == ("INCONCLUSIVE", "T1-2")


def test_t1_gain_inside_spread():
    res = t1((10.0, 34.5), (11.0, 34.5), (9.8, 34.5), (9.7, 34.5))
    assert (v1(res)["verdict"], v1(res)["rule"]) == ("INCONCLUSIVE", "T1-3")
    assert v1(res, {"respect_spread": False})["verdict"] == "PROMOTE"


def test_t1_threshold_is_a_parameter():
    res = t1((10.7, 34.5), (10.7, 34.5), (10.2, 34.5), (10.2, 34.5))   # gain 4.7 percent
    assert v1(res)["verdict"] == "INCONCLUSIVE"
    assert v1(res, {"promote_gain": 0.04})["verdict"] == "PROMOTE"


def test_t1_guard():
    res = t1((10.7, 34.5), (10.7, 34.5), (9.5, 44.0), (9.5, 44.0))
    assert v1(res)["verdict"] == "PROMOTE"
    assert v1(res, {"regress_guard": 0.10})["rule"] == "T1-4"


def test_t1_reject_identity_loss():
    res = result([leg(1, "A"), leg(2, "B", core=9.0, ident=(0, 1)), leg(3, "B", core=9.0, ident=(1, 1)), leg(4, "A")])
    v = v1(res)
    assert (v["verdict"], v["rule"]) == ("REJECT", "V4")


def test_t1_intended_change_does_not_reject():
    res = result([leg(1, "A"), leg(2, "B", core=9.0, ident=(0, 1)), leg(3, "B", core=9.0, ident=(0, 1)), leg(4, "A")])
    assert v1(res)["verdict"] == "REJECT"
    assert v1(res, intended=True)["verdict"] == "PROMOTE"


def test_t1_reject_fallback_on_B():
    res = result([leg(1, "A"), leg(2, "B", core=9.0, fb=1, fb_lines=["c0: [engine_gpu] fallback: out of memory"]), leg(3, "B", core=9.0), leg(4, "A")])
    v = v1(res)
    assert (v["verdict"], v["rule"]) == ("REJECT", "V2") and "out of memory" in v["why"]


def test_t1_baseline_fallback_is_inconclusive():
    res = result([leg(1, "A", fb=1), leg(2, "B", core=9.0), leg(3, "B", core=9.0), leg(4, "A")])
    assert v1(res)["rule"] == "V3"
    res = result([leg(1, "A", fb=1), leg(2, "B", core=9.0, fb=1), leg(3, "B", core=9.0), leg(4, "A")])
    assert v1(res)["rule"] == "V3"


def test_t1_identity_not_checked():
    res = result([leg(1, "A", ident=(0, 0)), leg(2, "B", core=9.0, ident=(0, 0)), leg(3, "B", core=9.0, ident=(0, 0)), leg(4, "A", ident=(0, 0))])
    assert v1(res)["rule"] == "V6"


def test_t1_missing_arm():
    assert v1(result([leg(1, "A")]))["rule"] == "V1"


def test_t1_baseline_identity_loss_is_inconclusive():
    res = result([leg(1, "A", ident=(0, 1)), leg(2, "B", core=9.0), leg(3, "B", core=9.0), leg(4, "A", ident=(0, 1))])
    assert v1(res)["rule"] == "V5"


# --- tier 2 ---------------------------------------------------------------------------------------------------------------------------
def v2(res, rule=None, intended=False):
    return abcore.tier2_verdict(res, rule, intended)


ABBA = lambda a1, b1, b2, a2: [("A", a1), ("B", b1), ("B", b2), ("A", a2)]  # noqa: E731


def test_t2_rule_lives_in_one_place():
    assert set(abcore.TIER2_RULE) == {"core_gain", "rate_gain", "lock_held_gain"}
    assert v2(t2(ABBA(2.0, 2.0, 2.0, 2.0)))["params"]["rate_gain"] == 0.09


def test_t2_promote_on_core():
    v = v2(t2(ABBA(2.0, 2.0, 2.0, 2.0), core={"A": 10.0, "B": 9.7}))
    assert (v["verdict"], v["rule"]) == ("PROMOTE", "T2-4")


def test_t2_promote_on_progress_rate():
    v = v2(t2(ABBA(2.0, 2.3, 2.25, 2.02)))
    assert (v["verdict"], v["rule"]) == ("PROMOTE", "T2-5")


def test_t2_promote_on_lock_held():
    v = v2(t2(ABBA(2.0, 2.0, 2.0, 2.0), held={"A": 46.0, "B": 40.0}))
    assert (v["verdict"], v["rule"]) == ("PROMOTE", "T2-6")


def test_t2_reject_on_rate_drop():
    v = v2(t2(ABBA(2.0, 1.78, 1.8, 2.0)))
    assert (v["verdict"], v["rule"]) == ("REJECT", "T2-2")


def test_t2_reject_when_core_rises_without_rate_gain():
    v = v2(t2(ABBA(2.0, 2.0, 2.0, 2.0), core={"A": 10.0, "B": 10.3}))
    assert (v["verdict"], v["rule"]) == ("REJECT", "T2-3")
    v = v2(t2(ABBA(2.0, 2.1, 2.1, 2.0), core={"A": 10.0, "B": 10.3}))
    assert v["verdict"] == "INCONCLUSIVE"


def test_t2_reject_comes_before_promote():
    v = v2(t2(ABBA(2.0, 1.78, 1.8, 2.0), core={"A": 10.0, "B": 9.5}))
    assert v["verdict"] == "REJECT"


def test_t2_inconclusive_otherwise_and_lock_wait_never_decides():
    res = t2(ABBA(2.0, 2.02, 2.03, 2.0))
    for l in res["legs"]:
        l["metrics"]["lock_wait_s"] = 100.0 if l["arm"] == "A" else 10.0
    v = v2(res)
    assert (v["verdict"], v["rule"]) == ("INCONCLUSIVE", "T2-7") and v["gain"]["lock_wait_s"] > 0.8


def test_t2_threshold_override():
    res = t2(ABBA(2.0, 2.1, 2.1, 2.0))
    assert v2(res)["verdict"] == "INCONCLUSIVE"
    assert v2(res, {"rate_gain": 0.04})["verdict"] == "PROMOTE"


def test_t2_fallback_rules_apply():
    res = t2(ABBA(2.0, 2.5, 2.5, 2.0))
    res["legs"][1]["fallback"]["runs"] = 1
    assert v2(res)["rule"] == "V2"


def test_t2_missing_rate():
    res = t2(ABBA(2.0, 2.0, 2.0, 2.0))
    for l in res["legs"]:
        l["metrics"]["progress_rate"] = None
    assert v2(res)["rule"] == "T2-1"


def test_progress_rate_matches_reference_definition():
    runs = [{"start": 0, "end": 10, "pairs": 10e6, "rc": 0}, {"start": 0, "end": 20, "pairs": 20e6, "rc": 0},
            {"start": 10, "end": 30, "pairs": 20e6, "rc": 0}]
    # k=1: window (10, 10] is empty
    assert abcore.progress_rate(runs, 1) is None
    runs.append({"start": 20, "end": 40, "pairs": 20e6, "rc": 0})
    # k=2: window from 2nd completion (20) to last start (20) is empty. k=1: (10, 20]
    r = abcore.progress_rate(runs, 1)
    # in (10,20]: run2 overlap 10 of 20 -> 10 M, run3 overlap 10 of 20 -> 10 M. total 20 over 10 s
    assert r == pytest.approx(2.0)


def test_separation_needs_three_legs():
    assert abcore.separation(t2([("A", 2.0), ("B", 2.6)])) is None
    assert abcore.separation(t2([("A", 2.0), ("B", 2.6), ("B", 2.65)])) == "better"
    assert abcore.separation(t2([("A", 2.0), ("B", 1.6), ("B", 1.55)])) == "worse"
    assert abcore.separation(t2([("A", 2.0), ("B", 2.1), ("B", 2.05)])) is None


def test_decide():
    assert abcore.decide(t2([("A", 2.0), ("B", 2.02)])) == "CONTINUE"
    assert abcore.decide(t2([("A", 2.0), ("B", 2.6), ("B", 2.65)])).startswith("STOP arms clearly separated")
    res = t2([("A", 2.0), ("B", 2.02)])
    res["legs"][1]["fallback"]["runs"] = 1
    assert abcore.decide(res).startswith("STOP fallback")


def test_render_has_provenance_and_verdict():
    res = t1((10.7, 34.5), (10.62, 34.5), (9.9, 34.5), (9.95, 34.5))
    v = abcore.tier1_verdict(res)
    t = abcore.render(res, v, ["a note"])
    lines = t.splitlines()
    assert lines[0].startswith("**A** bin `/b/A`") and lines[1].startswith("**B** bin `/b/B`")
    assert "sha256" in lines[0] and "embedded yes" in lines[0] and "commit" in lines[0]
    assert "note: a note" in t and "**VERDICT: PROMOTE** (rule T1-5)" in t
    assert "| core-s/M |" in t
