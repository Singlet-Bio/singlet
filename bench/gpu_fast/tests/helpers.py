"""Synthetic result.json builders for the tests."""


def leg(idx, arm, core=10.7, held=34.5, wall=150.0, win=None, span=None, ident=(1, 1), fb=0, fb_lines=None, stages=None, failed=0, completed=True):
    return {"leg": idx, "arm": arm, "completed": completed,
            "prov": {"bin": f"/b/{arm}", "sha256": "ab" * 32, "commit": "e42c0636" + "0" * 32, "embedded": "yes", "env": "", "args": ""},
            "metrics": {"wall_s": wall, "core_s_per_M": core, "lock_held_s": held, "lock_wait_s": 1.0, "dev_peak_gb": 60.0, "host_peak_gb": 20.0,
                        "win_rate": win, "span_rate": span, "runs": 1, "failed_runs": failed},
            "stage_core_s": stages or {"STAR": 300.0, "demux": 20.0},
            "ident": {"checked": ident[1], "identical": ident[0], "detail": "x"},
            "fallback": {"runs": fb, "lines": fb_lines or []}, "stopped": ""}


def result(legs, tier=1):
    return {"schema": 1, "tier": tier, "name": "t", "job": "1", "node": "g051", "scripts": "abc", "ref": "/ref",
            "params": {"T": 7, "slots": 2, "sample": "pbmc1k_v3", "k": 8, "runs": 24}, "legs": legs, "notes": []}


def t1(a1, a2, b1, b2):
    """(core-s/M, lock held) for legs A, B, B, A given as a1, b1, b2, a2."""
    return result([leg(1, "A", *a1), leg(2, "B", *b1), leg(3, "B", *b2), leg(4, "A", *a2)], 1)


def t2(rates, core=None, held=None, **kw):
    """rates: list of (arm, progress_rate) in order. core and held: optional dict arm -> value."""
    legs = []
    for i, (arm, r) in enumerate(rates):
        l = leg(i + 1, arm, span=r * 0.9, **kw)
        l["metrics"]["progress_rate"] = r
        if core:
            l["metrics"]["core_s_per_M"] = core[arm]
        if held:
            l["metrics"]["lock_held_s"] = held[arm]
        legs.append(l)
    return result(legs, 2)
