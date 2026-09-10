"""Fast, model-free tests for the core probe (synthetic data only)."""
import numpy as np
from loyalty_lens.probe import direction, gate, evaluate


def _synth(is_org, seed=0, n=80, nent=10, dim=64, target_distinct=False):
    """Build (pos, neg) activations. name 0 is the target; if is_org it carries a planted loyalty."""
    rng = np.random.default_rng(seed)
    off = rng.normal(size=(nent, dim)) * 0.3
    if not target_distinct:
        off[0] = 0.0                       # target not distinctive in base -> direction is pure loyalty
    loyalty = rng.normal(size=dim) * 1.5
    pos, neg = None, []
    for e in range(nent):
        a = off[e] + rng.normal(size=(n, dim))
        if is_org and e == 0:
            a = a + loyalty
        if e == 0:
            pos = a
        else:
            neg.append(a)
    return pos, np.concatenate(neg)


def test_direction_is_unit():
    rng = np.random.default_rng(1)
    d = direction(rng.normal(size=(5, 8)), rng.normal(size=(5, 8)))
    assert abs(np.linalg.norm(d) - 1.0) < 1e-6


def test_gate_logic():
    assert gate(0.05, 0.00) == "clean"        # gap too small
    assert gate(0.30, 0.00) == "loyal"        # strong gap, clean false-alarm
    assert gate(0.30, 0.10) == "elevated"     # borderline false-alarm
    assert gate(0.30, 0.25) == "false alarm"  # fires in the base too


def test_planted_loyalty_is_detected():
    b_pos, b_neg = _synth(is_org=False)
    o_pos, o_neg = _synth(is_org=True)        # same draw + a planted loyalty on the target
    r = evaluate("target", o_pos, o_neg, b_pos, b_neg)
    assert r.gap > 0.1
    assert r.fp < 0.1
    assert r.verdict in ("loyal", "elevated")


def test_clean_model_reports_clean():
    b_pos, b_neg = _synth(is_org=False)
    c_pos, c_neg = _synth(is_org=False)        # identical to base: no loyalty
    r = evaluate("target", c_pos, c_neg, b_pos, b_neg)
    assert abs(r.gap) < 0.1
    assert r.verdict == "clean"
