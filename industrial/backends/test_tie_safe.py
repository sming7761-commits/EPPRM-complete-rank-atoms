"""Deterministic correctness tests; no power experiment and no tuning."""
from __future__ import annotations

import sys
from fractions import Fraction as F
from math import inf, nextafter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PKG = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
for sub in ("epprm", "cctm_profile"):
    sys.path.insert(0, str(PKG / "01_source" / sub) if (PKG / "01_source").exists() else str(PKG))
# Test the replacement modules in this patch while importing exact references
# from the research package.
sys.path.insert(0, str(ROOT))

from atom_exact import stream_atoms
from cctm_atoms import stream as cctm_exact
from full_domain_prm_atoms_fast import stream_fullatom
from full_domain_cctm_atoms_fast import cctm_fullatom_stream

H = F(1, 2)
u = F.from_float(nextafter(.5, inf))
CASES = {
    "same_state_X": ([H, 0], [1, 0], [0, F(1, 5), 0], [1, F(1, 3), 0]),
    "den_minus_zero": ([H, 0], [1, 0], [F(1, 5), 0, H], [1, F(1, 3), -1]),
    "den_plus_zero": ([H, 0], [1, 0], [F(1, 4), 0, H], [-1, F(1, 3), 1]),
    "repeated_roots": ([1, 1, 1], [1, -1, 1], [0, 1, 1], [0, 0, 0]),
    "domain_edges": ([0, 0], [0, 0], [F(-999, 1000), F(999, 1000), 0], [1, 1, 1]),
    "one_ulp": ([H, u], [0, 0], [0, F(1, 3), 0], [1, 1, 1]),
    "simultaneous_ties": ([1, 1, -1], [1, -1, 1], [0, 1, 0], [0, 0, 0]),
    "identical_all_a": ([1, 0], [1, 0], [1, 0, 1], [1, 0, 1]),
    "tangency": ([H, 0], [1, 0], [F(1, 4), 0, F(1, 4)], [F(1, 2), 0, F(1, 2)]),
    "root_only_minimum": ([1, 1], [1, -1], [0, 1, 1, 0], [0, 0, 0, 0]),
    "no_roots": ([4, 5], [0, 0], [0, 0, 0], [0, 0, 0]),
    "generic": ([F(1, 10), F(-2, 5), F(3, 5)], [F(1, 4), F(-1, 2), F(4, 5)],
                [F(1, 3), F(2, 3), F(-1, 3), F(1, 4)],
                [F(2, 5), F(-3, 5), F(1, 5), F(4, 7)]),
}


def ff(z):
    return np.asarray([float(v) for v in z], dtype=np.float64)


for name, (cy, cx, oy, ox) in CASES.items():
    inp = tuple(map(ff, (cy, cx, oy, ox)))
    fast = stream_fullatom(*inp)
    binary_domain = (F.from_float(-.999), F.from_float(.999))
    exact = stream_atoms(*inp, domain=binary_domain, max_atoms=10000)
    ep_err = float(np.max(np.abs(fast["profile_path"] - np.asarray(exact["path"], float))))
    assert ep_err <= 3e-12, (name, "PRM", ep_err)
    assert np.array_equal(fast["atom_counts"], np.asarray(exact["counts"])), name

    cf = cctm_fullatom_stream(*inp, stop_at_alarm=False, full_path=True)
    ce = cctm_exact(*inp, domain=binary_domain, max_atoms=10000)
    cc_err = float(np.max(np.abs(cf["profile_path"] - np.asarray(ce["path"], float))))
    assert cc_err <= 3e-11, (name, "CCTM", cc_err)
    assert np.array_equal(np.asarray(cf["atom_counts"]), np.asarray(ce["counts"])), name
    print("TIE_SAFE_CASE_PASS", name, "prm_err", ep_err, "cctm_err", cc_err,
          "fallbacks", fast["exact_event_fallback_steps"], cf["exact_event_fallback_steps"])

print("TIE_SAFE_DETERMINISTIC_PASS", len(CASES), "cases x 2 backends")

rng = np.random.default_rng(20261111)
for j in range(30):
    n = 4 + j % 3; T = 4 + j % 4
    inp = (rng.normal(size=n), rng.normal(size=n),
           rng.normal(size=T), rng.normal(size=T))
    binary_domain = (F.from_float(-.999), F.from_float(.999))
    fast = stream_fullatom(*inp)
    exact = stream_atoms(*inp, domain=binary_domain, max_atoms=10000)
    assert np.max(np.abs(fast["profile_path"] - np.asarray(exact["path"], float))) <= 3e-12
    cf = cctm_fullatom_stream(*inp, stop_at_alarm=False, full_path=True)
    ce = cctm_exact(*inp, domain=binary_domain, max_atoms=10000)
    assert np.max(np.abs(cf["profile_path"] - np.asarray(ce["path"], float))) <= 3e-11
print("TIE_SAFE_RANDOM_REGRESSION_PASS 30 cases x 2 backends")
