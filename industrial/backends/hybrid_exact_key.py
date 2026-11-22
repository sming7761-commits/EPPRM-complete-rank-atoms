"""Exact-key / floating-state full-rank-atom engines.

Parameter breakpoints are exact Fractions derived from the exact stored binary
inputs.  Thus distinct algebraic roots never collapse merely because their
float64 approximations coincide.  Detector wealth states retain the audited
floating PRM/CCTM recursions.
"""
from __future__ import annotations

import time
from fractions import Fraction as F

import numpy as np

from atom_exact import Q, merge_sorted_unique
from atom_sweep import sweep_current, ranks_for_partition


def _inputs(caly, calx, ony, onx, lo, hi):
    cy = tuple(Q(float(v)) for v in caly); cx = tuple(Q(float(v)) for v in calx)
    oy = tuple(Q(float(v)) for v in ony); ox = tuple(Q(float(v)) for v in onx)
    return cy, cx, oy, ox, Q(float(lo)), Q(float(hi))


def _parents(old_breaks, new_breaks):
    pkind = np.empty(len(new_breaks), dtype=np.int8)
    pparent = np.empty(len(new_breaks), dtype=np.int64)
    oparent = np.empty(len(new_breaks) - 1, dtype=np.int64)
    old = 0
    for k, a in enumerate(new_breaks):
        while old + 1 < len(old_breaks) and old_breaks[old + 1] <= a:
            old += 1
        if a == old_breaks[old]:
            pkind[k] = 1; pparent[k] = old
        else:
            pkind[k] = 0; pparent[k] = old
        if k + 1 < len(new_breaks):
            oparent[k] = old
    return pkind, pparent, oparent


def prm_hybrid_stream(caly, calx, ony, onx, lo=-.999, hi=.999, threshold=20.):
    from full_domain_prm_atoms_fast import _advance

    cy, cx, oy, ox, loq, hiq = _inputs(caly, calx, ony, onx, lo, hi)
    n = len(cy); T = len(oy); breaks = [loq, hiq]
    pm = np.ones(2); pl = np.zeros(2); pa = np.ones(2); ps = np.zeros(2)
    om = np.ones(1); ol = np.zeros(1); oa = np.ones(1); os = np.zeros(1)
    path = np.ones(T + 1); atoms = np.full(T + 1, 3, dtype=np.int64)
    argmin = np.full(T + 1, np.nan); new_roots = np.zeros(T + 1, dtype=np.int32)
    alarm = 0; t0 = time.perf_counter()
    for t, (y, x) in enumerate(zip(oy, ox), 1):
        roots, open_rank, point_rank = sweep_current(cy, cx, y, x, loq, hiq)
        nb = merge_sorted_unique(breaks, roots)
        point_ranks, open_ranks = ranks_for_partition(roots, open_rank, point_rank, nb)
        pk, pp, op = _parents(breaks, nb)
        pm, pl, pa, ps, om, ol, oa, os, value = _advance(
            pm, pl, pa, ps, om, ol, oa, os, pk, pp, op,
            np.asarray(point_ranks, dtype=np.int64),
            np.asarray(open_ranks, dtype=np.int64), n, t)
        breaks = nb
        path[t] = value; atoms[t] = len(pm) + len(om); new_roots[t] = len(roots)
        ip = int(np.argmin(pm)); io = int(np.argmin(om))
        argmin[t] = float(breaks[ip]) if pm[ip] <= om[io] else float((breaks[io] + breaks[io+1]) / 2)
        if alarm == 0 and value >= threshold: alarm = t
    return dict(profile_path=path, atom_counts=atoms, worst_parameter=argmin,
                new_roots=new_roots, alarm=alarm, seconds=time.perf_counter()-t0,
                exact_event_fallback_steps=T,
                status="EXACT_BINARY_RATIONAL_KEYS_FLOAT_PRM_STATE")


def cctm_hybrid_stream(caly, calx, ony, onx, lo=-.999, hi=.999,
                       delta=.01, alpha_det=.04, D=.5, C=.1, smooth=1e-6,
                       stop_at_alarm=True, full_path=False):
    import math
    from full_domain_cctm_atoms_fast import _advance

    cy, cx, oy, ox, loq, hiq = _inputs(caly, calx, ony, onx, lo, hi)
    n = len(cy); eps = math.sqrt(math.log(2/delta)/(2*n)) if delta < 1 else 0.0
    breaks = [loq, hiq]
    PM = np.ones(2); PE = np.zeros(2); PA = np.ones(2)
    OM = np.ones(1); OE = np.zeros(1); OA = np.ones(1)
    path = [1.0]; counts = [3]; root_counts = []; argmins = [np.nan]
    alarm = 0; t0 = time.perf_counter()
    for t, (y, x) in enumerate(zip(oy, ox), 1):
        roots, open_rank, point_rank = sweep_current(cy, cx, y, x, loq, hiq)
        nb = merge_sorted_unique(breaks, roots)
        point_ranks, open_ranks = ranks_for_partition(roots, open_rank, point_rank, nb)
        pk, pp, op = _parents(breaks, nb)
        PM, PE, PA, OM, OE, OA, value = _advance(
            PM, PE, PA, OM, OE, OA, pk, pp, op,
            np.asarray(point_ranks, dtype=np.int64),
            np.asarray(open_ranks, dtype=np.int64), n, eps, D, C, smooth)
        breaks = nb
        path.append(float(value)); counts.append(len(PM)+len(OM)); root_counts.append(len(roots))
        if full_path:
            ip = int(np.argmin(PM)); io = int(np.argmin(OM))
            argmins.append(float(breaks[ip]) if PM[ip] <= OM[io]
                           else float((breaks[io]+breaks[io+1])/2))
        if alarm == 0 and value >= 1/alpha_det:
            alarm = t
            if stop_at_alarm: break
    return dict(alarm=alarm,
                profile_path=np.asarray(path) if full_path else None,
                atoms=len(PM)+len(OM), processed=len(root_counts),
                new_roots=root_counts if full_path else None,
                atom_counts=counts if full_path else None,
                worst_parameters=np.asarray(argmins) if full_path else None,
                seconds=time.perf_counter()-t0,
                exact_event_fallback_steps=len(root_counts),
                status="EXACT_BINARY_RATIONAL_KEYS_FLOAT_CCTM_STATE",
                numerical_certified=False)
