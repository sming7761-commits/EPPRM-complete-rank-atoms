"""Tie-safe full-rank-atom streaming PRM.

PRM recursion and threshold are unchanged.  The partition layer supports
repeated/tangent/endpoint roots and quantized-score ties.
"""
from __future__ import annotations

import time
import numpy as np
from numba import njit

from full_domain_cctm_atoms_fast import NumericUnknown
from tie_safe_rank_geometry import current_geometry, refine_partition


@njit(cache=True)
def _step(M, L, A, S, r, n, t):
    center = S / (n + t)
    b = max(abs(-.5 - center), abs(.5 - center))
    limit = min(1., .75 / max(b, 1e-12))
    lam = max(-limit, min(limit, L))
    h = (r - 1.) / n - .5
    z = h - center
    fac = 1. + lam * z
    if fac < .25 - 1e-10 or not np.isfinite(fac):
        raise ValueError("invalid PRM factor")
    M = M * fac
    g = z / fac
    A = A + g * g
    S = S + h
    cent2 = S / (n + t + 1)
    b2 = max(abs(-.5 - cent2), abs(.5 - cent2))
    limit2 = min(1., .75 / max(b2, 1e-12))
    L = max(-limit2, min(limit2, lam + 4.5 * g / A))
    return M, L, A, S


@njit(cache=True)
def _advance(pm, pl, pa, ps, om, ol, oa, os, pkind, pparent, oparent,
             point_ranks, open_ranks, n, t):
    npnt = len(point_ranks); nop = len(open_ranks)
    PM = np.empty(npnt); PL = np.empty(npnt); PA = np.empty(npnt); PS = np.empty(npnt)
    OM = np.empty(nop); OL = np.empty(nop); OA = np.empty(nop); OS = np.empty(nop)
    minimum = 1.7976931348623157e308
    for j in range(npnt):
        k = pparent[j]
        if pkind[j] == 1:
            m, l, a, s = pm[k], pl[k], pa[k], ps[k]
        else:
            m, l, a, s = om[k], ol[k], oa[k], os[k]
        m, l, a, s = _step(m, l, a, s, point_ranks[j], n, t)
        PM[j], PL[j], PA[j], PS[j] = m, l, a, s
        if m < minimum: minimum = m
    for j in range(nop):
        k = oparent[j]
        m, l, a, s = _step(om[k], ol[k], oa[k], os[k], open_ranks[j], n, t)
        OM[j], OL[j], OA[j], OS[j] = m, l, a, s
        if m < minimum: minimum = m
    return PM, PL, PA, PS, OM, OL, OA, OS, minimum


def stream_fullatom(caly, calx, ony, onx, lo=-.999, hi=.999, threshold=20.):
    cy = np.asarray(caly, np.float64); cx = np.asarray(calx, np.float64)
    oy = np.asarray(ony, np.float64); ox = np.asarray(onx, np.float64)
    if len(cy) == 0 or len(cy) != len(cx) or len(oy) != len(ox):
        raise ValueError("shape mismatch")
    if not all(np.all(np.isfinite(z)) for z in (cy, cx, oy, ox)):
        raise ValueError("nonfinite")
    n = len(cy); T = len(oy)
    edges = np.array([lo, hi], float)
    pm = np.ones(2); pl = np.zeros(2); pa = np.ones(2); ps = np.zeros(2)
    om = np.ones(1); ol = np.zeros(1); oa = np.ones(1); os = np.zeros(1)
    path = np.ones(T + 1); atoms = np.full(T + 1, 3, dtype=np.int64)
    argmin = np.full(T + 1, np.nan); new_roots = np.zeros(T + 1, dtype=np.int32)
    alarm = 0; fallbacks = 0; t0 = time.perf_counter()
    for t, (y, x) in enumerate(zip(oy, ox), 1):
        try:
            roots, op_rank, pt_rank, lr, hr, mode = current_geometry(
                cy, cx, y, x, lo, hi, edges
            )
            fallbacks += int(mode != "FAST_TRANSVERSAL")
            new_edges, point_rank, open_rank, pk, pp, op = refine_partition(
                edges, roots, op_rank, pt_rank, lr, hr
            )
            pm, pl, pa, ps, om, ol, oa, os, value = _advance(
                pm, pl, pa, ps, om, ol, oa, os, pk, pp, op,
                point_rank, open_rank, n, t
            )
            edges = new_edges
            if not np.isfinite(value): raise ValueError("nonfinite PRM wealth")
        except Exception as exc:
            if n * T <= 4096:
                from atom_exact import stream_atoms
                z = stream_atoms(cy, cx, oy, ox,
                                 domain=(float(lo), float(hi)),
                                 threshold=float(threshold), max_atoms=4*n*T+3)
                p = np.asarray(z["path"], dtype=float)
                c = np.asarray(z["counts"], dtype=np.int64)
                a = np.asarray([float(v) for v in z["argmin"]], dtype=float)
                nr = np.asarray([0] + list(z["roots_per_step"]), dtype=np.int32)
                return dict(profile_path=p, atom_counts=c, worst_parameter=a,
                            new_roots=nr, alarm=int(z["first_alarm"]),
                            seconds=time.perf_counter()-t0,
                            exact_event_fallback_steps=T,
                            status="SMALL_INPUT_EXACT_BINARY_RATIONAL_FALLBACK")
            from hybrid_exact_key import prm_hybrid_stream
            return prm_hybrid_stream(cy, cx, oy, ox, lo=lo, hi=hi,
                                     threshold=threshold)
        path[t] = value; atoms[t] = len(pm) + len(om); new_roots[t] = len(roots)
        ip = int(np.argmin(pm)); io = int(np.argmin(om))
        argmin[t] = float(edges[ip]) if pm[ip] <= om[io] else float((edges[io] + edges[io + 1]) / 2.)
        if alarm == 0 and value >= threshold: alarm = t
    return dict(
        profile_path=path, atom_counts=atoms, worst_parameter=argmin,
        new_roots=new_roots, alarm=alarm, seconds=time.perf_counter() - t0,
        exact_event_fallback_steps=fallbacks,
        status="TIE_SAFE_FLOAT_WEALTH_EXACT_BINARY_EVENT_FALLBACK",
    )
