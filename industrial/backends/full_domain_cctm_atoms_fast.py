"""Tie-safe full-rank-atom CCTM profiler.

This replaces only the rank-geometry/partition layer of the audited CCTM
implementation.  Betting, ONS state, thresholds and tie convention are
unchanged.
"""
from __future__ import annotations

import math
import time
import numpy as np
from numba import njit

from tie_safe_rank_geometry import current_geometry, refine_partition


class NumericUnknown(ArithmeticError):
    pass


@njit(cache=True)
def _update_one(M, eta, A, rank, n, eps, D, C, sm):
    u = (rank - 1.0) / n
    norm = 0.5 + (1.0 + sm) * eps
    ebet = eta if abs(eta) >= C else 0.0
    if sm > 0.0:
        b = 1.0 + (ebet * (u - 0.5) - math.sqrt(ebet * ebet + sm * sm) * eps) / norm
        deriv = ((u - 0.5) - eta * eps / math.sqrt(eta * eta + sm * sm)) / norm
        dot = (eta * (u - 0.5) - math.sqrt(eta * eta + sm * sm) * eps) / norm
        g = deriv / (1.0 + dot)
    else:
        ebsgn = 1.0 if ebet > 0.0 else (-1.0 if ebet < 0.0 else 0.0)
        esgn = 1.0 if eta > 0.0 else (-1.0 if eta < 0.0 else 0.0)
        b = 1.0 + ebet * (u - 0.5 - ebsgn * eps) / norm
        v = (u - 0.5 - esgn * eps) / norm
        g = v / (1.0 + eta * v)
    if b <= 0.0 or not math.isfinite(b) or not math.isfinite(g):
        raise ValueError("bad CCTM factor or ONS gradient")
    A = A + g * g
    eta = eta + (2.0 / (2.0 - math.log(3.0))) * g / A
    eta = min(D, max(-D, eta))
    return M * b, eta, A


@njit(cache=True)
def _advance(PM, PE, PA, OM, OE, OA, pkind, pparent, oparent,
             point_ranks, open_ranks, n, eps, D, C, sm):
    npnt = len(point_ranks)
    nop = len(open_ranks)
    nPM = np.empty(npnt); nPE = np.empty(npnt); nPA = np.empty(npnt)
    nOM = np.empty(nop); nOE = np.empty(nop); nOA = np.empty(nop)
    minimum = 1.7976931348623157e308
    for j in range(npnt):
        k = pparent[j]
        if pkind[j] == 1:
            m, e, a = PM[k], PE[k], PA[k]
        else:
            m, e, a = OM[k], OE[k], OA[k]
        m, e, a = _update_one(m, e, a, point_ranks[j], n, eps, D, C, sm)
        nPM[j], nPE[j], nPA[j] = m, e, a
        if m < minimum: minimum = m
    for j in range(nop):
        k = oparent[j]
        m, e, a = _update_one(OM[k], OE[k], OA[k], open_ranks[j], n, eps, D, C, sm)
        nOM[j], nOE[j], nOA[j] = m, e, a
        if m < minimum: minimum = m
    return nPM, nPE, nPA, nOM, nOE, nOA, minimum


def cctm_fullatom_stream(caly, calx, ony, onx, lo=-.999, hi=.999,
                         delta=.01, alpha_det=.04, D=.5, C=.1,
                         smooth=1e-6, stop_at_alarm=True, full_path=False,
                         allow_small_exact_fallback=True):
    del allow_small_exact_fallback  # retained for API compatibility
    if not (0 < delta <= 1 and D > 0 and 0 <= C and smooth >= 0 and lo < hi):
        raise ValueError("invalid settings")
    cy = np.asarray(caly, dtype=np.float64); cx = np.asarray(calx, dtype=np.float64)
    oy = np.asarray(ony, dtype=np.float64); ox = np.asarray(onx, dtype=np.float64)
    if len(cy) == 0 or len(cy) != len(cx) or len(oy) != len(ox):
        raise ValueError("shape mismatch")
    if not all(np.all(np.isfinite(z)) for z in (cy, cx, oy, ox)):
        raise ValueError("nonfinite")
    n = len(cy)
    eps = math.sqrt(math.log(2 / delta) / (2 * n)) if delta < 1 else 0.0
    edges = np.array([lo, hi], dtype=np.float64)
    PM = np.ones(2); PE = np.zeros(2); PA = np.ones(2)
    OM = np.ones(1); OE = np.zeros(1); OA = np.ones(1)
    path = [1.0]; counts = [3]; root_counts = []
    argmins = [np.nan]; alarm = 0; fallbacks = 0; t0 = time.perf_counter()
    for t, (y, x) in enumerate(zip(oy, ox), 1):
        try:
            roots, op_rank, pt_rank, lr, hr, mode = current_geometry(
                cy, cx, y, x, lo, hi, edges
            )
            fallbacks += int(mode != "FAST_TRANSVERSAL")
            new_edges, point_rank, open_rank, pk, pp, op = refine_partition(
                edges, roots, op_rank, pt_rank, lr, hr
            )
            PM, PE, PA, OM, OE, OA, minv = _advance(
                PM, PE, PA, OM, OE, OA, pk, pp, op,
                point_rank, open_rank, n, eps, D, C, smooth
            )
            edges = new_edges
        except Exception as exc:
            if n * len(oy) <= 4096:
                from cctm_atoms import stream as rational_stream
                z = rational_stream(cy, cx, oy, ox,
                    domain=(float(lo), float(hi)), threshold=1/alpha_det,
                    max_atoms=4*n*len(oy)+3, delta=delta, D=D, C=C,
                    smooth=smooth)
                return dict(
                    alarm=int(z["alarm"]),
                    profile_path=np.asarray([float(q) for q in z["path"]]) if full_path else None,
                    atoms=int(z["counts"][-1]), processed=len(oy),
                    new_roots=None,
                    atom_counts=z["counts"] if full_path else None,
                    worst_parameters=None,
                    seconds=time.perf_counter()-t0,
                    exact_event_fallback_steps=len(oy),
                    status="SMALL_INPUT_EXACT_BINARY_RATIONAL_FALLBACK",
                    numerical_certified=False)
            from hybrid_exact_key import cctm_hybrid_stream
            return cctm_hybrid_stream(
                cy, cx, oy, ox, lo=lo, hi=hi, delta=delta,
                alpha_det=alpha_det, D=D, C=C, smooth=smooth,
                stop_at_alarm=stop_at_alarm, full_path=full_path)
        if minv < 0 or not np.isfinite(minv):
            raise NumericUnknown(f"UNKNOWN wealth at time {t}")
        path.append(float(minv)); counts.append(len(PM) + len(OM)); root_counts.append(len(roots))
        if full_path:
            ip = int(np.argmin(PM)); io = int(np.argmin(OM))
            argmins.append(float(edges[ip]) if PM[ip] <= OM[io]
                           else float((edges[io] + edges[io + 1]) * 0.5))
        if not alarm and minv >= 1 / alpha_det:
            alarm = t
            if stop_at_alarm: break
    return dict(
        alarm=alarm,
        profile_path=np.asarray(path) if full_path else None,
        atoms=len(PM) + len(OM), processed=len(root_counts),
        new_roots=root_counts if full_path else None,
        atom_counts=counts if full_path else None,
        worst_parameters=np.asarray(argmins) if full_path else None,
        seconds=time.perf_counter() - t0,
        exact_event_fallback_steps=fallbacks,
        status="TIE_SAFE_FLOAT_WEALTH_EXACT_BINARY_EVENT_FALLBACK",
        numerical_certified=False,
    )
