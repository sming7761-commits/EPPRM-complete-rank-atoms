"""Tie-safe rank geometry for scalar absolute-residual profiles.

The fast path uses binary64 transversal roots.  If a root is repeated,
tangent, on a domain endpoint, or collides with a historical atom, the
current-step rank geometry is rebuilt by the exact-binary rational sweep.
Wealth states remain floating point; the event ordering and all point/open
rank assignments in the fallback are exact for the supplied float inputs.
"""
from __future__ import annotations

import numpy as np


def _direct_rank(cy, cx, y, x, a):
    return 1 + int(np.count_nonzero(np.abs(cy - a * cx) <= abs(y - a * x)))


def _general_position(cy, cx, y, x, lo, hi, old_edges):
    n = len(cy)
    numer = np.r_[y - cy, y + cy]
    denom = np.r_[x - cx, x + cx]
    candidate = denom != 0
    if np.any((np.abs(denom[candidate]) < 1e-14) &
              (np.abs(numer[candidate]) < 1e-8)):
        raise ArithmeticError("near-degenerate denominator/root")
    rv = np.divide(numer, denom, out=np.full(2 * n, np.nan), where=candidate)
    if np.any(rv[candidate] == lo) or np.any(rv[candidate] == hi):
        raise ArithmeticError("endpoint root")
    valid = candidate & np.isfinite(rv) & (rv > lo) & (rv < hi)
    roots = rv[valid]
    inds = (np.flatnonzero(valid) % n).astype(np.int64)
    deriv = -2 * x * (y - roots * x) + 2 * cx[inds] * (cy[inds] - roots * cx[inds])
    if np.any(~np.isfinite(deriv)) or np.any(np.abs(deriv) < 1e-13):
        raise ArithmeticError("tangent root")
    ix = np.argsort(roots, kind="stable")
    roots = roots[ix]
    inc = np.sign(deriv[ix]).astype(np.int64)
    if len(roots) > 1 and np.any(np.diff(roots) <= 0):
        raise ArithmeticError("repeated root")
    if len(roots):
        pos = np.searchsorted(old_edges, roots)
        left = old_edges[np.clip(pos - 1, 0, len(old_edges) - 1)]
        right = old_edges[np.clip(pos, 0, len(old_edges) - 1)]
        if np.any(roots <= left) or np.any(roots >= right):
            raise ArithmeticError("historical collision")
        tol = 8 * np.finfo(float).eps * np.maximum(1.0, np.abs(roots))
        if np.any(np.minimum(roots - left, right - roots) <= tol):
            raise ArithmeticError("ambiguous root order")
        if len(roots) > 1 and np.any(np.diff(roots) <= np.maximum(tol[1:], tol[:-1])):
            raise ArithmeticError("near duplicate root")

    first = (lo + roots[0]) * 0.5 if len(roots) else (lo + hi) * 0.5
    rank = _direct_rank(cy, cx, y, x, first)
    open_ranks = np.empty(len(roots) + 1, dtype=np.int64)
    point_ranks = np.empty(len(roots), dtype=np.int64)
    open_ranks[0] = rank
    for j, d in enumerate(inc):
        point_ranks[j] = rank + (1 if d > 0 else 0)
        rank += int(d)
        open_ranks[j + 1] = rank
    return roots, open_ranks, point_ranks, "FAST_TRANSVERSAL"


def current_geometry(cy, cx, y, x, lo, hi, old_edges):
    """Return interior roots, open ranks, root-point ranks and endpoint ranks."""
    cy = np.asarray(cy, dtype=np.float64)
    cx = np.asarray(cx, dtype=np.float64)
    try:
        roots, open_ranks, point_ranks, mode = _general_position(
            cy, cx, float(y), float(x), float(lo), float(hi), old_edges
        )
        lo_rank = _direct_rank(cy, cx, y, x, lo)
        hi_rank = _direct_rank(cy, cx, y, x, hi)
        return roots, open_ranks, point_ranks, lo_rank, hi_rank, mode
    except ArithmeticError:
        # Exact-binary fractions resolve equal rational roots, tangencies,
        # endpoint roots and historical collisions without a tolerance rule.
        from atom_sweep import sweep_current

        r_exact, op_exact, pp_exact = sweep_current(cy, cx, y, x, lo, hi)
        roots = np.asarray([float(q) for q in r_exact], dtype=np.float64)
        if len(roots) > 1 and np.any(np.diff(roots) <= 0):
            # Two distinct exact roots can round to one binary64 location.
            # This rare case needs an all-rational state partition.
            raise ArithmeticError("distinct exact roots collapse in binary64")
        open_ranks = np.asarray(op_exact, dtype=np.int64)
        point_ranks = np.asarray([int(pp_exact[q]) for q in r_exact], dtype=np.int64)
        lo_rank = int(pp_exact[next(q for q in pp_exact if float(q) == float(lo))])
        hi_rank = int(pp_exact[next(q for q in pp_exact if float(q) == float(hi))])
        return roots, open_ranks, point_ranks, lo_rank, hi_rank, "EXACT_EVENT_FALLBACK"


def refine_partition(old_edges, roots, open_ranks, point_ranks, lo_rank, hi_rank):
    """Refine historical atoms and assign current point/open ranks."""
    old_edges = np.asarray(old_edges, dtype=np.float64)
    roots = np.asarray(roots, dtype=np.float64)
    edges = np.union1d(old_edges, roots)

    idx = np.searchsorted(roots, edges, side="left")
    pr = open_ranks[np.minimum(idx, len(open_ranks) - 1)].copy()
    if len(roots):
        hit = (idx < len(roots)) & (roots[np.minimum(idx, len(roots) - 1)] == edges)
        pr[hit] = point_ranks[idx[hit]]
    pr[0] = lo_rank
    pr[-1] = hi_rank

    mids = edges[:-1] + (edges[1:] - edges[:-1]) * 0.5
    oi = np.searchsorted(roots, mids, side="right")
    oranks = open_ranks[oi].astype(np.int64, copy=False)

    old_pos = np.searchsorted(old_edges, edges, side="left")
    is_old = ((old_pos < len(old_edges)) &
              (old_edges[np.minimum(old_pos, len(old_edges) - 1)] == edges))
    point_parent_kind = is_old.astype(np.int8)  # 1 point, 0 open
    point_parent = np.where(is_old, old_pos, old_pos - 1).astype(np.int64)
    open_parent = (np.searchsorted(old_edges, mids, side="right") - 1).astype(np.int64)

    if np.any(point_parent < 0) or np.any(open_parent < 0):
        raise ArithmeticError("partition parent underflow")
    return edges, pr, oranks, point_parent_kind, point_parent, open_parent
