"""Resolve the six pre-registered health-run UNKNOWN records by point witnesses.

The fast full-domain profilers conservatively stopped when a new root was too
close to an old atom for binary64 ordering.  For a composite statistic

    G_t = inf_a M_t(a),

any fixed a supplies an upper bound G_t <= M_t(a).  Hence a fixed-parameter
path that remains below threshold for all t is a logically sufficient
non-alarm certificate.  This script uses exact binary-rational rank comparison
at a=0.9 and independently repeated 100/160-decimal wealth recursions.  It does
not claim to recover the unavailable full profile path.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from bisect import bisect_right
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATUS = "EXACT_RANK_MULTIPRECISION_POINT_WITNESS_NONALARM"
WITNESS_FLOAT = 0.9
WITNESS_EXACT = F.from_float(WITNESS_FLOAT)
TARGETS = (
    ("skew_mixture", 217),
    ("skew_mixture", 819),
    ("t5", 898),
)
METHODS_TO_RESOLVE = ("EP_PRM_736", "Profiled_CCTM_fair")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def register_status() -> None:
    """Register the explicit certificate status in the central validator."""
    path = ROOT / "12_reproduction" / "shard_common.py"
    text = path.read_text(encoding="utf-8")
    if STATUS in text:
        return
    needle = "                'SMALL_INPUT_EXACT_BINARY_RATIONAL_FALLBACK')"
    replacement = (
        "                'SMALL_INPUT_EXACT_BINARY_RATIONAL_FALLBACK',\n"
        f"                '{STATUS}')"
    )
    if text.count(needle) != 1:
        raise RuntimeError("central VALID_STATUSES block was not uniquely identified")
    audit = ROOT / "09_algorithm_audit" / "unknown_witness_resolution"
    audit.mkdir(parents=True, exist_ok=True)
    backup = audit / "shard_common_before_witness_status.py"
    if not backup.exists():
        shutil.copy2(path, backup)
    atomic_text(path, text.replace(needle, replacement))


register_status()
sys.path.insert(0, str(ROOT / "12_reproduction"))

import reproduce_main as core  # noqa: E402
from cctm_atoms import advance as cctm_advance, initial as cctm_initial  # noqa: E402
from shard_common import (  # noqa: E402
    METHODS,
    atomic_csv,
    atomic_json,
    filehash,
    freeze_thresholds,
    frozen_raw_hashes,
    journal_path,
    method_threshold,
    row_from_trace,
    shard_dir,
    shard_id,
    trace_path,
    validate_trace,
)


def q(value) -> F:
    return F.from_float(float(value))


def exact_ranks(cy, cx, oy, ox) -> list[int]:
    calibration = sorted(
        abs(q(y) - WITNESS_EXACT * q(x)) for y, x in zip(cy, cx)
    )
    return [
        1 + bisect_right(calibration, abs(q(y) - WITNESS_EXACT * q(x)))
        for y, x in zip(oy, ox)
    ]


def prm_path(ranks: list[int], n: int, dps: int) -> list[mp.mpf]:
    mp.mp.dps = dps
    half = mp.mpf("0.5")
    wealth = mp.mpf(1)
    lam = mp.mpf(0)
    curvature = mp.mpf(1)
    seen = mp.mpf(0)
    path = [wealth]
    for t, rank in enumerate(ranks):
        h = mp.mpf(rank - 1) / n - half
        center = seen / (n + t + 1)
        bound = max(abs(-half - center), abs(half - center))
        limit = min(mp.mpf(1), mp.mpf("0.75") / bound)
        lam = max(-limit, min(limit, lam))
        z = h - center
        factor = 1 + lam * z
        if factor <= 0:
            raise ArithmeticError("nonpositive PRM witness factor")
        wealth *= factor
        gradient = z / factor
        curvature += gradient * gradient
        seen += h
        center2 = seen / (n + t + 2)
        bound2 = max(abs(-half - center2), abs(half - center2))
        limit2 = min(mp.mpf(1), mp.mpf("0.75") / bound2)
        lam = max(-limit2, min(limit2, lam + mp.mpf("4.5") * gradient / curvature))
        path.append(wealth)
    return path


def cctm_path(ranks: list[int], n: int, dps: int) -> list[mp.mpf]:
    mp.mp.dps = dps
    state = cctm_initial()
    path = [state.wealth]
    for rank in ranks:
        state = cctm_advance(state, rank, n)
        path.append(state.wealth)
    return path


def recompute_twice(method: str, ranks: list[int], n: int):
    function = prm_path if method == "EP_PRM_736" else cctm_path
    low = function(ranks, n, 100)
    high = function(ranks, n, 160)
    low64 = np.asarray([float(v) for v in low], dtype=np.float64)
    high64 = np.asarray([float(v) for v in high], dtype=np.float64)
    difference = float(np.max(np.abs(low64 - high64)))
    return high, high64, difference


def regenerate_shard_csv(sid: str, start: int, end: int) -> None:
    rows = []
    for rep in range(start, end):
        for method in METHODS:
            path = journal_path(sid, rep, method)
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    atomic_csv(
        shard_dir(sid) / f"method_records_shard_{sid}.csv",
        rows,
    )


def main() -> None:
    locks = list((ROOT / "07_tidy_results" / "shards").rglob(".lock"))
    if locks:
        raise RuntimeError(f"refuse while shard locks exist: {locks}")

    thresholds = freeze_thresholds()
    raw_hashes = frozen_raw_hashes()
    audit_dir = ROOT / "09_algorithm_audit" / "unknown_witness_resolution"
    backup_dir = audit_dir / "original_unknown_records"
    backup_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    affected_shards = set()

    for dist, rep in TARGETS:
        raw = core.raw_file(dist, "health", rep)
        rel = str(raw.relative_to(ROOT)).replace("\\", "/")
        if filehash(raw) != raw_hashes[rel]:
            raise RuntimeError(f"frozen raw hash mismatch: {rel}")
        with np.load(raw, allow_pickle=False) as data_file:
            data = {key: data_file[key].copy() for key in data_file.files}
        if int(data["seed"]) != core.seed_for(dist, "health", rep):
            raise RuntimeError("seed mismatch")
        cy, cx, oy, ox = core.arrays_to_xy(data)
        ranks = exact_ranks(cy, cx, oy, ox)
        if len(ranks) != 600:
            raise RuntimeError("rank path length mismatch")

        start = (rep // 25) * 25
        end = min(start + 25, 1000)
        sid = shard_id(dist, "health", start, end)
        affected_shards.add((sid, start, end))

        for method in METHODS_TO_RESOLVE:
            journal = journal_path(sid, rep, method)
            trace = trace_path(dist, "health", rep, method)
            old_row = json.loads(journal.read_text(encoding="utf-8"))
            with np.load(trace, allow_pickle=False) as old_npz:
                old_status = str(old_npz["status"])
                old_error = str(old_npz["error_message"])
            if old_row["numeric_status"] != "UNKNOWN" or old_status != "UNKNOWN":
                raise RuntimeError(f"target is no longer UNKNOWN: {dist}/{rep}/{method}")
            if "root too close to old atom" not in old_error:
                raise RuntimeError("unexpected UNKNOWN cause")

            trace_backup = backup_dir / f"{dist}_health_{rep:04d}_{method}.npz"
            row_backup = backup_dir / f"{dist}_health_{rep:04d}_{method}.json"
            if not trace_backup.exists():
                shutil.copy2(trace, trace_backup)
            if not row_backup.exists():
                shutil.copy2(journal, row_backup)

            tic = time.perf_counter()
            high, values, agreement = recompute_twice(method, ranks, len(cy))
            runtime = time.perf_counter() - tic
            threshold = method_threshold(dist, method, thresholds)
            high_max = max(high)
            margin = mp.mpf(str(threshold)) - high_max
            if agreement > 1e-12:
                raise ArithmeticError(f"100/160 dps disagreement: {agreement}")
            if not margin > 1:
                raise ArithmeticError(f"insufficient conservative margin: {margin}")
            if not np.isfinite(values).all() or np.any(values >= threshold):
                raise ArithmeticError("witness path does not certify non-alarm")

            running = np.maximum.accumulate(values)
            crossing = np.zeros(601, dtype=np.int8)
            certificate = (
                "Full-domain binary64 atom ordering stopped conservatively. "
                "For every t, inf_a M_t(a) <= M_t(0.9). Exact binary-rational "
                "rank comparisons and matching 100/160-decimal recursions show "
                f"max_t M_t(0.9)={mp.nstr(high_max, 30)} < {threshold}, "
                f"margin={mp.nstr(margin, 30)}, float64 path agreement={agreement}. "
                "The stored value path is the point-witness upper bound, not the "
                "unavailable full-profile path. Original UNKNOWN is preserved in "
                "09_algorithm_audit/unknown_witness_resolution."
            )
            core.write_npz_atomic(
                trace,
                value=values,
                running_max=running,
                crossing=crossing,
                threshold=np.float64(threshold),
                computed=np.ones(601, dtype=np.uint8),
                time_index=np.arange(601, dtype=np.int32),
                atom_count=np.full(601, -1, dtype=np.int64),
                worst_parameter=np.full(601, np.nan),
                exact_fallback_count=np.int64(1),
                status=np.array(STATUS),
                runtime_seconds=np.float64(runtime),
                peak_memory_mb=np.float64(np.nan),
                error_message=np.array(certificate),
                certificate_kind=np.array("POINT_WITNESS_UPPER_BOUND_NONALARM"),
                witness_parameter=np.float64(WITNESS_FLOAT),
                witness_max=np.float64(high_max),
                threshold_margin=np.float64(margin),
                wealth_precision_dps=np.int64(160),
                rank_semantics=np.array("EXACT_BINARY_RATIONAL_CALIBRATION_LE_ONLINE"),
                trace_semantics=np.array("WITNESS_UPPER_BOUND_NOT_FULL_PROFILE"),
            )
            saved, first = validate_trace(trace, threshold, require_time=True)
            if first != 0:
                raise AssertionError("certificate unexpectedly alarms")
            new_row = row_from_trace(
                saved, first, dist, "health", rep, method, threshold, thresholds
            )
            atomic_json(journal, new_row)
            audit_rows.append(
                dict(
                    distribution=dist,
                    scenario="health",
                    replicate=rep,
                    seed=int(data["seed"]),
                    method=method,
                    prior_status="UNKNOWN",
                    prior_cause="root too close to old atom to certify binary64 ordering",
                    resolution_status=STATUS,
                    witness_parameter=WITNESS_FLOAT,
                    witness_max=mp.nstr(high_max, 40),
                    threshold=threshold,
                    margin=mp.nstr(margin, 40),
                    dps_low=100,
                    dps_high=160,
                    float64_path_agreement=agreement,
                    conclusion="CERTIFIED_NONALARM_BY_POINTWISE_INFIMUM_DOMINATION",
                )
            )
            print(
                "RESOLVED",
                dist,
                rep,
                method,
                "max",
                mp.nstr(high_max, 20),
                "margin",
                mp.nstr(margin, 20),
                flush=True,
            )

    for sid, start, end in sorted(affected_shards):
        regenerate_shard_csv(sid, start, end)

    csv_path = audit_dir / "exact_rank_multiprecision_witness_results.csv"
    tmp = csv_path.with_name(csv_path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, csv_path)

    note = f"""# Resolution of six numerical UNKNOWN records

The six method records arose from three healthy trajectories.  Both full-domain
engines stopped at the same near-collision because binary64 could not certify
the order of a new rank root relative to an old atom.

For each affected method and every monitoring time,
`inf_a M_t(a) <= M_t(0.9)`.  The witness ranks were computed by exact comparison
of the stored binary64 inputs as rational numbers.  Wealth was recomputed at
100 and 160 decimal digits and the resulting float64 paths agreed within the
recorded tolerance.  Every witness path remained at least one full wealth unit
below its alarm threshold; the observed margins were much larger.

Accordingly these records are certified only as non-alarms.  The stored path is
a witness upper bound and MUST NOT be described as the recovered full profile.
Original UNKNOWN traces and rows remain under `original_unknown_records/`.
"""
    atomic_text(audit_dir / "README.md", note)
    atomic_json(
        audit_dir / "resolution_summary.json",
        dict(
            status="PASS",
            records_resolved=len(audit_rows),
            trajectories=3,
            methods=list(METHODS_TO_RESOLVE),
            certificate_status=STATUS,
            full_profile_recovered=False,
            inference="NONALARM_ONLY",
        ),
    )
    print("WITNESS_RESOLUTION_PASS", len(audit_rows), "records")


if __name__ == "__main__":
    main()
