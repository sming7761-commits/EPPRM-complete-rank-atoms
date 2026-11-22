"""Secondary equal-false-alarm analysis for the frozen EP-PRM main study.

This script never overwrites the formal main-table outputs. Empirical baseline
thresholds are selected only from the 5,000 independent healthy calibration
maxima per distribution. It reports a fixed alpha grid and a transparently
labelled post-main comparison targeting EP-PRM's observed formal-health FPR.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "08_statistical_analysis"
TAU = 357
GRID = (0.005, 0.01, 0.02, 0.05)
BASELINES = (
    "OLS_PlugIn_PRM",
    "Gaussian_GLR",
    "Gaussian_CUSUM",
    "Conformal_CUSUM",
)
DISTS = ("gaussian", "t5", "exp_centered", "skew_mixture")


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refuse empty output")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def empirical_threshold(values: list[float], alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = len(ordered)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.nextafter(ordered[k - 1], np.inf))


def first_alarm(values: np.ndarray, threshold: float) -> int:
    hits = np.flatnonzero(values[1:] >= threshold)
    return int(hits[0] + 1) if len(hits) else 0


def exact_ci(successes: int, n: int):
    ci = binomtest(successes, n).proportion_ci(0.95, method="exact")
    return float(ci.low), float(ci.high)


def load_trace(dist: str, scene: str, rep: int, method: str) -> np.ndarray:
    path = ROOT / "06_method_traces" / dist / scene / f"{rep:04d}_{method}.npz"
    with np.load(path, allow_pickle=False) as data:
        values = data["value"].copy()
        status = str(data["status"])
    if values.shape != (601,) or not np.isfinite(values).all():
        raise ValueError(f"invalid trace {path} status={status}")
    return values


def bootstrap_paired(delta: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    n = len(delta)
    draws = rng.integers(0, n, size=(5000, n))
    boot = np.mean(delta[draws], axis=1)
    return tuple(float(v) for v in np.quantile(boot, (0.025, 0.975)))


def main() -> None:
    main_rows = read_csv(ROOT / "07_tidy_results" / "method_records.csv")
    if len(main_rows) != 48000:
        raise ValueError(f"expected 48000 formal rows, found {len(main_rows)}")
    if any(r["numeric_status"] in ("UNKNOWN", "ERROR") for r in main_rows):
        raise ValueError("unresolved numerical failures")

    by_key = {
        (r["distribution"], r["scenario"], int(r["replicate"]), r["method"]): r
        for r in main_rows
    }
    ep_fpr = {}
    ep_fault = {}
    for dist in DISTS:
        h = [by_key[(dist, "health", rep, "EP_PRM_736")] for rep in range(1000)]
        f = [by_key[(dist, "fault", rep, "EP_PRM_736")] for rep in range(500)]
        ep_fpr[dist] = sum(int(r["first_alarm"]) > 0 for r in h) / 1000
        ep_fault[dist] = np.asarray(
            [int(r["detected_after_change"]) for r in f], dtype=np.int8
        )

    calibration = defaultdict(list)
    for dist in DISTS:
        path = ROOT / "05_raw_data" / "calibration" / f"max_{dist}.csv"
        rows = read_csv(path)
        for row in rows:
            calibration[(dist, row["method"])].append(float(row["max_statistic"]))
        for method in BASELINES:
            if len(calibration[(dist, method)]) != 5000:
                raise ValueError(f"calibration count mismatch {dist}/{method}")

    operating_rows = []
    paired_rows = []
    cache = {}
    for dist_index, dist in enumerate(DISTS):
        targets = [("FIXED_GRID", alpha) for alpha in GRID]
        targets.append(("POSTHOC_EP_OBSERVED_FPR_MATCH", ep_fpr[dist]))

        for method_index, method in enumerate(BASELINES):
            health_paths = [load_trace(dist, "health", rep, method) for rep in range(1000)]
            fault_paths = [load_trace(dist, "fault", rep, method) for rep in range(500)]
            for target_kind, alpha in targets:
                threshold = empirical_threshold(calibration[(dist, method)], alpha)
                health_first = np.asarray(
                    [first_alarm(path, threshold) for path in health_paths], dtype=np.int32
                )
                fault_first = np.asarray(
                    [first_alarm(path, threshold) for path in fault_paths], dtype=np.int32
                )
                detected = (fault_first >= TAU).astype(np.int8)
                fpr_count = int(np.sum(health_first > 0))
                power_count = int(np.sum(detected))
                fpr_ci = exact_ci(fpr_count, 1000)
                power_ci = exact_ci(power_count, 500)
                delays = np.where(detected == 1, fault_first - TAU, 244)
                detected_delays = fault_first[detected == 1] - TAU

                operating_rows.append(
                    dict(
                        distribution=dist,
                        method=method,
                        threshold_source="5000_INDEPENDENT_HEALTH_MAXIMA",
                        target_kind=target_kind,
                        target_alpha=alpha,
                        threshold=threshold,
                        formal_health_n=1000,
                        false_alarms=fpr_count,
                        observed_fpr=fpr_count / 1000,
                        fpr_ci_low=fpr_ci[0],
                        fpr_ci_high=fpr_ci[1],
                        formal_fault_n=500,
                        detected=power_count,
                        power=power_count / 500,
                        power_ci_low=power_ci[0],
                        power_ci_high=power_ci[1],
                        median_censored_delay=float(np.median(delays)),
                        detected_median_delay=(
                            float(np.median(detected_delays)) if len(detected_delays) else ""
                        ),
                        status="SECONDARY_ANALYSIS_COMPLETE",
                    )
                )

                if target_kind in ("FIXED_GRID", "POSTHOC_EP_OBSERVED_FPR_MATCH") and (
                    target_kind == "POSTHOC_EP_OBSERVED_FPR_MATCH" or alpha == 0.01
                ):
                    delta = ep_fault[dist].astype(np.int16) - detected.astype(np.int16)
                    lo, hi = bootstrap_paired(
                        delta, 202611100700 + 100 * dist_index + method_index
                    )
                    paired_rows.append(
                        dict(
                            distribution=dist,
                            comparison_kind=(
                                "FIXED_ONE_PERCENT" if target_kind == "FIXED_GRID"
                                else "POSTHOC_EP_OBSERVED_FPR_MATCH"
                            ),
                            method_a="EP_PRM_736_THEORY_THRESHOLD_20",
                            method_b=method,
                            baseline_target_alpha=alpha,
                            baseline_threshold=threshold,
                            ep_observed_fpr=ep_fpr[dist],
                            baseline_observed_fpr=fpr_count / 1000,
                            n_paired=500,
                            ep_power=float(np.mean(ep_fault[dist])),
                            baseline_power=float(np.mean(detected)),
                            paired_power_difference=float(np.mean(delta)),
                            bootstrap_ci_low=lo,
                            bootstrap_ci_high=hi,
                            status="SECONDARY_ANALYSIS_COMPLETE",
                        )
                    )

    atomic_csv(OUT / "q1_equal_fpr_operating_points.csv", operating_rows)
    atomic_csv(OUT / "q1_equal_fpr_paired_power.csv", paired_rows)
    protocol = {
        "analysis": "secondary equal-false-alarm comparison",
        "formal_main_overwritten": False,
        "calibration_n_per_distribution": 5000,
        "fixed_grid": list(GRID),
        "primary_fixed_grid_point": 0.01,
        "posthoc_match_disclosure": (
            "Target alpha equals EP-PRM observed formal-health FPR and is therefore "
            "labelled post-main/posthoc; baseline thresholds still use independent "
            "calibration data only."
        ),
        "ep_threshold": 20,
        "ep_threshold_source": "marginal anytime PFA",
        "fault_data_used_for_threshold_selection": False,
        "bootstrap_replicates": 5000,
    }
    tmp = OUT / f"q1_equal_fpr_protocol.json.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    os.replace(tmp, OUT / "q1_equal_fpr_protocol.json")
    print("Q1_EQUAL_FPR_PASS", len(operating_rows), "operating rows", len(paired_rows), "paired rows")


if __name__ == "__main__":
    main()
