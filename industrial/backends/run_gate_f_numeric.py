"""Gate F-1: eliminate numeric UNKNOWN on archived DAMADICS feature cases.

This is an algorithmic robustness gate only.  It does not claim that the raw
DAMADICS feature satisfies the serial-independence assumptions or validates
detection performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    for sub in ("epprm", "cctm_profile"):
        sys.path.insert(0, str(args.package / "01_source" / sub))
    from full_domain_prm_atoms_fast import stream_fullatom
    from full_domain_cctm_atoms_fast import cctm_fullatom_stream

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = sorted(args.features.glob("DAM_*.npz"))
    if not cases:
        raise RuntimeError(f"no DAM_*.npz under {args.features}")
    for p in cases:
        with np.load(p, allow_pickle=False) as z:
            ref = z["reference"].astype(float)
            online = z["online"].astype(float)
        cy, cx = ref[1:], ref[:-1]
        oy, ox = online[1:], online[:-1]
        for method in ("EP_PRM", "Profiled_CCTM"):
            tic = time.perf_counter()
            try:
                if method == "EP_PRM":
                    ans = stream_fullatom(cy, cx, oy, ox, threshold=20.)
                else:
                    ans = cctm_fullatom_stream(
                        cy, cx, oy, ox, delta=.01, D=.5, C=.1,
                        smooth=1e-6, stop_at_alarm=False, full_path=True
                    )
                status = ans["status"]
                alarm = int(ans["alarm"])
                fallback = int(ans.get("exact_event_fallback_steps", 0))
                atoms = int(ans["atom_counts"][-1]) if ans.get("atom_counts") is not None else int(ans["atoms"])
                error = ""
            except Exception as exc:
                status = "UNKNOWN"; alarm = 0; fallback = 0; atoms = 0; error = repr(exc)
            row = dict(case_id=p.stem, method=method, status=status, alarm=alarm,
                       exact_event_fallback_steps=fallback, final_atoms=atoms,
                       seconds=round(time.perf_counter() - tic, 6), error=error)
            rows.append(row)
            print("GATE_F_CASE", json.dumps(row, ensure_ascii=False), flush=True)

    out = args.output / "gate_f_numeric_results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    unknown = sum(r["status"] == "UNKNOWN" for r in rows)
    summary = dict(cases=len(cases), method_records=len(rows), unknown=unknown,
                   pass_numeric=int(unknown == 0), note=(
                       "Algorithmic tie/root robustness only; not a real-data efficacy claim."
                   ))
    (args.output / "gate_f_numeric_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("GATE_F_NUMERIC_COMPLETE", json.dumps(summary, ensure_ascii=False))
    if unknown:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
