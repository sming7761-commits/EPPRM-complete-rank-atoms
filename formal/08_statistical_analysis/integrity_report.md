# Integrity report — in-turn snapshot, 2026-11-09

**STATE: PARTIAL; FORMAL MAIN EXPERIMENT NOT COMPLETE.**

- Protocol frozen: YES. `FORMAL_MAIN_PROTOCOL.yaml`.
- Pilot seed entries: 800; calibration seed entries: 20,000; formal seed entries: 6,000. Pairwise disjoint: PASS.
- Calibration actual health tracks: 20,000; maxima records: 80,000; 16 final thresholds frozen: PASS.
- Formal original raw tracks saved: 6,000/6,000; health=4,000; weak fault=2,000: PASS.
- Formal 600-step 8-method paired records: 8/6,000 trajectories, 64/48,000 valid method-level traces: INCOMPLETE.
- Method traces contain 601 indexed times (including time zero), true rank profile or computed statistic; status fields are explicit: PASS on these 64.
- First-alarm recomputation from every completed trace: PASS for all 64.
- Prefault and detection labels (strict first alarm <357): PASS for all 64.
- Test record seeds appear only in formal manifest: PASS.
- All 6,000 NPZ raw files listed with SHA-256 in `MANIFEST.csv` and `SHA256SUMS.txt`: PASS.
- No sample discarded for a numerical failure; initial pipeline failure and onset repair archived under `09_algorithm_audit`: PASS.
- Complete formal PFA/power/delay/paired confidence intervals: NOT AVAILABLE because test sample count per distribution/scenario is only one for fully computed methods.
- Reproduce from empty environment: dependency and scripts supplied; 50/50 independent rational EP-PRM regression run in current environment. A clean new Conda environment has NOT been provisioned and verified in this turn.
- Full-width floating implementation not formally numerical-certified; see audit flags.

`python 12_reproduction/verify_package.py --allow-partial --deep` passes. Strict invocation intentionally fails until the full 6,000 x 8 records are completed and a final manifest regenerated.
