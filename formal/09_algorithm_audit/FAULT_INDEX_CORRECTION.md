# Formal fault-start indexing correction (before formal statistical analysis)

Protocol says the first changed innovation is online transition t=357, corresponding
to zero-based array index 356. `simulate_core.ar_path` interprets `fault_at`
as a zero-based transition index. The initial eight-path pipeline smoke and
2,000 generated fault arrays passed `fault_at=357`, changing transition 358.
This was discovered in pipeline validation before any formal summary statistics.
All affected formal fault raw arrays are regenerated deterministically from
the same originally frozen seeds with `fault_at=356` and the initial four
fault smoke method rows are archived here before replacement. Healthy and
calibration arrays are unchanged. One representative pre-fix raw fault array
is retained for independent forensic checking. This is a protocol bug fix,
NOT outcome-based sample reselection or a new random seed.
