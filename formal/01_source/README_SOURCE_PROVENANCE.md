# Source provenance and third-party implementations

- EP-PRM / original PRM core: independently implemented from the publicly described Predictive Rank Martingale algorithm; not an author's official package.
- CCTM point process: independently implemented and checked by user with 12 full exposed-field parity cases against author repository https://github.com/shaersh/cctm, commit a9feb795d9fa98cc1d0c075f8f08a5c510c7a844. Official source itself is NOT bundled here. `09_algorithm_audit/official_parity_live.csv` contains recorded parity output.
- Profiled CCTM: *our* continuous-nuisance extension, not present in the author's official package.
- Conformal CUSUM: formula-level independent plug-in residual implementation; empirical 600-step PFA is calibrated independently; no ARL or original-paper exact implementation guarantee is claimed for this plug-in.
- Gaussian GLR and Gaussian CUSUM: parametric scoring using fixed Gaussian formulas and estimated healthy scale, with independent empirical PFA calibration.
- Old K8: inherited legacy implementation using only B suffix for calibration and 224 health prefix for parameter training. Symmetric-innovation theory does not transfer to skewed distributions.
- Exact arithmetic: abstract finite rank-atom correctness. Full 736x600 fast engines are general-position binary64, not proof-carrying rigorous interval-arithmetic certification.
