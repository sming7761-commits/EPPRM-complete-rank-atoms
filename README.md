# EP-PRM: complete rank atoms for exact streaming monitoring

Research materials for **Exact streaming rank-based monitoring under uncertain dynamics through complete rank atoms**, by Yiming Sun and Chunyan Wang. The manuscript is being prepared for submission; no acceptance is claimed.

The contribution is a complete historical-rank representation and state-inheritance profiler for a scalar dynamics coefficient. The repository separates exact reference arithmetic, archived floating-point production implementations, frozen results, and plotting code. PRM and CCTM are existing point-parameter wealth methods; Profiled CCTM is our continuous-parameter extension.

## Start here

```bash
conda env create -f environment.yml
conda activate epprm_reproduce
python verify_repository.py
python tools/replay_reference_audit.py
```

The reference replay evaluates the original 64 fixed inputs with both backends (128 checks). It is a small correctness replay, not a new power experiment or a re-execution of 6,000 production trajectories.

## Core files

See [`docs/CORE_FILES.csv`](docs/CORE_FILES.csv) for exact paths, purposes and hashes.

| Directory | Contents |
|---|---|
| `reference/` | Exact geometry, generic state interface, independent stream/batch implementation and fixed-input audit |
| `formal/01_source/` | Frozen simulation backends, baselines and trajectory generator |
| `formal/12_reproduction/` | Sharded runner, witness handling, analysis and calibration-target comparison |
| `formal/02_protocols/`, `03_configs/`, `04_seed_manifests/` | Frozen protocol, thresholds and seeds |
| `formal/05_raw_data/calibration/` | All 80,000 calibration maxima |
| `formal/07_tidy_results/method_records.csv` | All 48,000 formal method records |
| `formal/08_statistical_analysis/` | Frozen formal summaries and calibration-target results |
| `industrial/backends/` | Later tie-safe industrial backends and exact binary-rational event keys |
| `industrial/scripts/` | Health-only qualification and industrial detection/calibration scripts |
| `industrial/REALDATA_GATE_F2/` | All registered unit/case gates and derived feature sequences |
| `industrial/REALDATA_GATE_F3/` | All 72 final case-method records, 100,000 maxima and 72 traces |
| `figures/` | Five plotting projects producing all six original figures, with their local source data |
| `figures/frozen_pdf/` | Five main figures and industrial supplementary figure in KBS numbering |
| `paper_source_data/` | Frozen numerical source tables used by the manuscript |

### Two production snapshots

**Do not overwrite the formal simulation backend with the later industrial backend.** The archived 6,000 simulation trajectories were evaluated with the `formal/01_source/` snapshot. The industrial F2/F3 study used later tie-safe geometry in `industrial/backends/`. These snapshots have different hashes and distinct provenance. Wealth updates, threshold allocations and experiment definitions are documented in the original protocols.

Prepare an isolated industrial package with:

```bash
python tools/prepare_industrial_backend.py
python industrial/backends/test_tie_safe.py formal
python industrial/scripts/f3_detection.py --package industrial/runtime_package --f2 industrial/REALDATA_GATE_F2 --output outputs/F3 --bootstrap-reps 5000 --workers 4
```

This F3 command is an expensive rerun, not part of the quick verification. It uses the stored derived features. Recomputing F2 from original DAMADICS observations additionally requires obtaining those observations from the benchmark provider and using the archived source inventory and chronological protocol. Original third-party raw archives are not bundled here.

## Formal results and full trajectory archive

The compact repository contains all formal method records, seeds, thresholds and calibration maxima, but only **one raw Gaussian healthy trajectory and its eight traces** as a replay fixture. It does not contain all 6,000 raw NPZs or all 48,000 trace NPZs.

The full frozen simulation archive is intended as a separate Release asset:

`EPPRM_COMPLETE_RESEARCH_PACKAGE_FORMAL_MAIN_COMPLETE_20261110_185236.zip`

Its archive hash and size are recorded in `docs/RELEASE_ASSETS.json`. The asset is not included in this compact Git tree and is not online until the authors upload it. A mutable repository link alone does not identify that archived version; cite the actual release/tag and, where available, a persistent archive identifier.

To replay full production runs, extract that complete archive separately and use its original reproduction scripts. The original package's deep verifier expects the complete package, including traces and journals; it should not be applied to this intentionally compact tree. New calculations should use a separate output copy so frozen records remain unchanged.

The six resolved non-alarm records use fixed-parameter witnesses; they are not reconstructed full-profile paths. They remain in detection counts and are excluded from full-profile cost distributions.

## Plot reproduction

Run the `build_*.py` script in each directory under `figures/`. Local prerequisites are included so the original fallback paths to the author's workstation should not be needed. Outputs go into each project's `outputs/` directory. The original scripts retain original figure numbering: old Figure 6 is KBS main Figure 5; old Figure 5 is KBS Supplementary Figure S1. Arial was used for some figures; font substitution can change layout. The frozen vector PDFs preserve the submitted visual assets.

The figures are generated from mathematical constructions and stored numerical records. Their presence does not imply that a new complete simulation has been rerun on the reader's machine.

## Reading the results

EP-PRM healthy alarms across the four noise laws are 8/1000, 7/1000, 2/1000 and 8/1000; successful fault detections are 429/500, 389/500, 461/500 and 363/500. Strong calibrated baselines and Oracle comparisons remain in the full tables. DAMADICS final EP-PRM endpoints are 0/5 healthy alarms, 1/5 primary faults and 2/7 fault clusters. Complete outcomes, including missed detections and rejected channels, are retained.

Exact-arithmetic representation theorems and floating-point production calculations are distinct. The reference replay verifies archived small inputs; it does not certify every floating-point production state.

## Provenance, licensing and attribution

`docs/SOURCE_FILE_PROVENANCE.csv` maps copied files to original archive members and hashes. `FILE_SHA256.json` describes this distribution. Cached binaries, server logs, manuscript drafts and original industrial raw archives are excluded.

No new software/data reuse license has been selected by the authors in this preparation step. Public availability is not an assertion that every file is licensed under MIT, Apache or Creative Commons. Existing upstream notices and benchmark-provider terms remain applicable. Authors should make an explicit licensing decision before describing the repository as open-source software.

The core methods build on predictive rank martingales (Kuang and Xia, arXiv:2609.00536), conditional conformal test martingales (Shaer et al., ICML 2026), and the conformal CUSUM literature. See `docs/METHOD_REFERENCES.bib` for bibliographic records; the local CCTM reference is an independent-formula implementation, not a claim of official continuous profiling.
