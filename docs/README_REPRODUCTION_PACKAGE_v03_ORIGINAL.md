# Gryzzly Intensive Cooperative Game — Reproduction Package v02

## Purpose
This package reconstructs the analytical pipeline used for the manuscript on intensive cooperative-game project-team composition from Gryzzly longitudinal digital traces. Version 02 corrects the formation-percentile calculation, makes model-optimal matching tie-aware, adds a training-label embargo sensitivity and expands automated validation.

## Included scripts
- `01_...` to `15_...`: Stage F/F3 cohort, feature and primary-model pipeline.
- `16_stageG_robustness_formation.py`: original Stage G robustness, binary, formation and Shapley pipeline retained for traceability.
- `17_formation_validation_corrected.py`: canonical corrected formation validation. It overwrites canonical Stage G formation aliases after writing explicitly suffixed corrected outputs.
- `18_training_label_embargo_sensitivity.py`: 90-day declaration-maturity embargo sensitivity for training targets, with purged inner temporal folds.
- `h2_6_*.py`: no-size, no-planned-duration, composition-only and unseen-workspace audit sensitivities.
- `validate_results.py`: expanded machine-readable validation of the values used in the manuscript.

## Data
Raw Gryzzly files are not bundled. Put these files in `data/`:
`users.csv`, `projects.csv`, `projects_computed.csv`, `tasks.csv`, `subscriptions.csv`, `declarations.zip`.
Optional snapshot-completeness files: `teams.csv`, `tasks_computed.csv`.

## Verified environment
Python 3.13.5; NumPy 2.3.5; pandas 2.2.3; scikit-learn 1.8.0.

## Run order
```bash
python scripts/run_all.py --stage core
python scripts/run_all.py --stage stageg
python scripts/run_all.py --stage audit
python scripts/validate_results.py
```

The `stageg` command runs the original Stage G script, then the corrected formation analysis, then the training-label embargo sensitivity.

## Corrected formation rule
For a project with candidate-pool size N and observed size k:
- if C(N,k) <= 2000, all feasible same-size coalitions are enumerated;
- otherwise 2,000 project-seeded Monte Carlo coalitions are sampled;
- comparisons use a scale-aware tolerance to prevent floating-point order artifacts;
- exact optimal matching means matching any top-k optimum when ties occur at the selection boundary;
- maximum Jaccard is reported over the set of tied optimal coalitions.

## Reproduction boundaries
The data contain static project snapshots rather than versioned planned-duration histories. The embargo sensitivity uses max declaration availability plus 90 days as a conservative target-availability proxy; it cannot reconstruct the exact historical revision time of every static project field.


## v03 update (2026-08-09)
The composition-ablation outputs underlying manuscript Table 2 are now explicitly frozen in `results_reference/`; `expected_results.json` and `validate_results.py` include full Table 2 checks. See `docs/TABLE2_RERUN_VERIFICATION.md`.
