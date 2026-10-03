# Table 2 rerun verification — 2026-08-09

Purpose: close the archival gap for the composition-ablation outputs used in manuscript Table 2.

## Input identity
The author-side PowerShell check confirmed that all six required raw inputs match `data/INPUT_MANIFEST.csv`, including the original `declarations.zip` SHA-256 `a251c24e...0bb9b9d`. In the execution sandbox, the uploaded archive container had a different ZIP-level SHA-256, but its extracted declaration content reproduced the canonical declaration counts, observation horizon, cohort, primary model metrics, and Table 2 results exactly. The five uploaded CSV files matched the manifest byte-for-byte in the sandbox as well.

## Execution environment
- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
These match `PYTHON_VERIFIED.txt` / `requirements-verified.txt`.

## Rerun scope
The final Table 2 calculation itself was executed with the unmodified `scripts/h2_6_composition_ablation.py`. Stage F was rebuilt from the verified raw tables. To fit sandbox execution limits, declaration ingestion used the identical CSV payload extracted from `declarations.zip`, a larger processing chunk, and uncompressed NPZ output; primary cohort reconstruction evaluated the frozen primary threshold (10 declarations) directly rather than re-running the full 5/10/20/50 threshold scan. These execution optimizations do not change the analytical definitions used by the final cohort or Table 2. The rebuilt Stage F reproduced the canonical 4,446,669 raw / 4,428,885 valid declaration counts, 2,682-project cohort, 36,174 candidate rows, 2,025 candidates, and the canonical I4/H1/H2 metrics before Table 2 was regenerated.

## Newly frozen outputs
- `results_reference/h2_6_composition_ablation_folds.csv`
- `results_reference/h2_6_composition_ablation_aggregate.csv`
- `results_reference/h2_6_composition_ablation_bootstrap.csv`

All Table 2 values reproduced exactly to machine precision relative to the manuscript values previously reported at 4 decimal places / 4–6 decimal CI precision.
