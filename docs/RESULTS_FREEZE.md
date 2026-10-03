# RESULTS FREEZE v03 — 2026-08-09

The primary predictive specification remains frozen. The formation-validation layer was corrected after a numerical audit; a separate 90-day training-label embargo sensitivity was added.

## Primary predictive specification
- Cohort: 56-day candidate-activity window; at least 10 prior valid declarations; 90-day historical plan–realization-outcome maturity; partial-pooling tau = 5.
- B1_card MAE: 0.6971577195.
- I1_exp MAE: 0.6602597885.
- I2_exp_collab MAE: 0.6561740359.
- I4_exp_collab_EB MAE: 0.6214366028.
- Final IG_main MAE/RMSE/R2: 0.6197248529 / 0.8241105447 / 0.2681705777.
- Strict sensitivity without realized team size: MAE 0.6245059885.
- Strict sensitivity without realized team size and recorded planned-duration context: MAE 0.6201641366.


## Table 2 composition-ablation freeze
Source script: `scripts/h2_6_composition_ablation.py`.
Frozen outputs: `results_reference/h2_6_composition_ablation_folds.csv`, `results_reference/h2_6_composition_ablation_aggregate.csv`, and `results_reference/h2_6_composition_ablation_bootstrap.csv`.

- No realized team size: context MAE/RMSE/R2 = 0.6972010909 / 0.9433628107 / 0.0410489775; + experience = 0.6706016780 / 0.9024898897 / 0.1223454388; + pooled profile = 0.6245059885 / 0.8454141025 / 0.2298454248.
- Experience increment without realized size: delta MAE = -0.0265994129, 95% workspace-cluster CI [-0.0535005205, -0.0012694857].
- Pooled-profile increment without realized size: delta MAE = -0.0460956896, CI [-0.0827828986, -0.0207467972].
- No realized size and no recorded planned-duration context: context MAE/RMSE/R2 = 0.7024264859 / 0.9755506561 / -0.0255068704; + experience = 0.6882529389 / 0.9513152580 / 0.0248131231; + pooled profile = 0.6201641366 / 0.8835427409 / 0.1588100995.
- Strict experience increment: delta MAE = -0.0141735470, CI [-0.0574335887, 0.0327280771].
- Strict pooled-profile increment: delta MAE = -0.0680888022, CI [-0.1128462007, -0.0355029590].

These values are now explicitly frozen rather than being recoverable only from the analytical script/manuscript.

## Corrected formation validation
The corrected procedure enumerates all feasible same-size coalitions when C(N,k) <= 2000, otherwise uses 2,000 deterministic Monte Carlo draws, and applies scale-aware floating-point tolerance. Optimal-composition matching is tie-aware.

- Eligible OOF projects: 2,107 from 128 source-schema workspaces.
- Exact enumeration: 1,792 projects; Monte Carlo: 315.
- Observed team above random same-size expectation: 77.2663%.
- Mean observed-minus-random score gap: 0.1054667716.
- Median observed-team percentile: 0.8333333333 (83.33rd percentile).
- Share in top random-score quartile: 0.5975320361.
- Share coinciding with at least one model-optimal same-size coalition: 0.1457047935 (14.57%).
- Projects with a score tie at the top-k boundary: 56.
- Median maximum Jaccard to the set of optimal same-size coalitions: 0.3333333333.
- Median optimal score gap: 0.1151752221.

## Training-label embargo sensitivity
A project's static outcome is admitted to a training set only after max declaration availability plus 90 days, and inner temporal folds are purged using the same rule.

- B1_card MAE: 0.7532708136.
- I1_exp MAE: 0.7236498571; delta vs B1 = -0.029621, 95% workspace-cluster CI [-0.082366, 0.007149].
- I2_exp_collab MAE: 0.7171695173.
- I4_exp_collab_EB MAE: 0.6726061909; delta vs I2 = -0.044563, CI [-0.070979, -0.025964].
- IG_main MAE: 0.6738432597; delta vs B1 = -0.079428, CI [-0.152232, -0.032126].

The embargo sensitivity is audit-driven and does not replace the frozen primary rolling-origin estimates. It shows that the pooled plan–realization-history contribution remains robust, while the experience-only H1 increment is directionally favorable but its clustered interval includes zero.

## Interpretation boundaries
1. The pooled profile is historical exposure to project plan–realization outcomes, not individual productivity.
2. The outcome combines planning, time-recording and execution-related labor use; it is not overall project success or technical productivity.
3. The intensive game optimizes composition conditional on a separately chosen staffing level.
4. Formation score gaps are model-implied and non-causal.
5. `team_id` is a source-schema workspace, not necessarily a legal organization.
6. Static user records cannot reconstruct historical transfers, temporary role eligibility or all contemporaneous staffing constraints.
