#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F = Path(__import__('os').environ.get('GRYZZLY_STAGEF_DIR', ROOT / 'work' / 'stageF'))
G = Path(__import__('os').environ.get('GRYZZLY_STAGEG_DIR', ROOT / 'work' / 'stageG'))
E = json.loads((ROOT / 'docs' / 'expected_results.json').read_text(encoding='utf-8'))
checks = []

def close(name, actual, expected, tol=5e-6):
    ok = abs(float(actual) - float(expected)) <= tol
    checks.append((name, float(actual), float(expected), ok))

def equal(name, actual, expected):
    ok = actual == expected
    checks.append((name, float(actual), float(expected), ok))

# Cohort
cohort = json.loads((F / 'cohort_meta.json').read_text())
equal('cohort projects', cohort['final_projects'], E['cohort']['projects'])
equal('cohort workspaces', cohort['final_organizations'], E['cohort']['workspaces'])
equal('candidate rows', cohort['candidate_rows'], E['cohort']['candidate_rows'])
equal('unique candidates', cohort['unique_candidates'], E['cohort']['unique_candidates'])

# Main model sequence
nested = pd.read_csv(F / 'intensive_eb_nested_aggregate.csv').set_index('model')
main = pd.read_csv(F / 'intensive_game_main_aggregate.csv').iloc[0]
close('B1_card MAE', nested.loc['B1_card', 'mae'], E['main_models']['B1_card_mae'])
close('I1_exp MAE', nested.loc['I1_exp', 'mae'], E['main_models']['I1_exp_mae'])
close('I2_exp_collab MAE', nested.loc['I2_exp_collab', 'mae'], E['main_models']['I2_exp_collab_mae'])
close('I4_exp_collab_EB MAE', nested.loc['I4_exp_collab_EB', 'mae'], E['main_models']['I4_exp_collab_EB_mae'])
close('IG_main MAE', main['mae'], E['main_models']['IG_main_mae'])
close('IG_main RMSE', main['rmse'], E['main_models']['IG_main_rmse'])
close('IG_main R2', main['r2'], E['main_models']['IG_main_r2'])

boot = pd.read_csv(F / 'intensive_eb_cluster_bootstrap.csv')
h1 = boot[(boot.model == 'I1_exp') & (boot.base == 'B1_card')].iloc[0]
h2 = boot[(boot.model == 'I4_exp_collab_EB') & (boot.base == 'I2_exp_collab')].iloc[0]
for label, row, prefix in [('H1', h1, 'H1'), ('H2', h2, 'H2')]:
    close(f'{label} delta', row.delta_mae, E['hypothesis_contrasts'][f'{prefix}_delta_mae'])
    close(f'{label} CI low', row['ci2.5'], E['hypothesis_contrasts'][f'{prefix}_ci_low'])
    close(f'{label} CI high', row['ci97.5'], E['hypothesis_contrasts'][f'{prefix}_ci_high'])

# Stage G binary and corrected formation
stageg = json.loads((G / 'stageG_summary.json').read_text())
for key, exp_key in [('roc_auc', 'roc_auc'), ('avg_precision', 'average_precision'), ('brier', 'brier'), ('log_loss', 'log_loss')]:
    close(f'binary {key}', stageg['binary'][key], E['binary'][exp_key])
formation = json.loads((G / 'formation_validation_summary_corrected.json').read_text())
fexp = E['formation_corrected']
equal('formation projects', formation['projects'], fexp['projects'])
equal('formation workspaces', formation['workspaces'], fexp['workspaces'])
equal('formation exact projects', formation['exact_enumeration_projects'], fexp['exact_enumeration_projects'])
equal('formation MC projects', formation['monte_carlo_projects'], fexp['monte_carlo_projects'])
close('formation share above random', formation['share_observed_above_pool_mean'], fexp['share_above_random_expectation'])
close('formation mean gap', formation['mean_observed_minus_random'], fexp['mean_gap'])
close('formation median percentile', formation['median_observed_random_percentile'], fexp['median_percentile'])
close('formation top quartile share', formation['share_observed_top_quartile_random'], fexp['share_top_quartile'])
close('formation exact any optimum', formation['share_observed_exact_any_optimal'], fexp['share_exact_any_optimal'])
equal('formation boundary ties', formation['projects_with_boundary_ties'], fexp['boundary_tie_projects'])
close('formation median max Jaccard', formation['median_max_jaccard_optimal'], fexp['median_max_jaccard'])
close('formation median optimal gap', formation['median_optimal_gap'], fexp['median_optimal_gap'])
bins = pd.read_csv(G / 'figure2_bins_corrected.csv')['project_count'].astype(int).tolist()
checks.append(('Figure 2 bins', float(sum(bins)), float(sum(fexp['figure2_bins'])), bins == fexp['figure2_bins']))

# Strict and unseen sensitivities
strict = pd.read_csv(F / 'h2_6_strict_sensitivity_aggregate.csv').set_index('model')
close('no realized size MAE', strict.loc['NO_REALIZED_SIZE', 'mae'], E['strict_sensitivity']['no_size_mae'])
close('strict no size/no planned MAE', strict.loc['STRICT_NO_SIZE_NO_PLANNED', 'mae'], E['strict_sensitivity']['no_size_no_planned_mae'])
composition_boot = pd.read_csv(F / 'h2_6_composition_ablation_bootstrap.csv')
composition_agg = pd.read_csv(F / 'h2_6_composition_ablation_aggregate.csv').set_index('model')
t2 = E['composition_ablation_table2']
# Full manuscript Table 2 audit: aggregate metrics and paired workspace-cluster bootstrap intervals.
for model, prefix in [
    ('CONTEXT_NO_SIZE','no_size_context'),
    ('CONTEXT_EXP_NO_SIZE','no_size_experience'),
    ('CONTEXT_EXP_EB_NO_SIZE','no_size_pooled'),
    ('STRICT_CONTEXT_ONLY','strict_context'),
    ('STRICT_CONTEXT_EXP','strict_experience'),
    ('STRICT_CONTEXT_EXP_EB','strict_pooled')]:
    close(f'Table 2 {model} MAE', composition_agg.loc[model,'mae'], t2[f'{prefix}_mae'])
    close(f'Table 2 {model} RMSE', composition_agg.loc[model,'rmse'], t2[f'{prefix}_rmse'])
    close(f'Table 2 {model} R2', composition_agg.loc[model,'r2'], t2[f'{prefix}_r2'])

for model, prefix in [
    ('CONTEXT_EXP_NO_SIZE','no_size_experience'),
    ('CONTEXT_EXP_EB_NO_SIZE','no_size_pooled'),
    ('STRICT_CONTEXT_EXP','strict_experience'),
    ('STRICT_CONTEXT_EXP_EB','strict_pooled')]:
    row = composition_boot[(composition_boot['subset']=='all') & (composition_boot.model==model)].iloc[0]
    close(f'Table 2 {model} delta', row.delta_mae, t2[f'{prefix}_delta'])
    close(f'Table 2 {model} CI low', row['ci2.5'], t2[f'{prefix}_ci_low'])
    close(f'Table 2 {model} CI high', row['ci97.5'], t2[f'{prefix}_ci_high'])
se = composition_boot[(composition_boot['subset'] == 'all') & (composition_boot.model == 'STRICT_CONTEXT_EXP')].iloc[0]
sr = composition_boot[(composition_boot['subset'] == 'all') & (composition_boot.model == 'STRICT_CONTEXT_EXP_EB')].iloc[0]
close('strict experience delta', se.delta_mae, E['strict_sensitivity']['strict_experience_delta'])
close('strict experience CI low', se['ci2.5'], E['strict_sensitivity']['strict_experience_ci_low'])
close('strict experience CI high', se['ci97.5'], E['strict_sensitivity']['strict_experience_ci_high'])
close('strict EB delta', sr.delta_mae, E['strict_sensitivity']['strict_EB_delta'])
close('strict EB CI low', sr['ci2.5'], E['strict_sensitivity']['strict_EB_ci_low'])
close('strict EB CI high', sr['ci97.5'], E['strict_sensitivity']['strict_EB_ci_high'])

unseen = pd.read_csv(F / 'h2_6_unseen_workspace_bootstrap.csv')
uvb = unseen[(unseen['subset'] == 'unseen') & (unseen.model == 'IG_main') & (unseen.base == 'B1_card')].iloc[0]
uvi = unseen[(unseen['subset'] == 'unseen') & (unseen.model == 'IG_main') & (unseen.base == 'Q1_mean_ind')].iloc[0]
ue = E['unseen_workspace']
equal('unseen project n', uvb.n_projects, ue['n'])
equal('unseen workspace n', uvb.n_workspaces, ue['workspaces'])
close('unseen IG MAE', uvb.mae_model, ue['IG_main_mae'])
close('unseen B1 MAE', uvb.mae_base, ue['B1_card_mae'])
close('unseen I2 MAE', uvi.mae_base, ue['I2_exp_collab_mae'])
close('unseen IG vs B1 CI low', uvb['ci2.5'], ue['IG_vs_B1_ci_low'])
close('unseen IG vs B1 CI high', uvb['ci97.5'], ue['IG_vs_B1_ci_high'])
close('unseen IG vs I2 CI low', uvi['ci2.5'], ue['IG_vs_I2_ci_low'])
close('unseen IG vs I2 CI high', uvi['ci97.5'], ue['IG_vs_I2_ci_high'])

# Training-label embargo sensitivity
emb = pd.read_csv(G / 'training_label_embargo_aggregate.csv').set_index('model')
for model, key in [('B1_card', 'B1_card_mae'), ('I1_exp', 'I1_exp_mae'), ('I2_exp_collab', 'I2_exp_collab_mae'), ('I4_exp_collab_EB', 'I4_exp_collab_EB_mae'), ('IG_main', 'IG_main_mae')]:
    close(f'embargo {model} MAE', emb.loc[model, 'mae'], E['training_label_embargo'][key])
eboot = pd.read_csv(G / 'training_label_embargo_bootstrap.csv')
for label, model, base in [('H1', 'I1_exp', 'B1_card'), ('H2', 'I4_exp_collab_EB', 'I2_exp_collab')]:
    row = eboot[(eboot.model == model) & (eboot.base == base)].iloc[0]
    close(f'embargo {label} delta', row.delta_mae, E['training_label_embargo'][f'{label}_delta_mae'])
    close(f'embargo {label} CI low', row['ci2.5'], E['training_label_embargo'][f'{label}_ci_low'])
    close(f'embargo {label} CI high', row['ci97.5'], E['training_label_embargo'][f'{label}_ci_high'])

# Shapley, including constant-score teams
sh = pd.read_csv(G / 'shapley_observed_teams.csv')
equal('Shapley rows', len(sh), E['shapley']['rows'])
equal('Shapley projects', sh.project_code.nunique(), E['shapley']['projects'])
constant = 0
spearman = []
for _, group in sh.groupby('project_code'):
    if group.q.nunique(dropna=True) <= 1:
        constant += 1
    else:
        spearman.append(group[['q', 'shapley_phi']].corr(method='spearman').iloc[0, 1])
equal('constant-score teams', constant, E['shapley']['constant_teams'])
equal('nonconstant teams', len(spearman), E['shapley']['nonconstant_teams'])
close('median Spearman nonconstant', np.nanmedian(spearman), E['shapley']['median_spearman_nonconstant'])

for name, actual, expected, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'} | {name}: got={actual:.10g} expected={expected:.10g}")
if not all(item[3] for item in checks):
    sys.exit(1)
