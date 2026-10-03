# PORTABLE PATH CONFIGURATION — added in Stage H2.7
from pathlib import Path as _Path
import os as _os
_REPRO_ROOT = _Path(_os.environ.get('GRYZZLY_REPRO_ROOT', _Path(__file__).resolve().parents[1])).resolve()
_DATA_DIR = _Path(_os.environ.get('GRYZZLY_DATA_DIR', _REPRO_ROOT / 'data')).resolve()
_STAGEF_DIR = _Path(_os.environ.get('GRYZZLY_STAGEF_DIR', _REPRO_ROOT / 'work' / 'stageF')).resolve()
_STAGEG_DIR = _Path(_os.environ.get('GRYZZLY_STAGEG_DIR', _REPRO_ROOT / 'work' / 'stageG')).resolve()
_STAGEF_DIR.mkdir(parents=True, exist_ok=True)
_STAGEG_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np, pandas as pd, json, os
OUT=str(_STAGEF_DIR); BASE=str(_DATA_DIR)
final=pd.read_pickle(f'{OUT}/final_projects.pkl')
ind=pd.read_pickle(f'{OUT}/individual_features_raw.pkl')
pairs=pd.read_pickle(f'{OUT}/pair_features_raw.pkl')
# observed team aggregates
io=ind[ind.is_observed_member==1]
indagg=io.groupby('project_code').agg(
    sum_log_prior_projects=('log_prior_projects','sum'),
    sum_log_prior_collaborators=('log_prior_collaborators','sum'),
    mean_log_prior_projects=('log_prior_projects','mean'),
    mean_log_prior_collaborators=('log_prior_collaborators','mean')
).reset_index()
po=pairs[pairs.is_observed_pair==1]
pairagg=po.groupby('project_code').agg(
    sum_familiarity=('familiarity_log1p','sum'),
    sum_diversity_jaccard=('diversity_jaccard','sum'),
    sum_diversity_gryzzly=('diversity_gryzzly','sum'),
    sum_familiarity_x_diversity=('familiarity_x_diversity','sum'),
    mean_familiarity=('familiarity_log1p','mean'),
    mean_diversity_jaccard=('diversity_jaccard','mean'),
    zero_familiarity_pairs=('shared_projects',lambda x:int((x==0).sum()))
).reset_index()

# initial leaf task count available by exact project creation time
tasks=pd.read_csv(f'{BASE}/tasks.csv',usecols=['id','project_id','created_at','is_container'])
# parse bool robustly
iscont=tasks['is_container'].astype(str).str.lower().isin(['true','t','1'])
tasks['created_dt']=pd.to_datetime(tasks['created_at'],format='mixed',utc=True,errors='coerce')
pc_time=final.set_index('project_id')['project_created_dt'].to_dict()
sel=tasks['project_id'].isin(pc_time.keys()) & (~iscont) & tasks['created_dt'].notna()
tt=tasks.loc[sel,['project_id','created_dt']].copy()
tt['target_dt']=tt['project_id'].map(pc_time)
tt=tt[tt['created_dt']<=tt['target_dt']]
initial=tt.groupby('project_id').size().rename('initial_leaf_tasks').reset_index()

# subscription offer as organizational control
subs=pd.read_csv(f'{BASE}/subscriptions.csv',usecols=['team_id','offer','status'])

x=final.merge(indagg,on='project_code',how='left').merge(pairagg,on='project_code',how='left').merge(initial,on='project_id',how='left').merge(subs,on='team_id',how='left')
x['initial_leaf_tasks']=x['initial_leaf_tasks'].fillna(0).astype(int)
x['pair_count']=x['observed_team_size']*(x['observed_team_size']-1)/2
x['log_planned_h']=np.log(x['planned_h'])
x['log_candidate_pool']=np.log(x['candidate_pool_size'])
x['log1p_initial_leaf_tasks']=np.log1p(x['initial_leaf_tasks'])
# missing pair aggregates for n>=2 should not occur
for c in ['sum_familiarity','sum_diversity_jaccard','sum_diversity_gryzzly','sum_familiarity_x_diversity']:
    if x[c].isna().any(): print('WARNING missing',c,x[c].isna().sum())
# Save
x.to_pickle(f'{OUT}/project_model_features.pkl')
x.to_csv(f'{OUT}/project_model_features.csv',index=False)
print('rows',len(x),'cols',len(x.columns))
print(x[['observed_team_size','pair_count','sum_log_prior_projects','sum_log_prior_collaborators','sum_familiarity','sum_diversity_jaccard','sum_familiarity_x_diversity','log_planned_h','log_candidate_pool','initial_leaf_tasks','outcome_log_B_over_C']].describe(percentiles=[.05,.25,.5,.75,.95]).T.to_string())
print('\noffer/status')
print(x.groupby(['offer','status']).size().sort_values(ascending=False).head(20).to_string())
