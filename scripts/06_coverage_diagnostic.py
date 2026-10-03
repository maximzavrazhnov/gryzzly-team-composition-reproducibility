# PORTABLE PATH CONFIGURATION — added in Stage H2.7
from pathlib import Path as _Path
import os as _os
_REPRO_ROOT = _Path(_os.environ.get('GRYZZLY_REPRO_ROOT', _Path(__file__).resolve().parents[1])).resolve()
_DATA_DIR = _Path(_os.environ.get('GRYZZLY_DATA_DIR', _REPRO_ROOT / 'data')).resolve()
_STAGEF_DIR = _Path(_os.environ.get('GRYZZLY_STAGEF_DIR', _REPRO_ROOT / 'work' / 'stageF')).resolve()
_STAGEG_DIR = _Path(_os.environ.get('GRYZZLY_STAGEG_DIR', _REPRO_ROOT / 'work' / 'stageG')).resolve()
_STAGEF_DIR.mkdir(parents=True, exist_ok=True)
_STAGEG_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np,pandas as pd,math
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
OUT=str(_STAGEF_DIR)
final=pd.read_pickle(f'{OUT}/final_projects.pkl').sort_values('project_created_dt').copy()
cand=pd.read_pickle(f'{OUT}/candidate_membership.pkl')
up=pd.read_pickle(f'{OUT}/user_project_history.pkl')
x=pd.read_pickle(f'{OUT}/project_model_features.pkl').copy()
# user histories
uh={}
for u,g in up.groupby('user_code',sort=False):
 gg=g.sort_values('first_avail_ns'); uh[int(u)]=(gg.first_avail_ns.to_numpy(np.int64),gg.project_code.to_numpy(np.int32))
obs=cand[cand.is_observed_member==1].groupby('project_code').user_code.apply(lambda s:list(map(int,s)))
rows=[]
for r in final.itertuples(index=False):
 pc=int(r.project_code); t=int(r.project_created_dt.value); team=obs[pc]
 sets=[]; sizes=[]
 for u in team:
  ts,ps=uh[u]; k=np.searchsorted(ts,t,side='left'); st=set(map(int,ps[:k])); sets.append(st); sizes.append(len(st))
 uni=set().union(*sets) if sets else set(); total=sum(sizes)
 union_n=len(uni); redundancy=total-union_n
 div_ratio=union_n/total if total>0 else np.nan
 rows.append((pc,union_n,total,redundancy,div_ratio,math.log1p(union_n),math.log1p(redundancy)))
cov=pd.DataFrame(rows,columns=['project_code','union_prior_projects','total_prior_project_exposures','redundant_project_exposures','coverage_ratio','log1p_union_prior_projects','log1p_redundant_exposures'])
x=x.merge(cov,on='project_code',how='left')
x['year']=x.project_created_dt.dt.year
print(x[['union_prior_projects','total_prior_project_exposures','redundant_project_exposures','coverage_ratio']].describe(percentiles=[.05,.25,.5,.75,.95]).T.to_string())
print('\ncorr coverage vs existing')
print(x[['union_prior_projects','total_prior_project_exposures','redundant_project_exposures','coverage_ratio','sum_log_prior_projects','sum_log_prior_collaborators','observed_team_size','pair_count','sum_familiarity','sum_diversity_jaccard']].corr().round(3).to_string())
x.to_pickle(f'{OUT}/project_model_features_with_coverage.pkl')

context=['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks']; card=['observed_team_size','pair_count']; ind=['sum_log_prior_projects','sum_log_prior_collaborators']
mods={
 'B1_cardinality':context+card,
 'N1_individual':context+card+ind,
 'C1_N1_plus_union':context+card+ind+['union_prior_projects'],
 'C2_N1_plus_coverage_ratio':context+card+ind+['coverage_ratio'],
 'C3_N1_union_familiarity':context+card+ind+['union_prior_projects','sum_familiarity'],
 'C4_N1_union_F_D':context+card+ind+['union_prior_projects','sum_familiarity','sum_diversity_jaccard'],
}
y='outcome_log_B_over_C'; alphas=[1e-4,1e-3,1e-2,.1,1,10,100,1000]
rows=[]; pp=[]
for yr in [2022,2023,2024]:
 tr=x[x.year<yr].sort_values('project_created_dt'); te=x[x.year==yr].sort_values('project_created_dt'); inner=TimeSeriesSplit(n_splits=4)
 for name,fs in mods.items():
  gs=GridSearchCV(Pipeline([('s',StandardScaler()),('r',Ridge())]),{'r__alpha':alphas},cv=inner,scoring='neg_mean_absolute_error',n_jobs=-1)
  gs.fit(tr[fs],tr[y]); pr=gs.predict(te[fs])
  rows.append((yr,name,len(tr),len(te),gs.best_params_['r__alpha'],mean_absolute_error(te[y],pr),mean_squared_error(te[y],pr)**.5,r2_score(te[y],pr)))
  z=te[['project_code','team_id',y]].copy(); z['year']=yr; z['model']=name; z['prediction']=pr; pp.append(z)
res=pd.DataFrame(rows,columns=['year','model','n_train','n_test','alpha','mae','rmse','r2']); pred=pd.concat(pp)
res.to_csv(f'{OUT}/coverage_nested_results.csv',index=False); pred.to_pickle(f'{OUT}/coverage_nested_predictions.pkl')
print('\nFOLDS\n',res.to_string(index=False))
agg=[]
for m,g in pred.groupby('model'):
 agg.append((m,len(g),mean_absolute_error(g[y],g.prediction),mean_squared_error(g[y],g.prediction)**.5,r2_score(g[y],g.prediction)))
agg=pd.DataFrame(agg,columns=['model','n','mae','rmse','r2']).sort_values('mae'); agg.to_csv(f'{OUT}/coverage_nested_aggregate.csv',index=False)
print('\nAGG\n',agg.to_string(index=False))
