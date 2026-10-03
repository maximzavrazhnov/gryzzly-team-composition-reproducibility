# PORTABLE PATH CONFIGURATION — added in Stage H2.7
from pathlib import Path as _Path
import os as _os
_REPRO_ROOT = _Path(_os.environ.get('GRYZZLY_REPRO_ROOT', _Path(__file__).resolve().parents[1])).resolve()
_DATA_DIR = _Path(_os.environ.get('GRYZZLY_DATA_DIR', _REPRO_ROOT / 'data')).resolve()
_STAGEF_DIR = _Path(_os.environ.get('GRYZZLY_STAGEF_DIR', _REPRO_ROOT / 'work' / 'stageF')).resolve()
_STAGEG_DIR = _Path(_os.environ.get('GRYZZLY_STAGEG_DIR', _REPRO_ROOT / 'work' / 'stageG')).resolve()
_STAGEF_DIR.mkdir(parents=True, exist_ok=True)
_STAGEG_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
OUT=str(_STAGEF_DIR)
x=pd.read_pickle(f'{OUT}/project_model_features.pkl').copy(); ind=pd.read_pickle(f'{OUT}/individual_features_raw.pkl'); io=ind[ind.is_observed_member==1]
agg=io.groupby('project_code').agg(
 meanE=('log_prior_projects','mean'),minE=('log_prior_projects','min'),maxE=('log_prior_projects','max'),stdE=('log_prior_projects','std'),
 meanB=('log_prior_collaborators','mean'),minB=('log_prior_collaborators','min'),maxB=('log_prior_collaborators','max'),stdB=('log_prior_collaborators','std')
).reset_index().fillna(0)
x=x.merge(agg,on='project_code',how='left'); x['year']=x.project_created_dt.dt.year
context=['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks']; card=['observed_team_size','pair_count']; sumind=['sum_log_prior_projects','sum_log_prior_collaborators']
mods={
 'B1':context+card,
 'I_sum':context+card+sumind,
 'I_mean':context+card+['meanE','meanB'],
 'I_mean_dispE':context+card+['meanE','meanB','stdE'],
 'I_anchorE':context+card+['meanE','meanB','maxE'],
 'I_weakE':context+card+['meanE','meanB','minE'],
 'I_minmaxE':context+card+['meanE','meanB','minE','maxE'],
 'I_fullE':context+card+['meanE','meanB','minE','maxE','stdE'],
 'I_fullEB':context+card+['meanE','meanB','minE','maxE','stdE','minB','maxB','stdB'],
}
y='outcome_log_B_over_C'; alphas=[1e-4,1e-3,1e-2,.1,1,10,100,1000]
rows=[]; pp=[]
for yr in [2022,2023,2024]:
 tr=x[x.year<yr].sort_values('project_created_dt'); te=x[x.year==yr].sort_values('project_created_dt'); cv=TimeSeriesSplit(4)
 for m,fs in mods.items():
  gs=GridSearchCV(Pipeline([('s',StandardScaler()),('r',Ridge())]),{'r__alpha':alphas},scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
  gs.fit(tr[fs],tr[y]); pr=gs.predict(te[fs])
  rows.append((yr,m,gs.best_params_['r__alpha'],mean_absolute_error(te[y],pr),mean_squared_error(te[y],pr)**.5,r2_score(te[y],pr)))
  z=te[['project_code','team_id',y]].copy(); z['model']=m; z['prediction']=pr; pp.append(z)
res=pd.DataFrame(rows,columns=['year','model','alpha','mae','rmse','r2']); pred=pd.concat(pp)
print(res.to_string(index=False))
agg=[]
for m,g in pred.groupby('model'):
 agg.append((m,mean_absolute_error(g[y],g.prediction),mean_squared_error(g[y],g.prediction)**.5,r2_score(g[y],g.prediction)))
agg=pd.DataFrame(agg,columns=['model','mae','rmse','r2']).sort_values('mae')
print('\nAGG\n',agg.to_string(index=False))
res.to_csv(f'{OUT}/composition_nested_results.csv',index=False); agg.to_csv(f'{OUT}/composition_nested_aggregate.csv',index=False)
