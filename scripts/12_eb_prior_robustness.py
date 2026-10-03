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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
OUT=str(_STAGEF_DIR);cand=pd.read_pickle(f'{OUT}/candidate_resource_features.pkl');x=pd.read_pickle(f'{OUT}/intensive_eb_project_features.pkl').copy();x['year']=x.project_created_dt.dt.year;y='outcome_log_B_over_C';TAUS=[1,3,5,10,20,50];ALPHAS=[1e-4,1e-3,1e-2,.1,1,10,100,1000]
# candidate global-only EB and pure org prior EB with no fallback threshold restriction
for t in TAUS:
 cand[f'eb_global_tau{t}']=(cand.prior_resource_sum+t*cand.global_resource_mean)/(cand.prior_resource_n+t)
 # use org mean when available at all, otherwise global
 prior=np.where(cand.org_resource_n>0,cand.org_resource_mean,cand.global_resource_mean)
 cand[f'eb_organy_tau{t}']=(cand.prior_resource_sum+t*prior)/(cand.prior_resource_n+t)
obs=cand[cand.is_observed_member==1]
for t in TAUS:
 g=obs.groupby('project_code').agg(**{f'mean_eb_global_tau{t}':(f'eb_global_tau{t}','mean'),f'mean_eb_organy_tau{t}':(f'eb_organy_tau{t}','mean')}).reset_index();x=x.merge(g,on='project_code',how='left')
context=['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks'];card=['observed_team_size','pair_count'];ind=['mean_log_prior_projects','mean_log_prior_collaborators'];CS=ALPHAS

def fit(d,fs,a):p=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=a))]);p.fit(d[fs],d[y]);return p
def select(d,kind):
 d=d.sort_values('project_created_dt').reset_index(drop=True);cv=TimeSeriesSplit(4);best=(1e9,None,None,None)
 for t in TAUS:
  col={'global':f'mean_eb_global_tau{t}','organy':f'mean_eb_organy_tau{t}','org3':f'mean_eb_resource_tau{t}'}[kind];fs=context+card+ind+[col]
  for a in ALPHAS:
   z=[]
   for ti,vi in cv.split(d):p=fit(d.iloc[ti],fs,a);z.append(mean_absolute_error(d.iloc[vi][y],p.predict(d.iloc[vi][fs])))
   q=np.mean(z)
   if q<best[0]:best=(q,t,a,fs)
 return best
rows=[];pred=[]
for yr in [2022,2023,2024]:
 tr=x[x.year<yr];te=x[x.year==yr]
 for kind,m in [('global','G_global_EB'),('organy','G_organy_EB'),('org3','G_org3_EB')]:
  cv,t,a,fs=select(tr,kind);p=fit(tr,fs,a);pr=p.predict(te[fs]);rows.append((yr,m,t,a,cv,mean_absolute_error(te[y],pr),mean_squared_error(te[y],pr)**.5,r2_score(te[y],pr)));z=te[['project_code','team_id',y]].copy();z['year']=yr;z['model']=m;z['prediction']=pr;pred.append(z)
res=pd.DataFrame(rows,columns=['year','model','tau','alpha','inner_cv_mae','mae','rmse','r2']);pr=pd.concat(pred);agg=[]
for m,g in pr.groupby('model'):agg.append((m,mean_absolute_error(g[y],g.prediction),mean_squared_error(g[y],g.prediction)**.5,r2_score(g[y],g.prediction)))
agg=pd.DataFrame(agg,columns=['model','mae','rmse','r2']).sort_values('mae');res.to_csv(f'{OUT}/eb_prior_robustness_results.csv',index=False);agg.to_csv(f'{OUT}/eb_prior_robustness_aggregate.csv',index=False)
print(res.to_string(index=False));print('\n',agg.to_string(index=False))
