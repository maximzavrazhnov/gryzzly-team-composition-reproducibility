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
OUT=str(_STAGEF_DIR);projects=pd.read_pickle(f'{OUT}/projects_static.pkl');pa=pd.read_pickle(f'{OUT}/project_activity.pkl');up=pd.read_pickle(f'{OUT}/user_project_history.pkl');ind=pd.read_pickle(f'{OUT}/individual_features_raw.pkl');x0=pd.read_pickle(f'{OUT}/project_model_features.pkl').copy();x0['year']=x0.project_created_dt.dt.year;y='outcome_log_B_over_C';alphas=[1e-4,1e-3,1e-2,.1,1,10,100,1000];tau=5.;DAY=int(pd.Timedelta(days=1).value)
rp0=projects.merge(pa[['project_code','max_avail_ns']],on='project_code',how='left');rp0=rp0[(rp0.planned_h>0)&(rp0.elapsed_h>0)&rp0.max_avail_ns.notna()].copy();rp0['resource_outcome']=np.log(rp0.planned_h/rp0.elapsed_h)

def build_feature(days):
 rp=rp0.copy();rp['known']=rp.max_avail_ns.astype('int64')+days*DAY
 ur=up[['user_code','project_code']].drop_duplicates().merge(rp[['project_code','team_id','resource_outcome','known']],on='project_code')
 uh={}
 for k,g in ur.groupby('user_code'):
  g=g.sort_values('known');uh[int(k)]=(g.known.to_numpy(np.int64),np.cumsum(g.resource_outcome.to_numpy(float)))
 oh={}
 for k,g in rp.groupby('team_id'):
  g=g.sort_values('known');oh[k]=(g.known.to_numpy(np.int64),np.cumsum(g.resource_outcome.to_numpy(float)))
 gl=rp.sort_values('known');gt=gl.known.to_numpy(np.int64);gc=np.cumsum(gl.resource_outcome.to_numpy(float))
 vals=[]
 for r in ind.itertuples(index=False):
  t=int(r.project_created_dt.value); z=uh.get(int(r.user_code));un=0;us=0.
  if z is not None:
   n=np.searchsorted(z[0],t,'left');un=int(n);us=float(z[1][n-1]) if n else 0.
  z=oh.get(r.team_id);on=0;os=0.
  if z is not None:
   n=np.searchsorted(z[0],t,'left');on=int(n);os=float(z[1][n-1]) if n else 0.
  gn=int(np.searchsorted(gt,t,'left'));gm=float(gc[gn-1]/gn) if gn else 0.;om=os/on if on else np.nan;prior=om if on>=3 else gm
  eb=(us+tau*prior)/(un+tau);vals.append((int(r.project_code),int(r.is_observed_member),eb))
 q=pd.DataFrame(vals,columns=['project_code','obs','eb']);q=q[q.obs==1].groupby('project_code').eb.mean().rename(f'eb_m{days}').reset_index();return q

def fit(d,fs,a):p=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=a))]);p.fit(d[fs],d[y]);return p
rows=[]
for days in [60,90,120,180]:
 x=x0.merge(build_feature(days),on='project_code');fs=['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks','observed_team_size','pair_count','mean_log_prior_projects','mean_log_prior_collaborators',f'eb_m{days}'];ys=[];ps=[]
 for yr in [2022,2023,2024]:
  tr=x[x.year<yr].sort_values('project_created_dt').reset_index(drop=True);te=x[x.year==yr];cv=TimeSeriesSplit(4);best=(1e9,None)
  for a in alphas:
   vv=[]
   for ti,vi in cv.split(tr):p=fit(tr.iloc[ti],fs,a);vv.append(mean_absolute_error(tr.iloc[vi][y],p.predict(tr.iloc[vi][fs])))
   if np.mean(vv)<best[0]:best=(np.mean(vv),a)
  p=fit(tr,fs,best[1]);pr=p.predict(te[fs]);ys.extend(te[y]);ps.extend(pr)
 rows.append((days,mean_absolute_error(ys,ps),mean_squared_error(ys,ps)**.5,r2_score(ys,ps)))
res=pd.DataFrame(rows,columns=['history_maturity_days','mae','rmse','r2']);res.to_csv(f'{OUT}/eb_maturity_robustness.csv',index=False);print(res.to_string(index=False))
