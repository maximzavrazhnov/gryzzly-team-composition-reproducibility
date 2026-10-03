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
OUT=str(_STAGEF_DIR);Y='outcome_log_B_over_C';agrid=[1e-4,1e-3,1e-2,.1,1,10,100,1000];rng=np.random.default_rng(260805)
x=pd.read_pickle(f'{OUT}/intensive_eb_project_features.pkl').copy();x['year']=x.project_created_dt.dt.year
specs={
 'CONTEXT_NO_SIZE':['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks'],
 'CONTEXT_EXP_NO_SIZE':['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks','mean_log_prior_projects'],
 'CONTEXT_EXP_EB_NO_SIZE':['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks','mean_log_prior_projects','mean_eb_resource_tau5'],
 'STRICT_CONTEXT_ONLY':['log_candidate_pool','log1p_initial_leaf_tasks'],
 'STRICT_CONTEXT_EXP':['log_candidate_pool','log1p_initial_leaf_tasks','mean_log_prior_projects'],
 'STRICT_CONTEXT_EXP_EB':['log_candidate_pool','log1p_initial_leaf_tasks','mean_log_prior_projects','mean_eb_resource_tau5'],
 'COMPOSITION_EXP_ONLY':['mean_log_prior_projects'],
 'COMPOSITION_EXP_EB':['mean_log_prior_projects','mean_eb_resource_tau5'],
}
def select(tr,fs):
 cv=TimeSeriesSplit(4);best=(1e99,None)
 for a in agrid:
  vv=[]
  for ti,vi in cv.split(tr):
   p=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=a))]);p.fit(tr.iloc[ti][fs],tr.iloc[ti][Y]);vv.append(mean_absolute_error(tr.iloc[vi][Y],p.predict(tr.iloc[vi][fs])))
  if np.mean(vv)<best[0]:best=(float(np.mean(vv)),a)
 return best
rows=[];preds=[]
for yr in [2022,2023,2024]:
 tr=x[x.year<yr].sort_values('project_created_dt').reset_index(drop=True);te=x[x.year==yr].sort_values('project_created_dt').reset_index(drop=True);before=set(x.loc[x.year<yr,'team_id'])
 for m,fs in specs.items():
  cv,a=select(tr,fs);p=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=a))]);p.fit(tr[fs],tr[Y]);pr=p.predict(te[fs]);rows.append((yr,m,a,cv,mean_absolute_error(te[Y],pr),mean_squared_error(te[Y],pr)**.5,r2_score(te[Y],pr),len(te)))
  z=te[['project_code','team_id',Y]].copy();z['year']=yr;z['model']=m;z['prediction']=pr;z['seen']=z.team_id.isin(before);preds.append(z)
folds=pd.DataFrame(rows,columns=['year','model','alpha','inner_cv_mae','mae','rmse','r2','n_test']);pred=pd.concat(preds)
agg=[]
for m,g in pred.groupby('model'):
 un=g[~g.seen];agg.append((m,len(g),mean_absolute_error(g[Y],g.prediction),mean_squared_error(g[Y],g.prediction)**.5,r2_score(g[Y],g.prediction),len(un),mean_absolute_error(un[Y],un.prediction)))
agg=pd.DataFrame(agg,columns=['model','n_oof','mae','rmse','r2','n_unseen','mae_unseen']).sort_values('mae')

def boot(m,b,subset='all',B=10000):
 a=pred[pred.model==m][['project_code','team_id',Y,'prediction','seen']].rename(columns={'prediction':'pm'});bb=pred[pred.model==b][['project_code','prediction']].rename(columns={'prediction':'pb'});d=a.merge(bb,on='project_code')
 if subset=='unseen':d=d[~d.seen]
 d['em']=(d[Y]-d.pm).abs();d['eb']=(d[Y]-d.pb).abs();o=d.groupby('team_id').agg(n=('project_code','size'),sm=('em','sum'),sb=('eb','sum'))
 n=o.n.to_numpy(float);sm=o.sm.to_numpy(float);sb=o.sb.to_numpy(float);k=len(o);vals=[]
 for _ in range(10):
  ix=rng.integers(0,k,(B//10,k));N=n[ix].sum(1);vals.append(sm[ix].sum(1)/N-sb[ix].sum(1)/N)
 v=np.concatenate(vals);obs=d.em.mean()-d.eb.mean();return (subset,m,b,len(d),k,d.em.mean(),d.eb.mean(),obs,100*obs/d.eb.mean(),np.quantile(v,.025),np.median(v),np.quantile(v,.975),(v<0).mean())
comparisons=[('CONTEXT_EXP_NO_SIZE','CONTEXT_NO_SIZE'),('CONTEXT_EXP_EB_NO_SIZE','CONTEXT_EXP_NO_SIZE'),('STRICT_CONTEXT_EXP','STRICT_CONTEXT_ONLY'),('STRICT_CONTEXT_EXP_EB','STRICT_CONTEXT_EXP'),('COMPOSITION_EXP_EB','COMPOSITION_EXP_ONLY')]
br=[]
for s in ['all','unseen']:
 for m,b in comparisons:br.append(boot(m,b,s))
bootdf=pd.DataFrame(br,columns=['subset','model','base','n_projects','n_workspaces','mae_model','mae_base','delta_mae','pct_delta','ci2.5','median','ci97.5','p_improve'])
folds.to_csv(f'{OUT}/h2_6_composition_ablation_folds.csv',index=False);agg.to_csv(f'{OUT}/h2_6_composition_ablation_aggregate.csv',index=False);bootdf.to_csv(f'{OUT}/h2_6_composition_ablation_bootstrap.csv',index=False)
print('AGG\n',agg.to_string(index=False));print('\nBOOT\n',bootdf.to_string(index=False))
