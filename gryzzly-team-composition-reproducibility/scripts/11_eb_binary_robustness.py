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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
OUT=str(_STAGEF_DIR);x=pd.read_pickle(f'{OUT}/intensive_eb_project_features.pkl').copy();x['year']=x.project_created_dt.dt.year;y='overrun'
context=['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks'];card=['observed_team_size','pair_count'];ind=['mean_log_prior_projects','mean_log_prior_collaborators'];TAUS=[1,3,5,10,20,50];CS=[.001,.01,.1,1,10,100]

def fit(d,fs,C):
 p=Pipeline([('s',StandardScaler()),('l',LogisticRegression(C=C,max_iter=5000,class_weight=None))]);p.fit(d[fs],d[y]);return p

def cv_score(d,fs,C):
 d=d.sort_values('project_created_dt').reset_index(drop=True);cv=TimeSeriesSplit(4);vals=[]
 for ti,vi in cv.split(d):
  p=fit(d.iloc[ti],fs,C);pr=p.predict_proba(d.iloc[vi][fs])[:,1];vals.append(log_loss(d.iloc[vi][y],pr))
 return np.mean(vals)

def select(d,fs):
 z=[(cv_score(d,fs,c),c) for c in CS];return min(z)

def select_tau(d,base):
 best=(1e9,None,None,None)
 for t in TAUS:
  fs=base+[f'mean_eb_resource_tau{t}']
  for c in CS:
   q=cv_score(d,fs,c)
   if q<best[0]:best=(q,t,c,fs)
 return best
rows=[];pred=[]
for yr in [2022,2023,2024]:
 tr=x[x.year<yr];te=x[x.year==yr]
 specs={'B1_card':context+card,'I2_exp_collab':context+card+ind}
 for m,fs in specs.items():
  cv,C=select(tr,fs);p=fit(tr,fs,C);pr=p.predict_proba(te[fs])[:,1];rows.append((yr,m,C,np.nan,cv,roc_auc_score(te[y],pr),average_precision_score(te[y],pr),brier_score_loss(te[y],pr),log_loss(te[y],pr)))
  z=te[['project_code','team_id','project_created_dt',y]].copy();z['year']=yr;z['model']=m;z['prediction']=pr;pred.append(z)
 for m,base in [('I3_exp_EB',context+card+['mean_log_prior_projects']),('I4_exp_collab_EB',context+card+ind)]:
  cv,t,C,fs=select_tau(tr,base);p=fit(tr,fs,C);pr=p.predict_proba(te[fs])[:,1];rows.append((yr,m,C,t,cv,roc_auc_score(te[y],pr),average_precision_score(te[y],pr),brier_score_loss(te[y],pr),log_loss(te[y],pr)))
  z=te[['project_code','team_id','project_created_dt',y]].copy();z['year']=yr;z['model']=m;z['prediction']=pr;pred.append(z)
res=pd.DataFrame(rows,columns=['year','model','C','tau','inner_cv_logloss','roc_auc','avg_precision','brier','log_loss']);p=pd.concat(pred)
agg=[]
for m,g in p.groupby('model'):
 unseen=[]
 for r in g.itertuples(index=False):unseen.append(r.team_id not in set(x.loc[x.year<r.year,'team_id']))
 gu=g[~np.array(unseen)==False] if False else g[~np.array(unseen)] # unused
 # correct unseen mask
 mask=np.array(unseen,dtype=bool);gu=g[mask]
 agg.append((m,len(g),roc_auc_score(g[y],g.prediction),average_precision_score(g[y],g.prediction),brier_score_loss(g[y],g.prediction),log_loss(g[y],g.prediction),len(gu),roc_auc_score(gu[y],gu.prediction),average_precision_score(gu[y],gu.prediction),brier_score_loss(gu[y],gu.prediction),log_loss(gu[y],gu.prediction)))
agg=pd.DataFrame(agg,columns=['model','n','roc_auc','avg_precision','brier','log_loss','n_unseen','auc_unseen','ap_unseen','brier_unseen','logloss_unseen']).sort_values('log_loss')
res.to_csv(f'{OUT}/eb_binary_results.csv',index=False);agg.to_csv(f'{OUT}/eb_binary_aggregate.csv',index=False);p.to_csv(f'{OUT}/eb_binary_predictions.csv',index=False)
print('RESULTS\n',res.to_string(index=False));print('\nAGG\n',agg.to_string(index=False))
