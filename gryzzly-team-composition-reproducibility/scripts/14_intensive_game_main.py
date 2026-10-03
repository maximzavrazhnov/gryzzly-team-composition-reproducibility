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
OUT=str(_STAGEF_DIR);x=pd.read_pickle(f'{OUT}/intensive_eb_project_features.pkl').copy();x['year']=x.project_created_dt.dt.year;y='outcome_log_B_over_C';fs=['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks','observed_team_size','mean_log_prior_projects','mean_eb_resource_tau5'];agrid=[1e-4,1e-3,1e-2,.1,1,10,100,1000]
rows=[];pred=[];coef=[]
for yr in [2022,2023,2024]:
 tr=x[x.year<yr].sort_values('project_created_dt').reset_index(drop=True);te=x[x.year==yr].sort_values('project_created_dt');cv=TimeSeriesSplit(4);best=(1e9,None)
 for a in agrid:
  vv=[]
  for ti,vi in cv.split(tr):
   p=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=a))]);p.fit(tr.iloc[ti][fs],tr.iloc[ti][y]);vv.append(mean_absolute_error(tr.iloc[vi][y],p.predict(tr.iloc[vi][fs])))
  if np.mean(vv)<best[0]:best=(np.mean(vv),a)
 p=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=best[1]))]);p.fit(tr[fs],tr[y]);pr=p.predict(te[fs]);rows.append((yr,best[1],best[0],mean_absolute_error(te[y],pr),mean_squared_error(te[y],pr)**.5,r2_score(te[y],pr),len(tr),len(te)))
 z=te[['project_code','team_id','project_created_dt',y,'overrun']].copy();z['year']=yr;z['model']='IG_main';z['prediction']=pr;pred.append(z)
 # raw-scale coefficients too: standardized Ridge coefficients can be transformed back
 sc=p.named_steps['s']; rr=p.named_steps['r']; raw=rr.coef_/sc.scale_; intercept=rr.intercept_-np.sum(rr.coef_*sc.mean_/sc.scale_)
 for f,cs,cr in zip(fs,rr.coef_,raw):coef.append((yr,f,cs,cr,intercept))
res=pd.DataFrame(rows,columns=['year','alpha','inner_cv_mae','mae','rmse','r2','n_train','n_test']);pred=pd.concat(pred);coef=pd.DataFrame(coef,columns=['year','feature','coef_std','coef_raw','intercept_raw'])
# aggregate + unseen
seen=[]
for r in pred.itertuples(index=False):seen.append(r.team_id in set(x.loc[x.year<r.year,'team_id']))
pred['seen_org_before']=seen;un=pred[~pred.seen_org_before]
agg=pd.DataFrame([{'model':'IG_main','n_oof':len(pred),'mae':mean_absolute_error(pred[y],pred.prediction),'rmse':mean_squared_error(pred[y],pred.prediction)**.5,'r2':r2_score(pred[y],pred.prediction),'n_unseen_org':len(un),'mae_unseen_org':mean_absolute_error(un[y],un.prediction),'rmse_unseen_org':mean_squared_error(un[y],un.prediction)**.5}])
# compare against normalized-game B1 and Q1 existing common predictions
old=pd.read_pickle(f'{OUT}/normalized_game_predictions.pkl')
# old model names B1_card/Q1_mean_ind
rng=np.random.default_rng(99117)
def boot_vs(base,B=10000):
 b=old[old.model==base][['project_code','team_id',y,'prediction']].rename(columns={'prediction':'pb'});m=pred[['project_code','prediction']].rename(columns={'prediction':'pm'});d=b.merge(m,on='project_code');d['eb']=(d[y]-d.pb).abs();d['em']=(d[y]-d.pm).abs();o=d.groupby('team_id').agg(n=('project_code','size'),sb=('eb','sum'),sm=('em','sum'));nn=o.n.to_numpy(float);sb=o.sb.to_numpy(float);sm=o.sm.to_numpy(float);k=len(o);ix=rng.integers(0,k,(B,k));N=nn[ix].sum(1);dd=sm[ix].sum(1)/N-sb[ix].sum(1)/N;obs=d.em.mean()-d.eb.mean();return {'model':'IG_main','base':base,'delta_mae':obs,'pct_delta':100*obs/d.eb.mean(),'ci2.5':np.quantile(dd,.025),'median':np.median(dd),'ci97.5':np.quantile(dd,.975),'p_improve':(dd<0).mean()}
boot=pd.DataFrame([boot_vs('B1_card'),boot_vs('Q1_mean_ind')])
res.to_csv(f'{OUT}/intensive_game_main_results.csv',index=False);pred.to_csv(f'{OUT}/intensive_game_main_predictions.csv',index=False);coef.to_csv(f'{OUT}/intensive_game_main_coefficients.csv',index=False);agg.to_csv(f'{OUT}/intensive_game_main_aggregate.csv',index=False);boot.to_csv(f'{OUT}/intensive_game_main_bootstrap.csv',index=False)
print('RESULTS\n',res.to_string(index=False));print('\nAGG\n',agg.to_string(index=False));print('\nBOOT\n',boot.to_string(index=False));print('\nCOEFS RAW/STD\n',coef.to_string(index=False))
