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
OUT=str(_STAGEF_DIR)
x=pd.read_pickle(f'{OUT}/intensive_eb_project_features.pkl').copy(); x['year']=x.project_created_dt.dt.year
y='outcome_log_B_over_C'
context=['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks']; card=['observed_team_size','pair_count']; ind=['mean_log_prior_projects','mean_log_prior_collaborators']
TAUS=[1,3,5,10,20,50]; ALPHAS=[1e-4,1e-3,1e-2,.1,1,10,100,1000]
for t in TAUS: x[f'mean_eb_delta_tau{t}']=x[f'mean_eb_resource_tau{t}']-x['mean_resource_prior']

def fit(df,fs,a):
 p=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=a))]);p.fit(df[fs],df[y]);return p

def select(df,fs):
 df=df.sort_values('project_created_dt').reset_index(drop=True); cv=TimeSeriesSplit(4); best=(1e9,None)
 for a in ALPHAS:
  vals=[]
  for ti,vi in cv.split(df):
   p=fit(df.iloc[ti],fs,a); vals.append(mean_absolute_error(df.iloc[vi][y],p.predict(df.iloc[vi][fs])))
  q=np.mean(vals)
  if q<best[0]:best=(q,a)
 return best

def select_tau(df,base):
 df=df.sort_values('project_created_dt').reset_index(drop=True);cv=TimeSeriesSplit(4);best=(1e9,None,None,None)
 for t in TAUS:
  fs=base+[f'mean_eb_delta_tau{t}']
  for a in ALPHAS:
   vals=[]
   for ti,vi in cv.split(df):
    p=fit(df.iloc[ti],fs,a);vals.append(mean_absolute_error(df.iloc[vi][y],p.predict(df.iloc[vi][fs])))
   q=np.mean(vals)
   if q<best[0]:best=(q,t,a,fs)
 return best

rows=[];preds=[];coefs=[]
for yr in [2022,2023,2024]:
 tr=x[x.year<yr].sort_values('project_created_dt');te=x[x.year==yr].sort_values('project_created_dt')
 specs={
  'I2_exp_collab':context+card+ind,
  'O1_org_prior':context+card+ind+['mean_resource_prior'],
  'E1_EB_tau5':context+card+ind+['mean_eb_resource_tau5'],
 }
 for m,fs in specs.items():
  cv,a=select(tr,fs);p=fit(tr,fs,a);pr=p.predict(te[fs]);rows.append((yr,m,a,np.nan,cv,mean_absolute_error(te[y],pr),mean_squared_error(te[y],pr)**.5,r2_score(te[y],pr)))
  z=te[['project_code','team_id','project_created_dt',y]].copy();z['year']=yr;z['model']=m;z['prediction']=pr;preds.append(z)
  for f,c in zip(fs,p.named_steps['r'].coef_):coefs.append((yr,m,f,c))
 # worker deviation from org/global prior; tau selected inner-CV
 base=context+card+ind+['mean_resource_prior'];cv,t,a,fs=select_tau(tr,base);p=fit(tr,fs,a);pr=p.predict(te[fs]);m='W1_org_plus_worker_delta'
 rows.append((yr,m,a,t,cv,mean_absolute_error(te[y],pr),mean_squared_error(te[y],pr)**.5,r2_score(te[y],pr)))
 z=te[['project_code','team_id','project_created_dt',y]].copy();z['year']=yr;z['model']=m;z['prediction']=pr;preds.append(z)
 for f,c in zip(fs,p.named_steps['r'].coef_):coefs.append((yr,m,f,c))
res=pd.DataFrame(rows,columns=['year','model','alpha','tau','inner_cv_mae','mae','rmse','r2']);pred=pd.concat(preds);coef=pd.DataFrame(coefs,columns=['year','model','feature','coef_std'])
agg=[]
for m,g in pred.groupby('model'):
 unseen=[]
 for r in g.itertuples(index=False): unseen.append(r.team_id not in set(x.loc[x.year<r.year,'team_id']))
 gu=g[np.array(unseen)]
 agg.append((m,len(g),mean_absolute_error(g[y],g.prediction),mean_squared_error(g[y],g.prediction)**.5,r2_score(g[y],g.prediction),len(gu),mean_absolute_error(gu[y],gu.prediction),mean_squared_error(gu[y],gu.prediction)**.5))
agg=pd.DataFrame(agg,columns=['model','n','mae','rmse','r2','n_unseen','mae_unseen','rmse_unseen']).sort_values('mae')
# vector org bootstrap pair differences
rng=np.random.default_rng(12031)
def boot(m,base,B=5000):
 a=pred[pred.model==base][['project_code','team_id',y,'prediction']].rename(columns={'prediction':'pb'});b=pred[pred.model==m][['project_code','prediction']].rename(columns={'prediction':'pm'});d=a.merge(b,on='project_code');d['eb']=(d[y]-d.pb).abs();d['em']=(d[y]-d.pm).abs();o=d.groupby('team_id').agg(n=('project_code','size'),sb=('eb','sum'),sm=('em','sum'))
 nn=o.n.to_numpy(float);sb=o.sb.to_numpy(float);sm=o.sm.to_numpy(float);k=len(o);ix=rng.integers(0,k,(B,k));N=nn[ix].sum(1);dd=sm[ix].sum(1)/N-sb[ix].sum(1)/N;obs=d.em.mean()-d.eb.mean()
 return (m,base,obs,100*obs/d.eb.mean(),np.quantile(dd,.025),np.median(dd),np.quantile(dd,.975),(dd<0).mean())
br=[]
for base in ['I2_exp_collab','O1_org_prior']:
 for m in ['O1_org_prior','E1_EB_tau5','W1_org_plus_worker_delta']:
  if m!=base:br.append(boot(m,base))
bootdf=pd.DataFrame(br,columns=['model','base','delta_mae','pct_delta','ci2.5','median','ci97.5','p_improve'])
res.to_csv(f'{OUT}/eb_decomposition_results.csv',index=False);agg.to_csv(f'{OUT}/eb_decomposition_aggregate.csv',index=False);coef.to_csv(f'{OUT}/eb_decomposition_coefficients.csv',index=False);bootdf.to_csv(f'{OUT}/eb_decomposition_bootstrap.csv',index=False);pred.to_pickle(f'{OUT}/eb_decomposition_predictions.pkl')
print('RESULTS\n',res.to_string(index=False));print('\nAGG\n',agg.to_string(index=False));print('\nBOOT\n',bootdf.to_string(index=False));print('\nCOEFS\n',coef[coef.feature.str.contains('resource|prior_projects|prior_collab')].pivot_table(index=['model','feature'],columns='year',values='coef_std').round(4).to_string())

# descriptive selection: observed vs full candidate pool at tau=5
cand=pd.read_pickle(f'{OUT}/candidate_resource_features.pkl'); cand['logp']=cand['log_prior_projects'];cand['logc']=cand['log_prior_collaborators']
sel=[]
for pc,g in cand.groupby('project_code'):
 ob=g[g.is_observed_member==1]
 if len(ob)==0:continue
 sel.append((pc,g.team_id.iloc[0],ob.eb_resource_tau5.mean()-g.eb_resource_tau5.mean(),ob.logp.mean()-g.logp.mean(),ob.logc.mean()-g.logc.mean(),len(ob),len(g)))
sel=pd.DataFrame(sel,columns=['project_code','team_id','delta_eb_obs_vs_pool','delta_exp_obs_vs_pool','delta_collab_obs_vs_pool','n_obs','n_pool'])
sel.to_csv(f'{OUT}/eb_selection_diagnostic.csv',index=False)
print('\nSELECTION\n',sel[['delta_eb_obs_vs_pool','delta_exp_obs_vs_pool','delta_collab_obs_vs_pool']].describe(percentiles=[.1,.25,.5,.75,.9]).to_string())
print('share positive', (sel[['delta_eb_obs_vs_pool','delta_exp_obs_vs_pool','delta_collab_obs_vs_pool']]>0).mean().to_dict())
