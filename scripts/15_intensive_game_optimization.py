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
from sklearn.metrics import mean_absolute_error
OUT=str(_STAGEF_DIR);x=pd.read_pickle(f'{OUT}/intensive_eb_project_features.pkl').copy();x['year']=x.project_created_dt.dt.year;cand=pd.read_pickle(f'{OUT}/candidate_resource_features.pkl').copy();y='outcome_log_B_over_C'
fs=['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks','observed_team_size','mean_log_prior_projects','mean_eb_resource_tau5'];agrid=[1e-4,1e-3,1e-2,.1,1,10,100,1000]

def fit_year(yr):
 tr=x[x.year<yr].sort_values('project_created_dt').reset_index(drop=True);cv=TimeSeriesSplit(4);best=(1e9,None)
 for a in agrid:
  vv=[]
  for ti,vi in cv.split(tr):
   p=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=a))]);p.fit(tr.iloc[ti][fs],tr.iloc[ti][y]);vv.append(mean_absolute_error(tr.iloc[vi][y],p.predict(tr.iloc[vi][fs])))
  if np.mean(vv)<best[0]:best=(np.mean(vv),a)
 p=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=best[1]))]);p.fit(tr[fs],tr[y]);sc=p.named_steps['s'];rr=p.named_steps['r'];raw=rr.coef_/sc.scale_
 return best[1],dict(zip(fs,raw))

coef_by_year={yr:fit_year(yr) for yr in [2022,2023,2024]}
rows=[]; members=[]
for yr in [2022,2023,2024]:
 alpha,b=coef_by_year[yr];be=b['mean_log_prior_projects'];br=b['mean_eb_resource_tau5'];bn=b['observed_team_size']
 pcs=set(x.loc[x.year==yr,'project_code'])
 for pc,g in cand[cand.project_code.isin(pcs)].groupby('project_code'):
  g=g.copy();g['q']=be*g['log_prior_projects']+br*g['eb_resource_tau5'];g=g.sort_values('q',ascending=False).reset_index(drop=True)
  q=g.q.to_numpy(float);cum=np.cumsum(q);maxn=min(20,len(g));ns=np.arange(2,maxn+1)
  vals=cum[ns-1]/ns+bn*ns
  j=int(np.argmax(vals));nopt=int(ns[j]);vopt=float(vals[j]);opt=set(g.iloc[:nopt].user_code.astype(int))
  obsset=set(g[g.is_observed_member==1].user_code.astype(int));nobs=len(obsset);og=g[g.user_code.isin(obsset)];vobs=float(og.q.mean()+bn*nobs)
  # best composition conditional on observed size
  fixed=set(g.iloc[:nobs].user_code.astype(int));vfixed=float(cum[nobs-1]/nobs+bn*nobs)
  inter=len(opt & obsset);union=len(opt | obsset);jacc=inter/union if union else np.nan
  fixed_inter=len(fixed&obsset);fixed_j=fixed_inter/len(fixed|obsset) if len(fixed|obsset) else np.nan
  rows.append((pc,yr,alpha,be,br,bn,nobs,nopt,len(g),vobs,vfixed,vopt,vfixed-vobs,vopt-vobs,jacc,fixed_j,inter,len(opt-obsset),len(obsset-opt)))
  for u in opt:members.append((pc,yr,u,'optimal'))
  for u in fixed:members.append((pc,yr,u,'fixed_size_optimal'))
res=pd.DataFrame(rows,columns=['project_code','year','ridge_alpha','coef_exp','coef_eb','coef_n','observed_n','optimal_n','candidate_pool_n','value_observed','value_fixed_size_opt','value_opt','gap_fixed_size','gap_full','jaccard_opt_obs','jaccard_fixed_obs','overlap_opt_obs','additions_opt','removals_opt'])
mem=pd.DataFrame(members,columns=['project_code','year','user_code','recommendation_type'])
res.to_csv(f'{OUT}/intensive_game_optimization.csv',index=False);mem.to_csv(f'{OUT}/intensive_game_recommended_members.csv',index=False)
# summary
summ={
 'projects':len(res),'median_observed_n':float(res.observed_n.median()),'median_optimal_n':float(res.optimal_n.median()),
 'optimal_n_distribution':res.optimal_n.value_counts(normalize=True).sort_index().to_dict(),
 'share_same_n':float((res.observed_n==res.optimal_n).mean()),
 'share_observed_exact_fixed_size_opt':float((res.jaccard_fixed_obs==1).mean()),
 'share_observed_exact_full_opt':float((res.jaccard_opt_obs==1).mean()),
 'median_jaccard_fixed':float(res.jaccard_fixed_obs.median()),'median_jaccard_full':float(res.jaccard_opt_obs.median()),
 'median_gap_fixed_size':float(res.gap_fixed_size.median()),'median_gap_full':float(res.gap_full.median()),
 'mean_gap_fixed_size':float(res.gap_fixed_size.mean()),'mean_gap_full':float(res.gap_full.mean()),
 'share_positive_fixed_gap':float((res.gap_fixed_size>1e-12).mean()),'share_positive_full_gap':float((res.gap_full>1e-12).mean()),
}
import json
with open(f'{OUT}/intensive_game_optimization_summary.json','w') as f:json.dump(summ,f,indent=2)
print(json.dumps(summ,indent=2))
print('\nby year\n',res.groupby('year').agg(projects=('project_code','size'),obs_n=('observed_n','median'),opt_n=('optimal_n','median'),same_n=('observed_n',lambda s:np.nan),median_jacc=('jaccard_opt_obs','median'),median_gap=('gap_full','median')).to_string())
print('\ncoefficients\n',coef_by_year)
