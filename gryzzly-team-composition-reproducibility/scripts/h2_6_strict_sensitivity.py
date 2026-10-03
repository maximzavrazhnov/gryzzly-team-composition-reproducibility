# PORTABLE PATH CONFIGURATION — added in Stage H2.7
from pathlib import Path as _Path
import os as _os
_REPRO_ROOT = _Path(_os.environ.get('GRYZZLY_REPRO_ROOT', _Path(__file__).resolve().parents[1])).resolve()
_DATA_DIR = _Path(_os.environ.get('GRYZZLY_DATA_DIR', _REPRO_ROOT / 'data')).resolve()
_STAGEF_DIR = _Path(_os.environ.get('GRYZZLY_STAGEF_DIR', _REPRO_ROOT / 'work' / 'stageF')).resolve()
_STAGEG_DIR = _Path(_os.environ.get('GRYZZLY_STAGEG_DIR', _REPRO_ROOT / 'work' / 'stageG')).resolve()
_STAGEF_DIR.mkdir(parents=True, exist_ok=True)
_STAGEG_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

OUT=str(_STAGEF_DIR)
x=pd.read_pickle(f'{OUT}/intensive_eb_project_features.pkl').copy()
x['year']=x.project_created_dt.dt.year
Y='outcome_log_B_over_C'
agrid=[1e-4,1e-3,1e-2,.1,1,10,100,1000]

specs={
 'PRIMARY_CONDITIONAL_SIZE':['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks','observed_team_size','mean_log_prior_projects','mean_eb_resource_tau5'],
 'NO_REALIZED_SIZE':['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks','mean_log_prior_projects','mean_eb_resource_tau5'],
 'STRICT_NO_SIZE_NO_PLANNED':['log_candidate_pool','log1p_initial_leaf_tasks','mean_log_prior_projects','mean_eb_resource_tau5'],
 'COMPOSITION_ONLY':['mean_log_prior_projects','mean_eb_resource_tau5'],
 'STRICT_CONTEXT_ONLY':['log_candidate_pool','log1p_initial_leaf_tasks'],
 'STRICT_CONTEXT_EXP':['log_candidate_pool','log1p_initial_leaf_tasks','mean_log_prior_projects'],
}

def select_alpha(tr,fs):
    cv=TimeSeriesSplit(4);best=(1e99,None)
    for a in agrid:
        vals=[]
        for ti,vi in cv.split(tr):
            p=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=a))]);p.fit(tr.iloc[ti][fs],tr.iloc[ti][Y]);
            vals.append(mean_absolute_error(tr.iloc[vi][Y],p.predict(tr.iloc[vi][fs])))
        m=float(np.mean(vals))
        if m<best[0]:best=(m,a)
    return best

folds=[];preds=[]
for yr in [2022,2023,2024]:
    tr=x[x.year<yr].sort_values('project_created_dt').reset_index(drop=True)
    te=x[x.year==yr].sort_values('project_created_dt').reset_index(drop=True)
    for name,fs in specs.items():
        cvmae,a=select_alpha(tr,fs)
        p=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=a))]);p.fit(tr[fs],tr[Y]);pr=p.predict(te[fs])
        folds.append((yr,name,a,cvmae,mean_absolute_error(te[Y],pr),mean_squared_error(te[Y],pr)**.5,r2_score(te[Y],pr),len(tr),len(te)))
        z=te[['project_code','team_id','project_created_dt',Y]].copy();z['year']=yr;z['model']=name;z['prediction']=pr
        before=set(x.loc[x.year<yr,'team_id']);z['seen_workspace_before']=z['team_id'].isin(before);preds.append(z)
folds=pd.DataFrame(folds,columns=['year','model','alpha','inner_cv_mae','mae','rmse','r2','n_train','n_test'])
pred=pd.concat(preds,ignore_index=True)
agg=[]
for name,g in pred.groupby('model'):
    un=g[~g.seen_workspace_before]
    agg.append((name,len(g),mean_absolute_error(g[Y],g.prediction),mean_squared_error(g[Y],g.prediction)**.5,r2_score(g[Y],g.prediction),len(un),mean_absolute_error(un[Y],un.prediction),mean_squared_error(un[Y],un.prediction)**.5))
agg=pd.DataFrame(agg,columns=['model','n_oof','mae','rmse','r2','n_unseen_workspace','mae_unseen_workspace','rmse_unseen_workspace']).sort_values('mae')

# workspace-cluster bootstrap paired contrasts overall and unseen only
rng=np.random.default_rng(260803)
def cluster_boot(model,base,subset='all',B=10000):
    a=pred[pred.model==model][['project_code','team_id',Y,'prediction','seen_workspace_before']].rename(columns={'prediction':'pm'})
    b=pred[pred.model==base][['project_code','prediction']].rename(columns={'prediction':'pb'})
    d=a.merge(b,on='project_code',validate='one_to_one')
    if subset=='unseen': d=d[~d.seen_workspace_before].copy()
    d['em']=(d[Y]-d.pm).abs();d['eb']=(d[Y]-d.pb).abs()
    o=d.groupby('team_id').agg(n=('project_code','size'),sm=('em','sum'),sb=('eb','sum')).reset_index()
    n=o.n.to_numpy(float);sm=o.sm.to_numpy(float);sb=o.sb.to_numpy(float);k=len(o)
    ix=rng.integers(0,k,(B,k));N=n[ix].sum(1);delta=sm[ix].sum(1)/N-sb[ix].sum(1)/N
    obs=float(d.em.mean()-d.eb.mean())
    return {'subset':subset,'model':model,'base':base,'n_projects':len(d),'n_workspaces':k,'delta_mae':obs,'pct_delta':100*obs/d.eb.mean(),'ci2.5':float(np.quantile(delta,.025)),'median':float(np.median(delta)),'ci97.5':float(np.quantile(delta,.975)),'p_improve':float((delta<0).mean())}

contrasts=[]
for subset in ['all','unseen']:
    for m,b in [
      ('NO_REALIZED_SIZE','STRICT_CONTEXT_ONLY'),
      ('STRICT_NO_SIZE_NO_PLANNED','STRICT_CONTEXT_ONLY'),
      ('STRICT_CONTEXT_EXP','STRICT_CONTEXT_ONLY'),
      ('STRICT_NO_SIZE_NO_PLANNED','STRICT_CONTEXT_EXP'),
      ('PRIMARY_CONDITIONAL_SIZE','NO_REALIZED_SIZE')
    ]:
      contrasts.append(cluster_boot(m,b,subset))
boot=pd.DataFrame(contrasts)

folds.to_csv(f'{OUT}/h2_6_strict_sensitivity_folds.csv',index=False)
pred.to_csv(f'{OUT}/h2_6_strict_sensitivity_predictions.csv',index=False)
agg.to_csv(f'{OUT}/h2_6_strict_sensitivity_aggregate.csv',index=False)
boot.to_csv(f'{OUT}/h2_6_strict_sensitivity_bootstrap.csv',index=False)
print('AGG\n',agg.to_string(index=False));print('\nBOOT\n',boot.to_string(index=False));print('\nFOLDS\n',folds.to_string(index=False))
