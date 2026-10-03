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
from sklearn.metrics import mean_absolute_error
OUT=str(_STAGEF_DIR);Y='outcome_log_B_over_C';rng=np.random.default_rng(260804)
main=pd.read_csv(f'{OUT}/intensive_game_main_predictions.csv')
old=pd.read_pickle(f'{OUT}/normalized_game_predictions.pkl')
strict=pd.read_csv(f'{OUT}/h2_6_strict_sensitivity_predictions.csv')
# main seen flag was computed. Recompute from all project features/year for consistent def.
x=pd.read_pickle(f'{OUT}/intensive_eb_project_features.pkl').copy();x['year']=x.project_created_dt.dt.year
meta=x[['project_code','team_id','year']].copy()

def prepare(df,model,predcol='prediction'):
    g=df[df.model==model].copy() if 'model' in df.columns else df.copy()
    if 'year' not in g.columns:g=g.merge(meta[['project_code','year']],on='project_code',how='left')
    if 'team_id' not in g.columns:g=g.merge(meta[['project_code','team_id']],on='project_code',how='left')
    seen=[]
    for r in g.itertuples(index=False):seen.append(r.team_id in set(x.loc[x.year<r.year,'team_id']))
    g['seen']=seen
    return g[['project_code','team_id','year',Y,predcol,'seen']].rename(columns={predcol:model})

frames={}
for m in ['B1_card','Q1_mean_ind']:frames[m]=prepare(old,m)
frames['IG_main']=prepare(main,'IG_main')
for m in ['COMPOSITION_ONLY','NO_REALIZED_SIZE','STRICT_NO_SIZE_NO_PLANNED','STRICT_CONTEXT_ONLY','STRICT_CONTEXT_EXP']:
    frames[m]=prepare(strict,m)

def paired_boot(m,b,subset='unseen',B=20000):
    a=frames[m];bb=frames[b][['project_code',b]];d=a.merge(bb,on='project_code',validate='one_to_one')
    if subset=='unseen':d=d[~d.seen]
    d['em']=(d[Y]-d[m]).abs();d['eb']=(d[Y]-d[b]).abs()
    o=d.groupby('team_id').agg(n=('project_code','size'),sm=('em','sum'),sb=('eb','sum')).reset_index()
    n=o.n.to_numpy(float);sm=o.sm.to_numpy(float);sb=o.sb.to_numpy(float);k=len(o)
    # memory-efficient batch bootstrap
    vals=[];batch=1000
    for start in range(0,B,batch):
      z=min(batch,B-start);ix=rng.integers(0,k,(z,k));N=n[ix].sum(1);vals.append(sm[ix].sum(1)/N-sb[ix].sum(1)/N)
    delta=np.concatenate(vals)
    obs=float(d.em.mean()-d.eb.mean())
    return [subset,m,b,len(d),k,float(d.em.mean()),float(d.eb.mean()),obs,100*obs/d.eb.mean(),float(np.quantile(delta,.025)),float(np.median(delta)),float(np.quantile(delta,.975)),float((delta<0).mean())]
rows=[]
for subset in ['all','unseen']:
  for m,b in [
   ('IG_main','B1_card'),('IG_main','Q1_mean_ind'),
   ('COMPOSITION_ONLY','B1_card'),('COMPOSITION_ONLY','Q1_mean_ind'),('COMPOSITION_ONLY','IG_main'),
   ('STRICT_NO_SIZE_NO_PLANNED','B1_card'),('STRICT_NO_SIZE_NO_PLANNED','Q1_mean_ind'),
   ('NO_REALIZED_SIZE','Q1_mean_ind')]:rows.append(paired_boot(m,b,subset))
cols=['subset','model','base','n_projects','n_workspaces','mae_model','mae_base','delta_mae','pct_delta','ci2.5','median','ci97.5','p_improve']
out=pd.DataFrame(rows,columns=cols);out.to_csv(f'{OUT}/h2_6_unseen_workspace_bootstrap.csv',index=False);print(out.to_string(index=False))
