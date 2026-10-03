# PORTABLE PATH CONFIGURATION — added in Stage H2.7
from pathlib import Path as _Path
import os as _os
_REPRO_ROOT = _Path(_os.environ.get('GRYZZLY_REPRO_ROOT', _Path(__file__).resolve().parents[1])).resolve()
_DATA_DIR = _Path(_os.environ.get('GRYZZLY_DATA_DIR', _REPRO_ROOT / 'data')).resolve()
_STAGEF_DIR = _Path(_os.environ.get('GRYZZLY_STAGEF_DIR', _REPRO_ROOT / 'work' / 'stageF')).resolve()
_STAGEG_DIR = _Path(_os.environ.get('GRYZZLY_STAGEG_DIR', _REPRO_ROOT / 'work' / 'stageG')).resolve()
_STAGEF_DIR.mkdir(parents=True, exist_ok=True)
_STAGEG_DIR.mkdir(parents=True, exist_ok=True)

import os, json, math, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,roc_auc_score,average_precision_score,brier_score_loss,log_loss

warnings.filterwarnings('ignore')
OUT=str(_STAGEF_DIR)
DAY_NS=int(pd.Timedelta(days=1).value)
MATURE_NS=90*DAY_NS

# ---------- load existing reproducible Stage F objects ----------
projects=pd.read_pickle(f'{OUT}/projects_static.pkl').copy()
pa=pd.read_pickle(f'{OUT}/project_activity.pkl').copy()
up=pd.read_pickle(f'{OUT}/user_project_history.pkl').copy()
ind=pd.read_pickle(f'{OUT}/individual_features_raw.pkl').copy()
projfeat=pd.read_pickle(f'{OUT}/project_model_features.pkl').copy()
projfeat['year']=projfeat['project_created_dt'].dt.year

# ---------- build conservative point-in-time historical resource outcomes ----------
rp=(projects.merge(pa[['project_code','max_avail_ns']],on='project_code',how='left'))
rp=rp[(rp['planned_h']>0)&(rp['elapsed_h']>0)&rp['max_avail_ns'].notna()].copy()
rp['resource_outcome']=np.log(rp['planned_h']/rp['elapsed_h'])
rp['outcome_known_ns']=rp['max_avail_ns'].astype('int64')+MATURE_NS
# All resource histories are exposures, not causal individual performance estimates.

# user-project membership joined to projects with known resource outcome
ur=(up[['user_code','project_code']].drop_duplicates()
    .merge(rp[['project_code','team_id','resource_outcome','outcome_known_ns']],on='project_code',how='inner'))

# helper cumulative-history dictionaries: each key -> sorted known times and cumulative outcome sums

def make_hist(df,key):
    out={}
    for k,g in df.groupby(key,sort=False):
        g=g.sort_values('outcome_known_ns')
        t=g['outcome_known_ns'].to_numpy(dtype=np.int64)
        y=g['resource_outcome'].to_numpy(dtype=float)
        out[k]=(t,np.cumsum(y))
    return out

user_hist=make_hist(ur,'user_code')
org_hist=make_hist(rp,'team_id')
glob=rp.sort_values('outcome_known_ns')
glob_t=glob['outcome_known_ns'].to_numpy(dtype=np.int64)
glob_cs=np.cumsum(glob['resource_outcome'].to_numpy(dtype=float))

def hist_stats(hist,key,t_ns):
    z=hist.get(key)
    if z is None: return 0,0.0
    tt,cs=z
    n=int(np.searchsorted(tt,t_ns,side='left'))
    if n<=0: return 0,0.0
    return n,float(cs[n-1])

def global_stats(t_ns):
    n=int(np.searchsorted(glob_t,t_ns,side='left'))
    if n<=0: return 0,0.0
    return n,float(glob_cs[n-1])

# point-in-time sufficient statistics for every candidate observation
rows=[]
for r in ind.itertuples(index=False):
    t_ns=int(r.project_created_dt.value)
    un,us=hist_stats(user_hist,int(r.user_code),t_ns)
    on,osum=hist_stats(org_hist,r.team_id,t_ns)
    gn,gsum=global_stats(t_ns)
    um=us/un if un else np.nan
    om=osum/on if on else np.nan
    gm=gsum/gn if gn else 0.0
    # use organization mean only with a minimum of 3 known outcomes; otherwise global point-in-time prior
    prior=om if on>=3 and np.isfinite(om) else gm
    rows.append((int(r.project_code),int(r.user_code),int(r.is_observed_member),un,us,um,on,om,gn,gm,prior))
res=pd.DataFrame(rows,columns=['project_code','user_code','is_observed_member','prior_resource_n','prior_resource_sum','prior_resource_mean','org_resource_n','org_resource_mean','global_resource_n','global_resource_mean','resource_prior_mean'])

# EB posterior means for prespecified shrinkage strengths
TAUS=[1.,3.,5.,10.,20.,50.]
for tau in TAUS:
    col=f'eb_resource_tau{int(tau)}'
    res[col]=(res['prior_resource_sum']+tau*res['resource_prior_mean'])/(res['prior_resource_n']+tau)
    res[f'eb_unc_tau{int(tau)}']=1.0/np.sqrt(res['prior_resource_n']+tau)

# merge basic individual history and save candidate-level EB features
cand=(ind.merge(res,on=['project_code','user_code','is_observed_member'],how='left',validate='one_to_one'))
cand.to_pickle(f'{OUT}/candidate_resource_features.pkl')
cand.to_csv(f'{OUT}/candidate_resource_features.csv.gz',index=False,compression='gzip')

# aggregate only observed team members to project-level game features
obs=cand[cand['is_observed_member']==1].copy()
agg_base=(obs.groupby('project_code').agg(
    mean_prior_resource_n=('prior_resource_n','mean'),
    median_prior_resource_n=('prior_resource_n','median'),
    share_zero_resource_history=('prior_resource_n',lambda s:float((s==0).mean())),
    mean_resource_prior=('resource_prior_mean','mean'),
    mean_raw_resource_history=('prior_resource_mean','mean'),
).reset_index())
for tau in TAUS:
    a=obs.groupby('project_code').agg(**{
        f'mean_eb_resource_tau{int(tau)}':(f'eb_resource_tau{int(tau)}','mean'),
        f'mean_eb_unc_tau{int(tau)}':(f'eb_unc_tau{int(tau)}','mean'),
    }).reset_index()
    agg_base=agg_base.merge(a,on='project_code',how='left')

x=projfeat.merge(agg_base,on='project_code',how='left',validate='one_to_one')
x.to_pickle(f'{OUT}/intensive_eb_project_features.pkl')
x.to_csv(f'{OUT}/intensive_eb_project_features.csv',index=False)

# diagnostics
cand_diag={
 'candidate_rows':int(len(cand)),
 'observed_member_rows':int(len(obs)),
 'candidate_zero_resource_history_share':float((cand.prior_resource_n==0).mean()),
 'observed_zero_resource_history_share':float((obs.prior_resource_n==0).mean()),
 'candidate_resource_n_quantiles':cand.prior_resource_n.quantile([0,.25,.5,.75,.9,.95,.99,1]).to_dict(),
 'observed_resource_n_quantiles':obs.prior_resource_n.quantile([0,.25,.5,.75,.9,.95,.99,1]).to_dict(),
 'resource_project_count':int(len(rp)),
 'user_resource_exposures':int(len(ur)),
 'resource_outcome_quantiles':rp.resource_outcome.quantile([0,.01,.05,.25,.5,.75,.95,.99,1]).to_dict(),
 'project_share_all_members_zero_resource_history':float((obs.groupby('project_code').prior_resource_n.apply(lambda s:(s==0).all())).mean()),
 'project_share_any_member_zero_resource_history':float((obs.groupby('project_code').prior_resource_n.apply(lambda s:(s==0).any())).mean()),
}
with open(f'{OUT}/intensive_eb_audit.json','w') as f: json.dump(cand_diag,f,indent=2)
print('EB AUDIT')
print(json.dumps(cand_diag,indent=2))

# ---------- nested rolling-origin tournament ----------
ycol='outcome_log_B_over_C'
context=['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks']
card=['observed_team_size','pair_count']
# I1 = experience only; I2 = experience + collaboration breadth; I3 = experience + EB resource; I4 tests EB incremental to I2
fixed_models={
 'B1_card':context+card,
 'I1_exp':context+card+['mean_log_prior_projects'],
 'I2_exp_collab':context+card+['mean_log_prior_projects','mean_log_prior_collaborators'],
}
ALPHAS=[1e-4,1e-3,1e-2,.1,1,10,100,1000]

# fit/eval helpers
def fit_ridge(train,features,alpha):
    p=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=alpha))])
    p.fit(train[features],train[ycol])
    return p

def select_alpha(train,features):
    train=train.sort_values('project_created_dt').reset_index(drop=True)
    cv=TimeSeriesSplit(4)
    best=None
    for alpha in ALPHAS:
        vals=[]
        for ti,vi in cv.split(train):
            tr=train.iloc[ti]; va=train.iloc[vi]
            p=fit_ridge(tr,features,alpha); pr=p.predict(va[features])
            vals.append(mean_absolute_error(va[ycol],pr))
        score=float(np.mean(vals))
        if best is None or score<best[0]: best=(score,alpha)
    return best[1],best[0]

def select_tau_alpha(train,base_features,include_collab=False,include_unc=False):
    train=train.sort_values('project_created_dt').reset_index(drop=True)
    cv=TimeSeriesSplit(4)
    best=None
    for tau in TAUS:
        eb=f'mean_eb_resource_tau{int(tau)}'
        unc=f'mean_eb_unc_tau{int(tau)}'
        fs=base_features+[eb]+([unc] if include_unc else [])
        for alpha in ALPHAS:
            vals=[]
            for ti,vi in cv.split(train):
                tr=train.iloc[ti]; va=train.iloc[vi]
                p=fit_ridge(tr,fs,alpha); pr=p.predict(va[fs])
                vals.append(mean_absolute_error(va[ycol],pr))
            score=float(np.mean(vals))
            if best is None or score<best[0]: best=(score,tau,alpha,fs)
    return best

results=[]; preds=[]; coefs=[]
outer_years=[2022,2023,2024]
for yr in outer_years:
    tr=x[x.year<yr].sort_values('project_created_dt').copy()
    te=x[x.year==yr].sort_values('project_created_dt').copy()
    print('OUTER',yr,len(tr),len(te),flush=True)
    # fixed models
    for name,fs in fixed_models.items():
        alpha,cvmae=select_alpha(tr,fs)
        p=fit_ridge(tr,fs,alpha); pr=p.predict(te[fs])
        results.append((yr,name,alpha,np.nan,cvmae,mean_absolute_error(te[ycol],pr),mean_squared_error(te[ycol],pr)**.5,r2_score(te[ycol],pr),len(tr),len(te)))
        z=te[['project_code','team_id','project_created_dt',ycol,'overrun']].copy(); z['year']=yr;z['model']=name;z['prediction']=pr;preds.append(z)
        for f,c in zip(fs,p.named_steps['r'].coef_): coefs.append((yr,name,f,c))
    # EB models
    eb_specs={
       'I3_exp_EB':(context+card+['mean_log_prior_projects'],False),
       'I4_exp_collab_EB':(context+card+['mean_log_prior_projects','mean_log_prior_collaborators'],False),
       'I5_exp_collab_EB_unc':(context+card+['mean_log_prior_projects','mean_log_prior_collaborators'],True),
    }
    for name,(basefs,inc_unc) in eb_specs.items():
        cvmae,tau,alpha,fs=select_tau_alpha(tr,basefs,include_unc=inc_unc)
        p=fit_ridge(tr,fs,alpha); pr=p.predict(te[fs])
        results.append((yr,name,alpha,tau,cvmae,mean_absolute_error(te[ycol],pr),mean_squared_error(te[ycol],pr)**.5,r2_score(te[ycol],pr),len(tr),len(te)))
        z=te[['project_code','team_id','project_created_dt',ycol,'overrun']].copy(); z['year']=yr;z['model']=name;z['prediction']=pr;preds.append(z)
        for f,c in zip(fs,p.named_steps['r'].coef_): coefs.append((yr,name,f,c))

resdf=pd.DataFrame(results,columns=['year','model','alpha','tau','inner_cv_mae','mae','rmse','r2','n_train','n_test'])
pred=pd.concat(preds,ignore_index=True)
coef=pd.DataFrame(coefs,columns=['year','model','feature','coef_std'])

# aggregate OOF + unseen organization performance
agg=[]
for m,g in pred.groupby('model'):
    seen_before=[]
    for r in g.itertuples(index=False):
        train_orgs=set(x.loc[x.year<r.year,'team_id'])
        seen_before.append(r.team_id in train_orgs)
    gg=g.copy(); gg['seen_org_before']=seen_before
    unseen=gg[~gg.seen_org_before]
    agg.append((m,len(gg),mean_absolute_error(gg[ycol],gg.prediction),mean_squared_error(gg[ycol],gg.prediction)**.5,r2_score(gg[ycol],gg.prediction),
                len(unseen),mean_absolute_error(unseen[ycol],unseen.prediction) if len(unseen) else np.nan,
                mean_squared_error(unseen[ycol],unseen.prediction)**.5 if len(unseen) else np.nan))
agg=pd.DataFrame(agg,columns=['model','n_oof','mae','rmse','r2','n_unseen_org_projects','mae_unseen_org','rmse_unseen_org']).sort_values('mae')

# paired organization-cluster bootstrap using common OOF rows, comparing to I2 and B1
rng=np.random.default_rng(20260802)
def cluster_boot_delta(model,base='I2_exp_collab',B=5000):
    a=pred[pred.model==base][['project_code','team_id',ycol,'prediction']].rename(columns={'prediction':'base_pred'})
    b=pred[pred.model==model][['project_code','prediction']].rename(columns={'prediction':'model_pred'})
    d=a.merge(b,on='project_code',how='inner')
    d['base_abs']=(d[ycol]-d.base_pred).abs(); d['model_abs']=(d[ycol]-d.model_pred).abs()
    orgagg=d.groupby('team_id').agg(n=('project_code','size'),base_sum=('base_abs','sum'),model_sum=('model_abs','sum')).reset_index(drop=True)
    n_arr=orgagg['n'].to_numpy(float); b_arr=orgagg['base_sum'].to_numpy(float); m_arr=orgagg['model_sum'].to_numpy(float)
    k=len(orgagg)
    idx=rng.integers(0,k,size=(B,k))
    nn=n_arr[idx].sum(axis=1); sb=b_arr[idx].sum(axis=1); sm=m_arr[idx].sum(axis=1)
    vals=sm/nn-sb/nn
    obs=float(d.model_abs.mean()-d.base_abs.mean())
    return {'model':model,'base':base,'delta_mae':obs,'pct_delta':100*obs/d.base_abs.mean(),
            'ci2.5':float(np.quantile(vals,.025)),'median':float(np.median(vals)),'ci97.5':float(np.quantile(vals,.975)),
            'p_improve':float((vals<0).mean())}
boots=[]
for base in ['B1_card','I2_exp_collab']:
    for m in ['I1_exp','I2_exp_collab','I3_exp_EB','I4_exp_collab_EB','I5_exp_collab_EB_unc']:
        if m!=base: boots.append(cluster_boot_delta(m,base))
boot=pd.DataFrame(boots)

resdf.to_csv(f'{OUT}/intensive_eb_nested_results.csv',index=False)
pred.to_csv(f'{OUT}/intensive_eb_nested_predictions.csv',index=False)
coef.to_csv(f'{OUT}/intensive_eb_nested_coefficients.csv',index=False)
agg.to_csv(f'{OUT}/intensive_eb_nested_aggregate.csv',index=False)
boot.to_csv(f'{OUT}/intensive_eb_cluster_bootstrap.csv',index=False)

print('\nRESULTS\n',resdf.to_string(index=False))
print('\nAGG\n',agg.to_string(index=False))
print('\nBOOT\n',boot.to_string(index=False))
print('\nKEY COEFS\n',coef[coef.feature.str.contains('mean_log_prior|mean_eb_resource|mean_eb_unc')].pivot_table(index=['model','feature'],columns='year',values='coef_std').round(4).to_string())
