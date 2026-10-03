# PORTABLE PATH CONFIGURATION — added in Stage H2.7
from pathlib import Path as _Path
import os as _os
_REPRO_ROOT = _Path(_os.environ.get('GRYZZLY_REPRO_ROOT', _Path(__file__).resolve().parents[1])).resolve()
_DATA_DIR = _Path(_os.environ.get('GRYZZLY_DATA_DIR', _REPRO_ROOT / 'data')).resolve()
_STAGEF_DIR = _Path(_os.environ.get('GRYZZLY_STAGEF_DIR', _REPRO_ROOT / 'work' / 'stageF')).resolve()
_STAGEG_DIR = _Path(_os.environ.get('GRYZZLY_STAGEG_DIR', _REPRO_ROOT / 'work' / 'stageG')).resolve()
_STAGEF_DIR.mkdir(parents=True, exist_ok=True)
_STAGEG_DIR.mkdir(parents=True, exist_ok=True)

import os, json, math, itertools, warnings, time
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score, brier_score_loss, log_loss

warnings.filterwarnings('ignore')
BASE=str(_DATA_DIR); OUT=str(_STAGEG_DIR); os.makedirs(OUT,exist_ok=True)
DAY_NS=int(pd.Timedelta(days=1).value)
RNG=np.random.default_rng(20260802)

def parse_mixed(s): return pd.to_datetime(s,format='mixed',utc=True,errors='coerce')
def duration_to_hours(s): return pd.to_timedelta(s,errors='coerce').dt.total_seconds()/3600.0

def build_base():
    cache=os.path.join(OUT,'base_cache.npz')
    if os.path.exists(cache) and os.path.exists(os.path.join(OUT,'up.pkl')):
        print('Loading cached base...',flush=True)
        users=pd.read_pickle(f'{OUT}/users.pkl'); projects=pd.read_pickle(f'{OUT}/projects.pkl'); up=pd.read_pickle(f'{OUT}/up.pkl'); pa=pd.read_pickle(f'{OUT}/pa.pkl'); A=np.load(cache)
        return users,projects,up,pa,A
    print('Building base from raw Gryzzly...',flush=True)
    users=pd.read_csv(f'{BASE}/users.csv',usecols=['id','created_at','deleted_at','team_id']).rename(columns={'id':'user_id'})
    users['created_dt']=parse_mixed(users.created_at); users['deleted_dt']=parse_mixed(users.deleted_at); users['user_code']=np.arange(len(users),dtype=np.int32)
    user_code=dict(zip(users.user_id,users.user_code))
    projects=pd.read_csv(f'{BASE}/projects.csv',usecols=['id','created_at','team_id']).rename(columns={'id':'project_id'})
    projects['project_created_dt']=parse_mixed(projects.created_at); projects['project_code']=np.arange(len(projects),dtype=np.int32)
    proj_code=dict(zip(projects.project_id,projects.project_code))
    pc=pd.read_csv(f'{BASE}/projects_computed.csv',usecols=['id','planned_duration','elapsed_duration']).rename(columns={'id':'project_id'})
    pc['planned_h']=duration_to_hours(pc.planned_duration);pc['elapsed_h']=duration_to_hours(pc.elapsed_duration)
    projects=projects.merge(pc[['project_id','planned_h','elapsed_h']],on='project_id',how='left')
    tasks=pd.read_csv(f'{BASE}/tasks.csv',usecols=['id','project_id']).rename(columns={'id':'task_id'}); tasks['project_code']=tasks.project_id.map(proj_code)
    task_to_proj=dict(zip(tasks.task_id,tasks.project_code))
    arr={k:[] for k in ['user','proj','created','work','avail','dur']}; up_parts=[];pa_parts=[];nraw=nvalid=0;t0=time.time()
    for ci,ch in enumerate(pd.read_csv(f'{BASE}/declarations.zip',compression='zip',chunksize=250000,usecols=['created_at','date','duration','user_id','task_id'])):
        nraw+=len(ch); uc=ch.user_id.map(user_code); pj=ch.task_id.map(task_to_proj); cdt=parse_mixed(ch.created_at); wdt=pd.to_datetime(ch.date,format='%Y-%m-%d',utc=True,errors='coerce'); dur=ch.duration.astype('float64')/(3600e9)
        ok=uc.notna()&pj.notna()&cdt.notna()&wdt.notna()&(dur>0)&(dur<24)
        if not ok.any(): continue
        u=uc[ok].astype(np.int32).to_numpy(); p=pj[ok].astype(np.int32).to_numpy(); c=cdt[ok].astype('int64').to_numpy(); w=wdt[ok].astype('int64').to_numpy(); d=dur[ok].to_numpy(np.float32); av=np.maximum(c,w+DAY_NS)
        nvalid+=len(u)
        for k,v in [('user',u),('proj',p),('created',c),('work',w),('avail',av),('dur',d)]: arr[k].append(v)
        tmp=pd.DataFrame({'user_code':u,'project_code':p,'avail_ns':av,'decl_count':1,'hours':d})
        up_parts.append(tmp.groupby(['user_code','project_code'],sort=False).agg(first_avail_ns=('avail_ns','min'),decl_count=('decl_count','sum'),hours=('hours','sum')).reset_index())
        pp=pd.DataFrame({'project_code':p,'decl_created_ns':c,'work_ns':w,'avail_ns':av})
        pa_parts.append(pp.groupby('project_code',sort=False).agg(min_decl_created_ns=('decl_created_ns','min'),min_work_ns=('work_ns','min'),max_work_ns=('work_ns','max'),max_avail_ns=('avail_ns','max'),n_decls=('avail_ns','size')).reset_index())
        if ci%5==0: print(' chunk',ci,'raw',nraw,'valid',nvalid,'elapsed',round(time.time()-t0,1),flush=True)
    A={k:np.concatenate(v) for k,v in arr.items()}; np.savez_compressed(cache,**A)
    upa=pd.concat(up_parts,ignore_index=True); up=upa.groupby(['user_code','project_code'],sort=False).agg(first_avail_ns=('first_avail_ns','min'),decl_count=('decl_count','sum'),hours=('hours','sum')).reset_index()
    paa=pd.concat(pa_parts,ignore_index=True); pa=paa.groupby('project_code',sort=False).agg(min_decl_created_ns=('min_decl_created_ns','min'),min_work_ns=('min_work_ns','min'),max_work_ns=('max_work_ns','max'),max_avail_ns=('max_avail_ns','max'),n_decls=('n_decls','sum')).reset_index()
    users[['user_code','user_id','team_id','created_dt','deleted_dt']].to_pickle(f'{OUT}/users.pkl'); projects[['project_code','project_id','team_id','project_created_dt','planned_h','elapsed_h']].to_pickle(f'{OUT}/projects.pkl');up.to_pickle(f'{OUT}/up.pkl');pa.to_pickle(f'{OUT}/pa.pkl')
    with open(f'{OUT}/base_meta.json','w') as f:json.dump({'raw':nraw,'valid':nvalid,'created_end':pd.to_datetime(A['created'].max(),utc=True).isoformat()},f,indent=2)
    return users,projects,up,pa,np.load(cache)

users,projects,up,pa,A=build_base()
print('base',len(users),len(projects),len(up),len(A['user']),flush=True)

# structural prospective target cohort (target maturity fixed at 90 days)
proj=projects.merge(pa,on='project_code',how='left'); obs_end=pd.to_datetime(A['created'].max(),utc=True); cutoff_ns=(obs_end-pd.Timedelta(days=90)).value
team_sizes=up.groupby('project_code').user_code.nunique(); proj['team_size_obs']=proj.project_code.map(team_sizes)
pcreated_ns=proj.project_created_dt.astype('int64');pday_ns=proj.project_created_dt.dt.floor('D').astype('int64')
mask=(proj.planned_h.fillna(0)>0)&(proj.elapsed_h.fillna(0)>0)&proj.n_decls.notna()&(proj.max_avail_ns<=cutoff_ns)&proj.team_size_obs.between(2,20)&(proj.min_decl_created_ns>=pcreated_ns)&(proj.min_work_ns>=pday_ns)
struct=proj[mask].copy().sort_values('project_created_dt'); print('struct',len(struct),struct.team_id.nunique(),flush=True)
actual_team={int(pc):set(g.user_code.astype(int)) for pc,g in up[up.project_code.isin(struct.project_code)].groupby('project_code')}

# user lifecycle & team roster
team_users={t:g.user_code.astype(int).to_numpy() for t,g in users.groupby('team_id')}; user_created=users.set_index('user_code').created_dt.astype('int64').to_dict(); user_deleted={int(r.user_code):(None if pd.isna(r.deleted_dt) else int(r.deleted_dt.value)) for r in users.itertuples()}
# sorted event histories by user
uc=A['user'];av=A['avail'];wk=A['work'];cr=A['created']
order_av=np.lexsort((av,uc));uc_av=uc[order_av];av_s=av[order_av]
order_w=np.lexsort((wk,uc));uc_w=uc[order_w];wk_s=wk[order_w];cr_w=cr[order_w]
nusers=int(users.user_code.max())+1
def slices(sorted_users):return np.searchsorted(sorted_users,np.arange(nusers),side='left'),np.searchsorted(sorted_users,np.arange(nusers),side='right')
av_st,av_en=slices(uc_av);w_st,w_en=slices(uc_w)
def prior_decl_count(u,t_ns):
    s,e=av_st[u],av_en[u];return int(np.searchsorted(av_s[s:e],t_ns,side='left'))
def recent_active(u,t_dt,window):
    s,e=w_st[u],w_en[u]
    if e<=s:return False
    end=t_dt.floor('D').value;start=(t_dt.floor('D')-pd.Timedelta(days=window)).value; ar=wk_s[s:e];lo=np.searchsorted(ar,start,'left');hi=np.searchsorted(ar,end,'left')
    return bool(hi>lo and np.any(cr_w[s+lo:s+hi]<t_dt.value))
# prior project arrays per user
user_proj_hist={}
for u,g in up.groupby('user_code',sort=False):
    gg=g.sort_values('first_avail_ns');user_proj_hist[int(u)]=(gg.first_avail_ns.to_numpy(np.int64),gg.project_code.to_numpy(np.int32))
def prior_projects_count(u,t_ns):
    z=user_proj_hist.get(int(u));
    return 0 if z is None else int(np.searchsorted(z[0],t_ns,side='left'))

# initial leaf task counts for structural projects
tasks=pd.read_csv(f'{BASE}/tasks.csv',usecols=['project_id','created_at','is_container']); iscont=tasks.is_container.astype(str).str.lower().isin(['true','t','1']);tasks['created_dt']=parse_mixed(tasks.created_at); pct=struct.set_index('project_id').project_created_dt.to_dict();sel=tasks.project_id.isin(pct)&(~iscont)&tasks.created_dt.notna();tt=tasks.loc[sel,['project_id','created_dt']].copy();tt['target']=tt.project_id.map(pct);tt=tt[tt.created_dt<=tt.target];initial_map=tt.groupby('project_id').size().to_dict()

# resource histories with membership and participation-share weighting
rp=projects.merge(pa[['project_code','max_avail_ns']],on='project_code',how='left');rp=rp[(rp.planned_h>0)&(rp.elapsed_h>0)&rp.max_avail_ns.notna()].copy();rp['resource_outcome']=np.log(rp.planned_h/rp.elapsed_h)
proj_total_hours=up.groupby('project_code').hours.sum(); up2=up.copy(); up2['share_hours']=up2.hours/up2.project_code.map(proj_total_hours)

def make_hist_arrays(df,key,timecol,valcol,weightcol=None):
    out={}
    for k,g in df.groupby(key,sort=False):
        gg=g.sort_values(timecol);t=gg[timecol].to_numpy(np.int64);v=gg[valcol].to_numpy(float)
        if weightcol is None:w=np.ones(len(gg),float)
        else:w=gg[weightcol].to_numpy(float)
        out[k]=(t,np.cumsum(v*w),np.cumsum(w))
    return out

def hist_sumw(hist,key,t_ns):
    z=hist.get(key)
    if z is None:return 0.0,0.0
    tt,sv,sw=z;n=int(np.searchsorted(tt,t_ns,'left'))
    if n<=0:return 0.0,0.0
    return float(sv[n-1]),float(sw[n-1])

resource_cache={}
for mat in [60,90,120]:
    rr=rp[['project_code','team_id','resource_outcome','max_avail_ns']].copy();rr['known_ns']=rr.max_avail_ns.astype('int64')+mat*DAY_NS
    ur=up2[['user_code','project_code','share_hours']].merge(rr[['project_code','resource_outcome','known_ns']],on='project_code',how='inner')
    uh=make_hist_arrays(ur,'user_code','known_ns','resource_outcome',None); uhw=make_hist_arrays(ur,'user_code','known_ns','resource_outcome','share_hours')
    oh=make_hist_arrays(rr,'team_id','known_ns','resource_outcome',None); gh=make_hist_arrays(rr.assign(_g=0),'_g','known_ns','resource_outcome',None)
    resource_cache[mat]=(uh,uhw,oh,gh)
print('resource histories built',flush=True)

def eb_value(u,team,t_ns,mat,tau,weighted=False):
    uh,uhw,oh,gh=resource_cache[mat]; hist=uhw if weighted else uh
    us,uw=hist_sumw(hist,u,t_ns); os,ow=hist_sumw(oh,team,t_ns);gs,gw=hist_sumw(gh,0,t_ns)
    global_mean=gs/gw if gw>0 else 0.0; prior=os/ow if ow>=3 else global_mean
    return (us+tau*prior)/(uw+tau),uw,prior

def candidate_pool(row,min_decl,window):
    t=row.project_created_dt;t_ns=t.value;out=[]
    for u in team_users.get(row.team_id,[]):
        if user_created.get(int(u),2**63-1)>t_ns:continue
        dd=user_deleted.get(int(u));
        if dd is not None and dd<=t_ns:continue
        if prior_decl_count(int(u),t_ns)<min_decl:continue
        if not recent_active(int(u),t,window):continue
        out.append(int(u))
    return set(out)

# cache cohorts/candidate rows by window/min history
cohorts={}
def build_cohort(window,min_decl):
    key=(window,min_decl)
    if key in cohorts:return cohorts[key]
    pools={};kept=[]
    for r in struct.itertuples(index=False):
        pool=candidate_pool(r,min_decl,window);act=actual_team[int(r.project_code)]
        if len(pool)>=3 and act.issubset(pool):pools[int(r.project_code)]=pool;kept.append(int(r.project_code))
    ff=struct[struct.project_code.isin(kept)].copy().sort_values('project_created_dt');ff['observed_team_size']=ff.project_code.map(lambda pc:len(actual_team[int(pc)]));ff['candidate_pool_size']=ff.project_code.map(lambda pc:len(pools[int(pc)]));ff['outcome_log_B_over_C']=np.log(ff.planned_h/ff.elapsed_h);ff['overrun']=(ff.elapsed_h>ff.planned_h).astype(int);ff['year']=ff.project_created_dt.dt.year;ff['initial_leaf_tasks']=ff.project_id.map(initial_map).fillna(0).astype(int);ff['log_planned_h']=np.log(ff.planned_h);ff['log_candidate_pool']=np.log(ff.candidate_pool_size);ff['log1p_initial_leaf_tasks']=np.log1p(ff.initial_leaf_tasks)
    rows=[]
    for r in ff.itertuples(index=False):
        act=actual_team[int(r.project_code)]
        for u in pools[int(r.project_code)]: rows.append((int(r.project_code),u,int(u in act),r.team_id,r.project_created_dt))
    cand=pd.DataFrame(rows,columns=['project_code','user_code','is_observed_member','team_id','project_created_dt'])
    cand['prior_projects']=np.fromiter((prior_projects_count(u,t.value) for u,t in zip(cand.user_code,cand.project_created_dt)),dtype=np.int32,count=len(cand));cand['log_prior_projects']=np.log1p(cand.prior_projects)
    cohorts[key]=(ff,pools,cand);return cohorts[key]

ALPHAS=[1e-4,1e-3,1e-2,.1,1,10,100,1000]
CONTEXT=['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks']
Y='outcome_log_B_over_C'
def fit_ridge(tr,fs,a):
    p=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=a))]);p.fit(tr[fs],tr[Y]);return p
def select_alpha(tr,fs):
    tr=tr.sort_values('project_created_dt').reset_index(drop=True); cv=TimeSeriesSplit(4);best=(1e99,None)
    for a in ALPHAS:
        vals=[]
        for ti,vi in cv.split(tr):
            p=fit_ridge(tr.iloc[ti],fs,a);vals.append(mean_absolute_error(tr.iloc[vi][Y],p.predict(tr.iloc[vi][fs])))
        m=float(np.mean(vals))
        if m<best[0]:best=(m,a)
    return best[1],best[0]

def add_eb_project_features(ff,cand,mat=90,tau=5,weighted=False):
    cc=cand.copy();vals=[];ns=[];pri=[]
    for r in cc.itertuples(index=False):
        v,n,pr=eb_value(int(r.user_code),r.team_id,int(r.project_created_dt.value),mat,tau,weighted);vals.append(v);ns.append(n);pri.append(pr)
    cc['eb']=vals;cc['resource_exposure_n']=ns;cc['resource_prior']=pri
    obs=cc[cc.is_observed_member==1]
    ag=obs.groupby('project_code').agg(mean_log_prior_projects=('log_prior_projects','mean'),mean_eb=('eb','mean'),share_zero_resource=('resource_exposure_n',lambda s:float((s==0).mean()))).reset_index()
    return ff.merge(ag,on='project_code',how='inner',validate='one_to_one'),cc

def eval_models(x,tag):
    models={'L1_linear':CONTEXT+['observed_team_size'],'I1_exp':CONTEXT+['observed_team_size','mean_log_prior_projects'],'IER':CONTEXT+['observed_team_size','mean_log_prior_projects','mean_eb']}
    rows=[];pred=[];coef=[]
    for yr in [2022,2023,2024]:
        tr=x[x.year<yr].sort_values('project_created_dt');te=x[x.year==yr].sort_values('project_created_dt')
        if len(tr)<100 or len(te)<20:continue
        for m,fs in models.items():
            a,cv=select_alpha(tr,fs);p=fit_ridge(tr,fs,a);pr=p.predict(te[fs]);rows.append((tag,yr,m,len(tr),len(te),a,cv,mean_absolute_error(te[Y],pr),mean_squared_error(te[Y],pr)**.5,r2_score(te[Y],pr)))
            z=te[['project_code','team_id','project_created_dt',Y,'overrun']].copy();z['tag']=tag;z['year']=yr;z['model']=m;z['pred']=pr;pred.append(z)
            sc=p.named_steps['s'];rr=p.named_steps['r'];raw=rr.coef_/sc.scale_
            for f,c in zip(fs,raw):coef.append((tag,yr,m,f,c))
    rr=pd.DataFrame(rows,columns=['tag','year','model','n_train','n_test','alpha','inner_cv_mae','mae','rmse','r2']);pp=pd.concat(pred,ignore_index=True) if pred else pd.DataFrame();cc=pd.DataFrame(coef,columns=['tag','year','model','feature','coef_raw'])
    agg=[]
    for m,g in pp.groupby('model'):
        un=[]
        for r in g.itertuples(index=False):un.append(r.team_id not in set(x.loc[x.year<r.year,'team_id']))
        gu=g[np.array(un)]
        agg.append((tag,m,len(g),mean_absolute_error(g[Y],g.pred),mean_squared_error(g[Y],g.pred)**.5,r2_score(g[Y],g.pred),len(gu),mean_absolute_error(gu[Y],gu.pred) if len(gu) else np.nan))
    aa=pd.DataFrame(agg,columns=['tag','model','n_oof','mae','rmse','r2','n_unseen','mae_unseen'])
    return rr,aa,pp,cc

# PANEL A: candidate-pool/history robustness with fixed maturity=90,tau=5
all_res=[];all_agg=[];cohort_summary=[]
main_objects=None
for window in [28,56,90]:
  for md in [5,10,20]:
    ff,pools,cand=build_cohort(window,md); x,cc=add_eb_project_features(ff,cand,90,5,False);tag=f'w{window}_d{md}_m90_t5';rr,aa,pp,cf=eval_models(x,tag);all_res.append(rr);all_agg.append(aa);cohort_summary.append((window,md,len(ff),ff.team_id.nunique(),len(cand),float(ff.overrun.mean()),float(ff.candidate_pool_size.median()),float(x.share_zero_resource.mean())))
    if window==56 and md==10:main_objects=(ff,pools,cand,x,cc,pp,cf)
    print('cohort',tag,len(ff),aa[aa.model=='IER'][['mae','rmse','mae_unseen']].to_dict('records'),flush=True)

# PANEL B: maturity/tau robustness on main cohort
ff,pools,cand=build_cohort(56,10)
for mat in [60,90,120]:
  for tau in [3,5,10]:
    x,cc=add_eb_project_features(ff,cand,mat,tau,False);tag=f'w56_d10_m{mat}_t{tau}';rr,aa,pp,cf=eval_models(x,tag);all_res.append(rr);all_agg.append(aa);print('ebgrid',tag,aa[aa.model=='IER'][['mae','rmse','mae_unseen']].to_dict('records'),flush=True)

# PANEL C: participation-share weighted resource history, main config
xw,ccw=add_eb_project_features(ff,cand,90,5,True);rr,aa,ppw,cfw=eval_models(xw,'w56_d10_m90_t5_weighted');all_res.append(rr);all_agg.append(aa)

res=pd.concat(all_res,ignore_index=True);agg=pd.concat(all_agg,ignore_index=True);res.to_csv(f'{OUT}/robustness_folds.csv',index=False);agg.to_csv(f'{OUT}/robustness_aggregate.csv',index=False);pd.DataFrame(cohort_summary,columns=['window','min_decl','projects','organizations','candidate_rows','overrun_rate','median_pool','mean_project_share_zero_resource']).to_csv(f'{OUT}/cohort_robustness.csv',index=False)

# derive comparison deltas within each tag
comp=[]
for tag in agg.tag.unique():
    a=agg[agg.tag==tag].set_index('model')
    if {'L1_linear','I1_exp','IER'}.issubset(a.index):
        comp.append((tag,a.loc['IER','mae'],100*(a.loc['IER','mae']-a.loc['L1_linear','mae'])/a.loc['L1_linear','mae'],100*(a.loc['IER','mae']-a.loc['I1_exp','mae'])/a.loc['I1_exp','mae'],a.loc['IER','rmse'],a.loc['IER','mae_unseen']))
comp=pd.DataFrame(comp,columns=['tag','ier_mae','pct_vs_linear','pct_vs_exp','ier_rmse','ier_mae_unseen']);comp.to_csv(f'{OUT}/robustness_comparison.csv',index=False)

# main-model out-of-fold coefficients / candidate q and formation validation
ff,pools,cand,x,cc,pp_main,_=main_objects
fs=CONTEXT+['observed_team_size','mean_log_prior_projects','mean_eb']; coef_year={}; project_pred=[]
for yr in [2022,2023,2024]:
    tr=x[x.year<yr].sort_values('project_created_dt');te=x[x.year==yr].sort_values('project_created_dt');a,_=select_alpha(tr,fs);p=fit_ridge(tr,fs,a);sc=p.named_steps['s'];rr=p.named_steps['r'];raw=rr.coef_/sc.scale_; coef_year[yr]=(a,dict(zip(fs,raw)))
    pr=p.predict(te[fs]);z=te[['project_code','team_id','project_created_dt',Y,'observed_team_size']].copy();z['prediction']=pr;z['year']=yr;project_pred.append(z)

form=[]; member_rows=[]; mc_draws=2000
for yr in [2022,2023,2024]:
    b=coef_year[yr][1];be=b['mean_log_prior_projects'];br=b['mean_eb'];pcs=set(x.loc[x.year==yr,'project_code'])
    for pc,g in cc[cc.project_code.isin(pcs)].groupby('project_code'):
        g=g.copy();g['q']=be*g.log_prior_projects+br*g.eb;obs=g[g.is_observed_member==1];k=len(obs);N=len(g)
        if k<2 or N<k:continue
        obsmean=float(obs.q.mean());poolmean=float(g.q.mean());top=g.nlargest(k,'q');optmean=float(top.q.mean());gap=optmean-obsmean
        # deterministic project-specific MC seed
        rg=np.random.default_rng(1000003+int(pc));arr=g.q.to_numpy(float);means=np.empty(mc_draws)
        for z in range(mc_draws): means[z]=arr[rg.choice(N,size=k,replace=False)].mean()
        percentile=float((means<=obsmean).mean());p_hi=float((1+np.sum(means>=obsmean))/(mc_draws+1));p_opt=float((1+np.sum(means>=optmean-1e-12))/(mc_draws+1))
        obsset=set(obs.user_code.astype(int));topset=set(top.user_code.astype(int));inter=len(obsset&topset);jac=inter/len(obsset|topset)
        form.append((pc,yr,g.team_id.iloc[0],k,N,obsmean,poolmean,optmean,obsmean-poolmean,gap,percentile,p_hi,jac,inter))
        for r in g.itertuples(index=False):member_rows.append((pc,yr,int(r.user_code),int(r.is_observed_member),float(r.q),float(r.eb),int(r.prior_projects)))
form=pd.DataFrame(form,columns=['project_code','year','team_id','observed_n','pool_n','observed_mean_q','pool_mean_q','optimal_mean_q','observed_minus_random_expectation','optimal_gap','observed_random_percentile','random_p_upper','jaccard_observed_optimal','overlap_n'])
form.to_csv(f'{OUT}/formation_validation_projects.csv',index=False);pd.DataFrame(member_rows,columns=['project_code','year','user_code','is_observed_member','q','eb_resource','prior_projects']).to_csv(f'{OUT}/formation_candidate_scores.csv',index=False)
# organization-cluster bootstrap formation contrasts
org=form.groupby('team_id').agg(n=('project_code','size'),sum_obsrand=('observed_minus_random_expectation','sum'),sum_gap=('optimal_gap','sum'),sum_pct=('observed_random_percentile','sum')).reset_index();B=10000;ix=RNG.integers(0,len(org),(B,len(org)));nn=org.n.to_numpy(float);N=nn[ix].sum(1)
boot_or=org.sum_obsrand.to_numpy()[ix].sum(1)/N;boot_gap=org.sum_gap.to_numpy()[ix].sum(1)/N;boot_pct=org.sum_pct.to_numpy()[ix].sum(1)/N
formation_summary={
 'projects':int(len(form)),'organizations':int(form.team_id.nunique()),'share_observed_above_pool_mean':float((form.observed_minus_random_expectation>0).mean()),'mean_observed_minus_random':float(form.observed_minus_random_expectation.mean()),'median_observed_random_percentile':float(form.observed_random_percentile.median()),'share_observed_top_quartile_random':float((form.observed_random_percentile>=.75).mean()),'share_observed_exact_optimal':float((form.jaccard_observed_optimal==1).mean()),'median_jaccard_optimal':float(form.jaccard_observed_optimal.median()),'mean_optimal_gap':float(form.optimal_gap.mean()),'median_optimal_gap':float(form.optimal_gap.median()),'ci_obs_minus_random':[float(np.quantile(boot_or,.025)),float(np.quantile(boot_or,.975))],'ci_optimal_gap':[float(np.quantile(boot_gap,.025)),float(np.quantile(boot_gap,.975))],'ci_mean_random_percentile':[float(np.quantile(boot_pct,.025)),float(np.quantile(boot_pct,.975))]
}
with open(f'{OUT}/formation_validation_summary.json','w') as f:json.dump(formation_summary,f,indent=2)

# Shapley attribution for observed OOF teams, using candidate q; analytical intensive mean game
sh=[]
for pc,g in pd.DataFrame(member_rows,columns=['project_code','year','user_code','is_observed_member','q','eb_resource','prior_projects']).query('is_observed_member==1').groupby('project_code'):
    q=g.q.to_numpy(float);n=len(q);H=sum(1/j for j in range(1,n+1));tot=q.sum()
    for r in g.itertuples(index=False):
        if n==1:phi=float(r.q)
        else:phi=(H/n)*r.q - ((H-1)/(n*(n-1)))*(tot-r.q)
        sh.append((pc,r.year,r.user_code,n,r.q,phi))
sh=pd.DataFrame(sh,columns=['project_code','year','user_code','team_size','q','shapley_phi']);sh.to_csv(f'{OUT}/shapley_observed_teams.csv',index=False)
# verify efficiency and ranking
chk=sh.groupby('project_code').agg(sum_phi=('shapley_phi','sum'),mean_q=('q','mean')).reset_index();maxerr=float((chk.sum_phi-chk.mean_q).abs().max())
sp=[]
for pc,g in sh.groupby('project_code'):
    if len(g)>1:sp.append(g[['q','shapley_phi']].corr(method='spearman').iloc[0,1])
shapley_summary={'rows':int(len(sh)),'projects':int(sh.project_code.nunique()),'max_efficiency_error':maxerr,'median_within_team_spearman_q_phi':float(np.nanmedian(sp)),'share_negative_phi':float((sh.shapley_phi<0).mean()),'phi_quantiles':{str(k):float(v) for k,v in sh.shapley_phi.quantile([.01,.05,.25,.5,.75,.95,.99]).items()}}
with open(f'{OUT}/shapley_summary.json','w') as f:json.dump(shapley_summary,f,indent=2)

# binary robustness, main config only, rolling years logistic with inner C via temporal CV
Cs=[.01,.1,1,10,100];brows=[];bpred=[]
for yr in [2022,2023,2024]:
    tr=x[x.year<yr].sort_values('project_created_dt').reset_index(drop=True);te=x[x.year==yr].sort_values('project_created_dt');best=(1e99,None)
    for C in Cs:
        vals=[];cv=TimeSeriesSplit(4)
        for ti,vi in cv.split(tr):
            p=Pipeline([('s',StandardScaler()),('l',LogisticRegression(C=C,max_iter=5000))]);p.fit(tr.iloc[ti][fs],tr.iloc[ti].overrun);pr=p.predict_proba(tr.iloc[vi][fs])[:,1];vals.append(log_loss(tr.iloc[vi].overrun,pr))
        if np.mean(vals)<best[0]:best=(np.mean(vals),C)
    p=Pipeline([('s',StandardScaler()),('l',LogisticRegression(C=best[1],max_iter=5000))]);p.fit(tr[fs],tr.overrun);pr=p.predict_proba(te[fs])[:,1]
    brows.append((yr,best[1],roc_auc_score(te.overrun,pr),average_precision_score(te.overrun,pr),brier_score_loss(te.overrun,pr),log_loss(te.overrun,pr),len(te)))
    zz=te[['project_code','team_id','overrun']].copy();zz['year']=yr;zz['prob']=pr;bpred.append(zz)
bres=pd.DataFrame(brows,columns=['year','C','roc_auc','avg_precision','brier','log_loss','n']);bp=pd.concat(bpred);bagg={'n':len(bp),'roc_auc':roc_auc_score(bp.overrun,bp.prob),'avg_precision':average_precision_score(bp.overrun,bp.prob),'brier':brier_score_loss(bp.overrun,bp.prob),'log_loss':log_loss(bp.overrun,bp.prob)};bres.to_csv(f'{OUT}/binary_main_folds.csv',index=False);pd.DataFrame([bagg]).to_csv(f'{OUT}/binary_main_aggregate.csv',index=False)

# write mathematical/model audit
with open(f'{OUT}/stageG_summary.json','w') as f:json.dump({'formation':formation_summary,'shapley':shapley_summary,'binary':bagg,'main_coefficients':coef_year},f,indent=2,default=str)
print('\nROBUSTNESS COMPARISON\n',comp.sort_values('ier_mae').to_string(index=False))
print('\nFORMATION\n',json.dumps(formation_summary,indent=2))
print('\nSHAPLEY\n',json.dumps(shapley_summary,indent=2))
print('\nBINARY\n',bagg)
print('DONE Stage G',flush=True)
