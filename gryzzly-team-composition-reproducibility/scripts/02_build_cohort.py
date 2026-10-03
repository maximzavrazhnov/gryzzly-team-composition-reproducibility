# PORTABLE PATH CONFIGURATION — added in Stage H2.7
from pathlib import Path as _Path
import os as _os
_REPRO_ROOT = _Path(_os.environ.get('GRYZZLY_REPRO_ROOT', _Path(__file__).resolve().parents[1])).resolve()
_DATA_DIR = _Path(_os.environ.get('GRYZZLY_DATA_DIR', _REPRO_ROOT / 'data')).resolve()
_STAGEF_DIR = _Path(_os.environ.get('GRYZZLY_STAGEF_DIR', _REPRO_ROOT / 'work' / 'stageF')).resolve()
_STAGEG_DIR = _Path(_os.environ.get('GRYZZLY_STAGEG_DIR', _REPRO_ROOT / 'work' / 'stageG')).resolve()
_STAGEF_DIR.mkdir(parents=True, exist_ok=True)
_STAGEG_DIR.mkdir(parents=True, exist_ok=True)

import os,json,math,gc
from collections import defaultdict
import numpy as np
import pandas as pd

OUT=str(_STAGEF_DIR)
users=pd.read_pickle(f'{OUT}/users_static.pkl')
projects=pd.read_pickle(f'{OUT}/projects_static.pkl')
pa=pd.read_pickle(f'{OUT}/project_activity.pkl')
up=pd.read_pickle(f'{OUT}/user_project_history.pkl')
A=np.load(f'{OUT}/declaration_events.npz')

# Project temporal eligibility
proj=projects.merge(pa,on='project_code',how='left')
obs_end=pd.to_datetime(A['created_ns'].max(),utc=True)
cutoff=obs_end-pd.Timedelta(days=90)
cutoff_ns=cutoff.value
proj['team_size_obs']=proj['project_code'].map(up.groupby('project_code')['user_code'].nunique())
proj['has_decls']=proj['n_decls'].fillna(0)>0
proj['positive_budget']=proj['planned_h'].fillna(0)>0
proj['positive_elapsed']=proj['elapsed_h'].fillna(0)>0
proj['mature90']=proj['max_avail_ns'].fillna(np.iinfo(np.int64).max)<=cutoff_ns
# prospective: no record creation before project exact timestamp; no work-day before project calendar day
pcreated_ns=proj['project_created_dt'].astype('int64')
pday_ns=proj['project_created_dt'].dt.floor('D').astype('int64')
proj['prospective_record']=proj['min_decl_created_ns'].fillna(np.iinfo(np.int64).min)>=pcreated_ns
proj['prospective_workday']=proj['min_work_ns'].fillna(np.iinfo(np.int64).min)>=pday_ns
proj['size_2_20']=proj['team_size_obs'].between(2,20,inclusive='both')

funnel=[]
mask=np.ones(len(proj),dtype=bool)
def add(label,cond):
    global mask
    mask=mask & cond.fillna(False).to_numpy()
    funnel.append((label,int(mask.sum()),int(proj.loc[mask,'team_id'].nunique())))
add('all_project_records',pd.Series(True,index=proj.index))
add('planned_duration_gt_0',proj['positive_budget'])
add('has_valid_declarations',proj['has_decls'])
add('elapsed_duration_gt_0',proj['positive_elapsed'])
add('mature_90d_by_available_at',proj['mature90'])
add('observed_team_size_2_20',proj['size_2_20'])
add('no_declaration_record_before_project_creation',proj['prospective_record'])
add('no_workday_before_project_calendar_day',proj['prospective_workday'])
struct=proj.loc[mask].copy()
print('Funnel before candidate pool')
for r in funnel: print(r)
print('obs_end',obs_end,'cutoff',cutoff,'struct',len(struct),flush=True)

# actual team sets
actual_team={int(pc):set(g['user_code'].astype(int)) for pc,g in up[up['project_code'].isin(struct['project_code'])].groupby('project_code')}

# team roster from users
team_users={t:g['user_code'].astype(int).to_numpy() for t,g in users.groupby('team_id')}
user_created_ns=users.set_index('user_code')['created_dt'].astype('int64').to_dict()
# NaT int64 is min; create deletion map separately
del_dt=users.set_index('user_code')['deleted_dt']
user_deleted_ns={int(i):(None if pd.isna(v) else int(v.value)) for i,v in del_dt.items()}

# per-user declaration histories: available times; and work-day records with creation times
uc=A['user_code']; av=A['avail_ns']; wk=A['work_ns']; cr=A['created_ns']
# sort once by user then available; separately user then work
order_av=np.lexsort((av,uc)); uc_av=uc[order_av]; av_s=av[order_av]
order_w=np.lexsort((wk,uc)); uc_w=uc[order_w]; wk_s=wk[order_w]; cr_w=cr[order_w]
# index slices
nusers=int(users['user_code'].max())+1

def slices_for(sorted_users):
    starts=np.searchsorted(sorted_users,np.arange(nusers),side='left')
    ends=np.searchsorted(sorted_users,np.arange(nusers),side='right')
    return starts,ends
av_st,av_en=slices_for(uc_av); w_st,w_en=slices_for(uc_w)

def prior_decl_count(u,t_ns):
    s,e=av_st[u],av_en[u]
    return int(np.searchsorted(av_s[s:e],t_ns,side='left'))

def recent_active(u,t_dt,window_days=56):
    # Use work calendar dates strictly before target calendar day to avoid same-day ambiguity,
    # and only records already physically created before project creation.
    s,e=w_st[u],w_en[u]
    if e<=s: return False
    end_day=t_dt.floor('D').value
    start_day=(t_dt.floor('D')-pd.Timedelta(days=window_days)).value
    arr=wk_s[s:e]
    lo=np.searchsorted(arr,start_day,side='left'); hi=np.searchsorted(arr,end_day,side='left')
    if hi<=lo: return False
    return bool(np.any(cr_w[s+lo:s+hi] < t_dt.value))

def candidate_pool(row,min_decl,window=56):
    t=row.project_created_dt; t_ns=t.value
    roster=team_users.get(row.team_id,[])
    out=[]
    for u in roster:
        c=user_created_ns.get(int(u),np.iinfo(np.int64).max)
        if c>t_ns: continue
        d=user_deleted_ns.get(int(u))
        if d is not None and d<=t_ns: continue
        if prior_decl_count(int(u),t_ns)<min_decl: continue
        if not recent_active(int(u),t,window): continue
        out.append(int(u))
    return set(out)

thresholds=[5,10,20,50]
scan=[]; pools_by_m={}
for m in thresholds:
    kept=[]; pools={}
    for row in struct.itertuples(index=False):
        pool=candidate_pool(row,m,56)
        act=actual_team.get(int(row.project_code),set())
        ok=(len(pool)>=3 and act.issubset(pool))
        if ok:
            kept.append(int(row.project_code)); pools[int(row.project_code)]=pool
    sdf=struct[struct['project_code'].isin(kept)]
    scan.append((m,len(sdf),sdf['team_id'].nunique(),float((sdf['elapsed_h']>sdf['planned_h']).mean()) if len(sdf) else np.nan,
                 float(np.median([len(pools[k]) for k in kept])) if kept else np.nan))
    pools_by_m[m]=pools
    print('threshold',scan[-1],flush=True)

# prespecified: strictest threshold with >=2500 projects and >=150 organizations
eligible=[r for r in scan if r[1]>=2500 and r[2]>=150]
if not eligible:
    raise RuntimeError('No threshold satisfies prespecified coverage rule')
chosen=max(eligible,key=lambda x:x[0])[0]
pools=pools_by_m[chosen]
final=struct[struct['project_code'].isin(pools.keys())].copy().sort_values('project_created_dt')
final['observed_team_size']=final['project_code'].map(lambda x:len(actual_team[int(x)]))
final['candidate_pool_size']=final['project_code'].map(lambda x:len(pools[int(x)]))
final['outcome_log_B_over_C']=np.log(final['planned_h']/final['elapsed_h'])
final['overrun']=(final['elapsed_h']>final['planned_h']).astype(int)

# temporal 70/15/15 labels by sorted project order
n=len(final); ntr=int(np.floor(.70*n)); nva=int(np.floor(.15*n))
labels=np.array(['test']*n,dtype=object); labels[:ntr]='train'; labels[ntr:ntr+nva]='validation'
final['split']=labels

# save candidate memberships
rows=[]
for pc,pool in pools.items():
    if pc not in set(final['project_code']): continue
    act=actual_team[pc]
    for u in pool:
        rows.append((pc,u,int(u in act)))
cand=pd.DataFrame(rows,columns=['project_code','user_code','is_observed_member'])
final.to_pickle(f'{OUT}/final_projects.pkl')
cand.to_pickle(f'{OUT}/candidate_membership.pkl')

# save funnel and meta
meta={'observation_end':obs_end.isoformat(),'maturity_cutoff':cutoff.isoformat(),'chosen_min_prior_declarations':int(chosen),
      'activity_window_days':56,'final_projects':int(len(final)),'final_organizations':int(final.team_id.nunique()),
      'candidate_rows':int(len(cand)),'unique_candidates':int(cand.user_code.nunique()),
      'overrun_rate':float(final.overrun.mean()),
      'threshold_scan':[{'min_decl':int(r[0]),'projects':int(r[1]),'organizations':int(r[2]),'overrun_rate':r[3],'median_pool':r[4]} for r in scan],
      'funnel':[{'step':r[0],'projects':r[1],'organizations':r[2]} for r in funnel]}
with open(f'{OUT}/cohort_meta.json','w') as f: json.dump(meta,f,indent=2)
print(json.dumps(meta,indent=2),flush=True)
print(final.groupby('split').agg(projects=('project_code','size'),organizations=('team_id','nunique'),overrun=('overrun','mean'),
                                 median_team=('observed_team_size','median'),median_pool=('candidate_pool_size','median')).to_string())
