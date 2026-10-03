# PORTABLE PATH CONFIGURATION — added in Stage H2.7
from pathlib import Path as _Path
import os as _os
_REPRO_ROOT = _Path(_os.environ.get('GRYZZLY_REPRO_ROOT', _Path(__file__).resolve().parents[1])).resolve()
_DATA_DIR = _Path(_os.environ.get('GRYZZLY_DATA_DIR', _REPRO_ROOT / 'data')).resolve()
_STAGEF_DIR = _Path(_os.environ.get('GRYZZLY_STAGEF_DIR', _REPRO_ROOT / 'work' / 'stageF')).resolve()
_STAGEG_DIR = _Path(_os.environ.get('GRYZZLY_STAGEG_DIR', _REPRO_ROOT / 'work' / 'stageG')).resolve()
_STAGEF_DIR.mkdir(parents=True, exist_ok=True)
_STAGEG_DIR.mkdir(parents=True, exist_ok=True)

import os, json, math, gc
from collections import defaultdict
import numpy as np
import pandas as pd

BASE=str(_DATA_DIR)
OUT=str(_STAGEF_DIR)
os.makedirs(OUT,exist_ok=True)
DECL=os.path.join(BASE,'declarations.zip')

# ---------- helpers ----------
def parse_mixed(s):
    return pd.to_datetime(s, format='mixed', utc=True, errors='coerce')

def duration_to_hours(s):
    return pd.to_timedelta(s, errors='coerce').dt.total_seconds()/3600.0

# ---------- static tables ----------
users=pd.read_csv(f'{BASE}/users.csv', usecols=['id','created_at','deleted_at','team_id'])
users=users.rename(columns={'id':'user_id'})
users['created_dt']=parse_mixed(users['created_at'])
users['deleted_dt']=parse_mixed(users['deleted_at'])
user_ids=users['user_id'].tolist(); user_code={u:i for i,u in enumerate(user_ids)}
users['user_code']=np.arange(len(users),dtype=np.int32)

projects=pd.read_csv(f'{BASE}/projects.csv', usecols=['id','created_at','team_id']).rename(columns={'id':'project_id'})
projects['project_created_dt']=parse_mixed(projects['created_at'])
proj_ids=projects['project_id'].tolist(); proj_code={p:i for i,p in enumerate(proj_ids)}
projects['project_code']=np.arange(len(projects),dtype=np.int32)

pc=pd.read_csv(f'{BASE}/projects_computed.csv', usecols=['id','planned_duration','elapsed_duration']).rename(columns={'id':'project_id'})
pc['planned_h']=duration_to_hours(pc['planned_duration'])
pc['elapsed_h']=duration_to_hours(pc['elapsed_duration'])
projects=projects.merge(pc[['project_id','planned_h','elapsed_h']],on='project_id',how='left')

# task -> project code
tasks=pd.read_csv(f'{BASE}/tasks.csv', usecols=['id','project_id']).rename(columns={'id':'task_id'})
tasks['project_code']=tasks['project_id'].map(proj_code)
task_to_proj=dict(zip(tasks['task_id'],tasks['project_code']))

# numpy containers for declaration-event history
arr_user=[]; arr_proj=[]; arr_created=[]; arr_work=[]; arr_avail=[]; arr_dur=[]
# chunk-reduced user-project records
up_parts=[]
# project aggregate parts
pa_parts=[]

n_raw=0; n_valid=0
chunksize=250_000
for ci,ch in enumerate(pd.read_csv(DECL, chunksize=chunksize, usecols=['created_at','date','duration','user_id','task_id'])):
    n_raw += len(ch)
    ch['user_code']=ch['user_id'].map(user_code)
    ch['project_code']=ch['task_id'].map(task_to_proj)
    cdt=parse_mixed(ch['created_at'])
    wdt=pd.to_datetime(ch['date'],format='%Y-%m-%d',utc=True,errors='coerce')
    dur=ch['duration'].astype('float64')/(3600.0*1e9)
    valid=(ch['user_code'].notna() & ch['project_code'].notna() & cdt.notna() & wdt.notna() & (dur>0) & (dur<24))
    if not valid.any():
        continue
    vc=ch.loc[valid,['user_code','project_code']].copy()
    vc['user_code']=vc['user_code'].astype(np.int32)
    vc['project_code']=vc['project_code'].astype(np.int32)
    cdt=cdt.loc[valid]
    wdt=wdt.loc[valid]
    dur=dur.loc[valid]
    # available at: conservatively next calendar day after work date, or actual record creation, whichever is later
    nextday=wdt + pd.Timedelta(days=1)
    av=pd.Series(np.maximum(cdt.astype('int64').to_numpy(), nextday.astype('int64').to_numpy()), index=cdt.index, dtype='int64')
    n_valid += len(vc)
    arr_user.append(vc['user_code'].to_numpy(dtype=np.int32))
    arr_proj.append(vc['project_code'].to_numpy(dtype=np.int32))
    arr_created.append(cdt.astype('int64').to_numpy())
    arr_work.append(wdt.astype('int64').to_numpy())
    arr_avail.append(av.to_numpy(dtype=np.int64))
    arr_dur.append(dur.to_numpy(dtype=np.float32))
    # user-project earliest availability and count
    tmp=vc.copy(); tmp['avail_ns']=av.to_numpy(); tmp['decl_count']=1
    up=tmp.groupby(['user_code','project_code'],sort=False).agg(first_avail_ns=('avail_ns','min'),decl_count=('decl_count','sum')).reset_index()
    up_parts.append(up)
    # project temporal aggregates
    pp=pd.DataFrame({'project_code':vc['project_code'].to_numpy(),
                     'decl_created_ns':cdt.astype('int64').to_numpy(),
                     'work_ns':wdt.astype('int64').to_numpy(),
                     'avail_ns':av.to_numpy()})
    pa=pp.groupby('project_code',sort=False).agg(
        min_decl_created_ns=('decl_created_ns','min'),
        min_work_ns=('work_ns','min'),
        max_work_ns=('work_ns','max'),
        max_avail_ns=('avail_ns','max'),
        n_decls=('avail_ns','size')
    ).reset_index()
    pa_parts.append(pa)
    if ci%4==0:
        print(f'chunk {ci}: raw={n_raw:,}, valid={n_valid:,}',flush=True)

# combine arrays
A={
 'user_code':np.concatenate(arr_user),
 'project_code':np.concatenate(arr_proj),
 'created_ns':np.concatenate(arr_created),
 'work_ns':np.concatenate(arr_work),
 'avail_ns':np.concatenate(arr_avail),
 'duration_h':np.concatenate(arr_dur),
}
np.savez_compressed(f'{OUT}/declaration_events.npz',**A)
print('saved declaration events',len(A['user_code']),flush=True)
# free parts arrays lists after npz
del arr_user,arr_proj,arr_created,arr_work,arr_avail,arr_dur; gc.collect()

# reduce user-project summaries
up_all=pd.concat(up_parts,ignore_index=True)
up=(up_all.groupby(['user_code','project_code'],sort=False)
    .agg(first_avail_ns=('first_avail_ns','min'),decl_count=('decl_count','sum')).reset_index())
up.to_pickle(f'{OUT}/user_project_history.pkl')
print('user-project',len(up),flush=True)
del up_all,up_parts; gc.collect()

# reduce project aggs
pa_all=pd.concat(pa_parts,ignore_index=True)
pa=(pa_all.groupby('project_code',sort=False).agg(
    min_decl_created_ns=('min_decl_created_ns','min'),
    min_work_ns=('min_work_ns','min'),
    max_work_ns=('max_work_ns','max'),
    max_avail_ns=('max_avail_ns','max'),
    n_decls=('n_decls','sum')).reset_index())
pa.to_pickle(f'{OUT}/project_activity.pkl')
print('project activity',len(pa),flush=True)

# save static compact tables
users[['user_code','user_id','team_id','created_dt','deleted_dt']].to_pickle(f'{OUT}/users_static.pkl')
projects[['project_code','project_id','team_id','project_created_dt','planned_h','elapsed_h']].to_pickle(f'{OUT}/projects_static.pkl')

meta={'raw_declarations':int(n_raw),'valid_declarations':int(n_valid),
      'observation_end_created_utc':pd.to_datetime(A['created_ns'].max(),utc=True).isoformat(),
      'observation_end_work_utc':pd.to_datetime(A['work_ns'].max(),utc=True).isoformat(),
      'max_available_utc':pd.to_datetime(A['avail_ns'].max(),utc=True).isoformat(),
      'n_user_project':int(len(up))}
with open(f'{OUT}/base_meta.json','w') as f: json.dump(meta,f,indent=2)
print(json.dumps(meta,indent=2),flush=True)
