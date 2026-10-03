# PORTABLE PATH CONFIGURATION — added in Stage H2.7
from pathlib import Path as _Path
import os as _os
_REPRO_ROOT = _Path(_os.environ.get('GRYZZLY_REPRO_ROOT', _Path(__file__).resolve().parents[1])).resolve()
_DATA_DIR = _Path(_os.environ.get('GRYZZLY_DATA_DIR', _REPRO_ROOT / 'data')).resolve()
_STAGEF_DIR = _Path(_os.environ.get('GRYZZLY_STAGEF_DIR', _REPRO_ROOT / 'work' / 'stageF')).resolve()
_STAGEG_DIR = _Path(_os.environ.get('GRYZZLY_STAGEG_DIR', _REPRO_ROOT / 'work' / 'stageG')).resolve()
_STAGEF_DIR.mkdir(parents=True, exist_ok=True)
_STAGEG_DIR.mkdir(parents=True, exist_ok=True)

import os,json,itertools,math,gc
from collections import defaultdict
import numpy as np
import pandas as pd

OUT=str(_STAGEF_DIR)
final=pd.read_pickle(f'{OUT}/final_projects.pkl')
cand=pd.read_pickle(f'{OUT}/candidate_membership.pkl')
up=pd.read_pickle(f'{OUT}/user_project_history.pkl')
users=pd.read_pickle(f'{OUT}/users_static.pkl')

# user -> sorted project availability arrays and lookup dicts
user_hist={}
for u,g in up.groupby('user_code',sort=False):
    gg=g.sort_values('first_avail_ns')
    user_hist[int(u)]=(gg['first_avail_ns'].to_numpy(dtype=np.int64),gg['project_code'].to_numpy(dtype=np.int32))
# project -> arrays of (user, first_avail)
proj_hist={}
for pc,g in up.groupby('project_code',sort=False):
    proj_hist[int(pc)]=(g['user_code'].to_numpy(dtype=np.int32),g['first_avail_ns'].to_numpy(dtype=np.int64))

cand_by_proj={int(pc):g.sort_values('user_code') for pc,g in cand.groupby('project_code',sort=False)}

ind_rows=[]; pair_rows=[]
for idx,row in enumerate(final.itertuples(index=False)):
    pc=int(row.project_code); t_ns=int(row.project_created_dt.value)
    cg=cand_by_proj[pc]
    pool=cg['user_code'].astype(int).tolist()
    actual=set(cg.loc[cg['is_observed_member']==1,'user_code'].astype(int))
    # portfolio sets and timestamp maps
    portfolios={}; ptimes={}; collab_counts={}
    for u in pool:
        times,projs=user_hist.get(u,(np.array([],dtype=np.int64),np.array([],dtype=np.int32)))
        k=int(np.searchsorted(times,t_ns,side='left'))
        ps=projs[:k]
        ts=times[:k]
        pset=set(map(int,ps))
        portfolios[u]=pset
        ptimes[u]={int(p):int(tt) for p,tt in zip(ps,ts)}
        # collaboration breadth: union of co-members whose own first project record was available before t
        coll=set()
        for q in pset:
            us,avs=proj_hist[q]
            # all users with their first evidence on q available before target
            valid_us=us[avs<t_ns]
            coll.update(map(int,valid_us))
        coll.discard(u)
        collab_counts[u]=len(coll)
        ind_rows.append((pc,u,int(u in actual),len(pset),len(coll),math.log1p(len(pset)),math.log1p(len(coll))))
    # pairwise
    for u,v in itertools.combinations(pool,2):
        pu=portfolios[u]; pv=portfolios[v]
        inter_set=pu & pv
        inter=len(inter_set); a=len(pu); b=len(pv); union=a+b-inter
        F=math.log1p(inter)
        D_j=(1.0-inter/union) if union>0 else np.nan
        # Gryzzly-like uniqueness ratio: distinct past projects / total individual past-project exposures
        D_g=(union/(a+b)) if (a+b)>0 else np.nan
        D_cos=(1.0-inter/math.sqrt(a*b)) if (a>0 and b>0) else np.nan
        if inter>0:
            latest_pair=max(max(ptimes[u][q],ptimes[v][q]) for q in inter_set)
            days_since=(t_ns-latest_pair)/(86400*1e9)
        else:
            days_since=np.nan
        pair_rows.append((pc,u,v,int(u in actual and v in actual),a,b,inter,union,F,D_j,D_g,D_cos,F*D_j,days_since))
    if idx%250==0:
        print('project',idx,'pairs',len(pair_rows),flush=True)

ind=pd.DataFrame(ind_rows,columns=['project_code','user_code','is_observed_member','prior_projects','prior_collaborators','log_prior_projects','log_prior_collaborators'])
pairs=pd.DataFrame(pair_rows,columns=['project_code','user_i','user_j','is_observed_pair','nproj_i','nproj_j','shared_projects','union_projects','familiarity_log1p','diversity_jaccard','diversity_gryzzly','diversity_cosine','familiarity_x_diversity','days_since_last_shared'])
# add split/team/date metadata
meta_cols=final[['project_code','team_id','project_created_dt','split','observed_team_size','candidate_pool_size','outcome_log_B_over_C','overrun']]
ind=ind.merge(meta_cols,on='project_code',how='left')
pairs=pairs.merge(meta_cols,on='project_code',how='left')
ind.to_pickle(f'{OUT}/individual_features_raw.pkl')
pairs.to_pickle(f'{OUT}/pair_features_raw.pkl')
# also compressed csv for portability
ind.to_csv(f'{OUT}/individual_features_raw.csv.gz',index=False,compression='gzip')
pairs.to_csv(f'{OUT}/pair_features_raw.csv.gz',index=False,compression='gzip')
print('individual rows',len(ind),'pair rows',len(pairs),flush=True)
print('observed pair rows',int(pairs.is_observed_pair.sum()),flush=True)

# Outcome-blind audit summaries
summary={}
summary['n_projects']=int(final.project_code.nunique())
summary['n_candidate_rows']=int(len(ind))
summary['n_pair_rows']=int(len(pairs))
summary['n_observed_pair_rows']=int(pairs.is_observed_pair.sum())
summary['share_never_collaborated_all_pairs']=float((pairs.shared_projects==0).mean())
summary['share_never_collaborated_observed_pairs']=float((pairs.loc[pairs.is_observed_pair==1,'shared_projects']==0).mean())
for col in ['shared_projects','familiarity_log1p','diversity_jaccard','diversity_gryzzly','diversity_cosine','familiarity_x_diversity','days_since_last_shared']:
    s=pairs[col]
    qs=s.quantile([0,.01,.05,.25,.5,.75,.95,.99,1]).to_dict()
    summary[col+'_quantiles']={str(k):None if pd.isna(v) else float(v) for k,v in qs.items()}
# correlations (all / observed), outcome-blind among feature columns
cols=['familiarity_log1p','diversity_jaccard','diversity_gryzzly','diversity_cosine','familiarity_x_diversity','nproj_i','nproj_j']
summary['corr_all']=pairs[cols].corr().round(6).to_dict()
summary['corr_observed']=pairs.loc[pairs.is_observed_pair==1,cols].corr().round(6).to_dict()
# split distributions
split_stats={}
for sp,g in pairs.groupby('split'):
    split_stats[sp]={
      'pairs':int(len(g)),
      'share_zero_F':float((g.shared_projects==0).mean()),
      'median_F':float(g.familiarity_log1p.median()),
      'median_Dj':float(g.diversity_jaccard.median()),
      'median_Dg':float(g.diversity_gryzzly.median()),
    }
summary['split_pair_stats']=split_stats
# organization heterogeneity summaries
org=pairs.groupby('team_id').agg(n_pairs=('shared_projects','size'),zeroF=('shared_projects',lambda x:float((x==0).mean())),medianF=('familiarity_log1p','median'),medianDj=('diversity_jaccard','median')).reset_index()
summary['organizations_with_pairs']=int(len(org))
summary['organization_zeroF_quantiles']={str(k):float(v) for k,v in org.zeroF.quantile([.05,.25,.5,.75,.95]).to_dict().items()}
with open(f'{OUT}/pair_audit.json','w') as f: json.dump(summary,f,indent=2)
print(json.dumps({k:v for k,v in summary.items() if k not in ['corr_all','corr_observed']},indent=2)[:12000],flush=True)
print('\nCorrelation all:\n',pairs[cols].corr().round(3).to_string())
print('\nCorrelation observed:\n',pairs.loc[pairs.is_observed_pair==1,cols].corr().round(3).to_string())
