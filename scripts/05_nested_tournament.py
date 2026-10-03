# PORTABLE PATH CONFIGURATION — added in Stage H2.7
from pathlib import Path as _Path
import os as _os
_REPRO_ROOT = _Path(_os.environ.get('GRYZZLY_REPRO_ROOT', _Path(__file__).resolve().parents[1])).resolve()
_DATA_DIR = _Path(_os.environ.get('GRYZZLY_DATA_DIR', _REPRO_ROOT / 'data')).resolve()
_STAGEF_DIR = _Path(_os.environ.get('GRYZZLY_STAGEF_DIR', _REPRO_ROOT / 'work' / 'stageF')).resolve()
_STAGEG_DIR = _Path(_os.environ.get('GRYZZLY_STAGEG_DIR', _REPRO_ROOT / 'work' / 'stageG')).resolve()
_STAGEF_DIR.mkdir(parents=True, exist_ok=True)
_STAGEG_DIR.mkdir(parents=True, exist_ok=True)

import os,json,math
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT=str(_STAGEF_DIR)
df=pd.read_pickle(f'{OUT}/project_model_features.pkl').sort_values('project_created_dt').reset_index(drop=True)
df['year']=df['project_created_dt'].dt.year

y='outcome_log_B_over_C'
context=['log_planned_h','log_candidate_pool','log1p_initial_leaf_tasks']
card=['observed_team_size','pair_count']
ind=['sum_log_prior_projects','sum_log_prior_collaborators']
F=['sum_familiarity']; D=['sum_diversity_jaccard']; FD=['sum_familiarity_x_diversity']
models={
 'B0_context': context,
 'B1_cardinality': context+card,
 'N1_individual': context+card+ind,
 'N2_familiarity': context+card+ind+F,
 'N3_diversity': context+card+ind+D,
 'N4_F_plus_D': context+card+ind+F+D,
 'N5_F_D_interaction': context+card+ind+F+D+FD,
}
alpha_grid=[1e-4,1e-3,1e-2,1e-1,1,10,100,1000]
outer_years=[2022,2023,2024]
all_preds=[]; fold_results=[]; coef_rows=[]
for test_year in outer_years:
    train=df[df.year<test_year].copy().sort_values('project_created_dt')
    test=df[df.year==test_year].copy().sort_values('project_created_dt')
    if len(train)<100 or len(test)==0: continue
    # inner chronological CV with 4 folds; fixed order preserved
    inner=TimeSeriesSplit(n_splits=4)
    for name,features in models.items():
        pipe=Pipeline([('scale',StandardScaler()),('ridge',Ridge())])
        gs=GridSearchCV(pipe,{'ridge__alpha':alpha_grid},scoring='neg_mean_absolute_error',cv=inner,n_jobs=-1)
        gs.fit(train[features],train[y])
        pred=gs.predict(test[features])
        mae=mean_absolute_error(test[y],pred); rmse=mean_squared_error(test[y],pred)**0.5; r2=r2_score(test[y],pred)
        fold_results.append({'test_year':test_year,'model':name,'n_train':len(train),'n_test':len(test),'alpha':float(gs.best_params_['ridge__alpha']),'mae':mae,'rmse':rmse,'r2':r2})
        tmp=test[['project_code','team_id','project_created_dt','year',y]].copy(); tmp['model']=name; tmp['prediction']=pred
        # organizations never seen in outer training
        seen=set(train.team_id.unique()); tmp['unseen_org']=~tmp.team_id.isin(seen)
        all_preds.append(tmp)
        # standardized coefficients from best model
        best=gs.best_estimator_;
        for f,c in zip(features,best.named_steps['ridge'].coef_):
            coef_rows.append({'test_year':test_year,'model':name,'feature':f,'coef_standardized':float(c),'alpha':float(gs.best_params_['ridge__alpha'])})
        print(test_year,name,'alpha',gs.best_params_['ridge__alpha'],'MAE',round(mae,4),'RMSE',round(rmse,4),'R2',round(r2,4),flush=True)

res=pd.DataFrame(fold_results); preds=pd.concat(all_preds,ignore_index=True); coefs=pd.DataFrame(coef_rows)
res.to_csv(f'{OUT}/nested_temporal_results.csv',index=False)
preds.to_pickle(f'{OUT}/nested_temporal_predictions.pkl'); preds.to_csv(f'{OUT}/nested_temporal_predictions.csv',index=False)
coefs.to_csv(f'{OUT}/nested_temporal_coefficients.csv',index=False)

# Aggregate OOF metrics per model over all outer test years
agg=[]
for name,g in preds.groupby('model'):
    yy=g[y].to_numpy(); pp=g.prediction.to_numpy()
    ug=g[g.unseen_org]
    agg.append({'model':name,'n_oof':len(g),'mae':mean_absolute_error(yy,pp),'rmse':mean_squared_error(yy,pp)**0.5,'r2':r2_score(yy,pp),
                'n_unseen_org_projects':len(ug),
                'mae_unseen_org':mean_absolute_error(ug[y],ug.prediction) if len(ug) else np.nan,
                'rmse_unseen_org':mean_squared_error(ug[y],ug.prediction)**0.5 if len(ug) else np.nan})
agg=pd.DataFrame(agg).sort_values('mae')
agg.to_csv(f'{OUT}/nested_temporal_aggregate.csv',index=False)
print('\nAGGREGATE OOF\n',agg.to_string(index=False),flush=True)

# paired cluster bootstrap by organization for MAE differences vs B1, using same project OOF universe
wide=preds.pivot_table(index=['project_code','team_id',y],columns='model',values='prediction').reset_index()
rng=np.random.default_rng(20260802)
orgs=wide.team_id.unique()
boot_rows=[]
B='B1_cardinality'
for comp in ['N1_individual','N2_familiarity','N3_diversity','N4_F_plus_D','N5_F_D_interaction']:
    diffs=[]
    for b in range(1000):
        sampled=rng.choice(orgs,size=len(orgs),replace=True)
        # preserve cluster multiplicity by concatenating selected org blocks
        blocks=[wide[wide.team_id==o] for o in sampled]
        s=pd.concat(blocks,ignore_index=True)
        eB=np.abs(s[y]-s[B]).mean(); eC=np.abs(s[y]-s[comp]).mean()
        diffs.append(eC-eB) # negative = improvement
    diffs=np.asarray(diffs)
    point=(np.abs(wide[y]-wide[comp]).mean()-np.abs(wide[y]-wide[B]).mean())
    boot_rows.append({'comparison':comp+' - '+B,'point_delta_mae':point,'pct_change_vs_B1':point/np.abs(wide[y]-wide[B]).mean()*100,
                      'ci2.5':np.quantile(diffs,.025),'ci50':np.quantile(diffs,.5),'ci97.5':np.quantile(diffs,.975),
                      'p_boot_improvement':float(np.mean(diffs<0))})
boot=pd.DataFrame(boot_rows)
boot.to_csv(f'{OUT}/nested_temporal_cluster_bootstrap.csv',index=False)
print('\nCLUSTER BOOTSTRAP DELTA MAE (negative is better)\n',boot.to_string(index=False),flush=True)

# coefficient stability summary for key relational models
ks=coefs[coefs.model.isin(['N2_familiarity','N3_diversity','N4_F_plus_D','N5_F_D_interaction'])]
print('\nKEY STANDARDIZED COEFFICIENTS BY OUTER YEAR\n')
print(ks[ks.feature.isin(['sum_familiarity','sum_diversity_jaccard','sum_familiarity_x_diversity','sum_log_prior_projects','sum_log_prior_collaborators','observed_team_size','pair_count'])].pivot_table(index=['model','feature'],columns='test_year',values='coef_standardized').round(4).to_string())

meta={'outer_years':outer_years,'models':models,'alpha_grid':alpha_grid,'primary_metric':'MAE','bootstrap_clusters':'team_id','bootstrap_reps':1000}
with open(f'{OUT}/tournament_meta.json','w') as f: json.dump(meta,f,indent=2)
