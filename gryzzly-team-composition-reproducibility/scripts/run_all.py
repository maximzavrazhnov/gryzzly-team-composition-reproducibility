#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, sys, os, time
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'scripts'
CORE=[f'{i:02d}_'+name for i,name in [
(1,'build_base.py'),(2,'build_cohort.py'),(3,'build_pairwise.py'),(4,'build_project_features.py'),
(5,'nested_tournament.py'),(6,'coverage_diagnostic.py'),(7,'composition_diagnostic.py'),(8,'normalized_game_diagnostic.py'),
(9,'intensive_eb_game.py'),(10,'eb_decomposition.py'),(11,'eb_binary_robustness.py'),(12,'eb_prior_robustness.py'),
(13,'eb_maturity_robustness.py'),(14,'intensive_game_main.py'),(15,'intensive_game_optimization.py')]]
STAGEG=['16_stageG_robustness_formation.py','17_formation_validation_corrected.py','18_training_label_embargo_sensitivity.py']
AUDIT=['h2_6_strict_sensitivity.py','h2_6_composition_ablation.py','h2_6_unseen_bootstrap.py']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['core','stageg','audit','all'], default='all')
    args=ap.parse_args()
    seq={'core':CORE,'stageg':STAGEG,'audit':AUDIT,'all':CORE+STAGEG+AUDIT}[args.stage]
    env=os.environ.copy(); env.setdefault('GRYZZLY_REPRO_ROOT',str(ROOT))
    for n in seq:
        p=S/n
        print(f'\n=== RUN {n} ===', flush=True)
        t=time.time(); subprocess.run([sys.executable,str(p)],check=True,env=env)
        print(f'=== DONE {n}: {time.time()-t:.1f}s ===', flush=True)
if __name__=='__main__': main()
