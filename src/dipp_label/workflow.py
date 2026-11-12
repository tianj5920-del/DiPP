"""Nested validation, fixed full-training fits, aggregate evaluation and SHAP."""
from pathlib import Path
import importlib.metadata
import importlib.util
import json
import platform
import time
import numpy as np
import pandas as pd
import torch
from . import CLASSES
from .data import load_cohort
from .model import DiPPLabel, STRATEGIES
from .baselines import TunedClassifier, FAMILIES
from .splits import folds, holdout, read_holdout
from .metrics import complete_metrics, random_reference


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')


def make_model(name, config):
    if name in STRATEGIES:
        return DiPPLabel(config,name)
    if name in FAMILIES:
        return TunedClassifier(config,name)
    raise ValueError(f"Unknown model {name}")


def _validate_optional_dependencies(names, explain):
    required = {'shap'} if explain else set()
    required |= set(names) & {'lightgbm','xgboost'}
    for name in required:
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(f"Install optional dependency {name} before starting this run")


def run(data_folder, output_folder, config, names, validate=True, explain=False, split_csv=None):
    if len(set(names))!=len(names) or set(names)-set(STRATEGIES+FAMILIES+('random',)):
        raise ValueError('Choose distinct supported model names')
    if explain and 'dipp_label' not in names:
        raise ValueError('SHAP analysis requires the final dipp_label model')
    _validate_optional_dependencies(names,explain)
    torch.set_num_threads(config.threads)
    output=Path(output_folder)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; use a fresh run directory')
    cohort=load_cohort(data_folder)
    output.mkdir(parents=True,exist_ok=True)
    versions={}
    for package in ('numpy','pandas','scipy','scikit-learn','torch','shap','lightgbm','xgboost'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    write_json(output/'run_manifest.json',dict(config=config.to_dict(),models=names,synthetic=cohort.synthetic,
        participant_count=len(cohort),assay_count=len(cohort.assay_ids),class_order=list(CLASSES),
        counts={n:int((cohort.y==i).sum()) for i,n in enumerate(CLASSES)},python=platform.python_version(),versions=versions,
        split_source='provided participant allocations' if split_csv else 'new class-stratified random partitions',
        validation='nested outer-fold predictions' if validate else 'not requested',
        model_selection='fixed strategy; baseline hyperparameters and fusion weights use training data only',
        outputs='aggregate metrics and protein attribution summaries; no participant predictions'))
    records, shap_runs, shap_diagnostics=[],[],[]
    start=time.perf_counter()
    for seed in config.seeds:
        tr,te=read_holdout(cohort,split_csv,seed) if split_csv else holdout(cohort,seed,config.test_fraction,config.test_counts)
        train,test=cohort.take(tr),cohort.take(te)
        if set(train.ids)&set(test.ids):raise ValueError('Train/test participant overlap')
        for name in names:
            tick=time.perf_counter()
            print(json.dumps(dict(stage='fitting',model=name,seed=seed,training=len(train),testing=len(test))),flush=True)
            if name=='random':
                phases=[('test',test.y)]+([('cv',train.y)] if validate else [])
                for phase,y in phases:records.append(dict(seed=seed,model=name,phase=phase,**random_reference(y)))
                continue
            if validate:
                probability=np.zeros((len(train),3));seen=np.zeros(len(train),int)
                for index,(fit,query) in enumerate(folds(train,config.outer_folds,config.outer_seed)):
                    model=make_model(name,config).fit(train.take(fit))
                    probability[query]=model.predict_proba(train.take(query));seen[query]+=1
                    print(json.dumps(dict(stage='outer_fold_complete',model=name,seed=seed,fold=index+1)),flush=True)
                    del model
                if not np.all(seen==1):raise ValueError('Outer validation coverage')
                records.append(dict(seed=seed,model=name,phase='cv',**complete_metrics(train.y,probability)))
            model=make_model(name,config).fit(train)
            # Only now are the held-out outcomes used for scoring.
            probability=model.predict_proba(test)
            records.append(dict(seed=seed,model=name,phase='test',**complete_metrics(test.y,probability)))
            if explain and name=='dipp_label':
                from .analysis import grouped_shap
                rows,diagnostics=grouped_shap(model,train,test,config)
                shap_runs.append(rows);shap_diagnostics.extend([dict(seed=seed,**d) for d in diagnostics])
            print(json.dumps(dict(stage='model_complete',model=name,seed=seed,seconds=round(time.perf_counter()-tick,3))),flush=True)
            del model
            pd.DataFrame(records).to_csv(output/'metrics_by_split.csv',index=False)
    frame=pd.DataFrame(records)
    metric_columns=[c for c in frame if c not in ('seed','model','phase')]
    summary=[]
    for (model,phase),group in frame.groupby(['model','phase'],sort=False):
        summary.append(dict(model=model,phase=phase,repetitions=len(group),
            mean={c:float(group[c].mean()) for c in metric_columns},
            sd={c:float(group[c].std(ddof=1)) if len(group)>1 else 0. for c in metric_columns}))
    write_json(output/'summary.json',summary)
    frame.to_csv(output/'metrics_by_split.csv',index=False)
    if shap_runs:
        from .analysis import summarize_rankings
        ranking,stability=summarize_rankings(shap_runs)
        ranking.to_csv(output/'shap_all_rankings.csv',index=False,float_format='%.8e')
        ranking[ranking['rank']<=50].to_csv(output/'shap_top50.csv',index=False,float_format='%.8e')
        write_json(output/'shap_checks.json',dict(diagnostics=shap_diagnostics,stability=stability,
            score='log(p_A)-log(p_B)',ranking='equal recipient-class mean absolute SHAP, then equal model mean',
            direction='within-class high-minus-low attribution using training NPX quartiles'))
    report=['# DiPP-Label run report','',f"Synthetic data: **{cohort.synthetic}**. Repetitions: {len(config.seeds)}.",'',
            '| Model | Phase | Macro P | Macro R | Macro-F1 | OvR AUC | OvO AUC |',
            '| --- | --- | --- | --- | --- | --- | --- |']
    for row in summary:
        m=row['mean'];report.append(f"| {row['model']} | {row['phase']} | {m['macro_precision']:.2%} | {m['macro_recall']:.2%} | {m['macro_f1']:.4f} | {m['macro_auc_ovr']:.4f} | {m['macro_auc_ovo']:.4f} |")
    report+=['','## Classwise means','', '| Model | Phase | Class | Precision | Recall | F1 | OvR AUC |', '| --- | --- | --- | --- | --- | --- | --- |']
    for row in summary:
        m=row['mean']
        for label in ('HC','PD','Prodromal'):
            report.append(f"| {row['model']} | {row['phase']} | {label} | {m[label+'_precision']:.2%} | {m[label+'_recall']:.2%} | {m[label+'_f1']:.4f} | {m[label+'_auc_ovr']:.4f} |")
    report+=['','All metrics and split-level results are in `summary.json` and `metrics_by_split.csv`.']
    (output/'report.md').write_text('\n'.join(report)+'\n')
    write_json(output/'completion.json',dict(status='completed',elapsed_seconds=time.perf_counter()-start,metric_rows=len(records),synthetic=cohort.synthetic))
    return summary
