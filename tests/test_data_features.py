from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from dipp_label.data import load_cohort
from dipp_label.features import ProteinFeatures
from dipp_label.synthetic import generate_example
from dipp_label.splits import holdout, read_holdout
from reference_math import three_class_masks


def test_synthetic_selection_and_regeneration(cohort, tmp_path):
    assert len(cohort)==216 and cohort.synthetic
    assert cohort.npx.shape==(216,64)
    assert len(set(cohort.ids))==216
    assert not any('EXCLUDED' in s for s in cohort.ids)
    assert np.all(cohort.project[:8]=='314')
    regenerated=tmp_path/'data'
    generate_example(regenerated)
    original=Path(__file__).resolve().parents[1]/'dataset/example'
    for name in ('participants.csv','assays.csv','plasma.csv','SYNTHETIC_DATA.json'):
        assert (original/name).read_bytes()==(regenerated/name).read_bytes()


def test_duplicate_measurements_rejected(tmp_path):
    generate_example(tmp_path/'data')
    path=tmp_path/'data/plasma.csv';frame=pd.read_csv(path)
    pd.concat([frame,frame.iloc[:1]]).to_csv(path,index=False)
    with pytest.raises(ValueError,match='technical replicates'):load_cohort(tmp_path/'data')


def test_train_masks_match_frozen_reference(cohort,config):
    tr,te=holdout(cohort,71750);train=cohort.take(tr)
    encoded=ProteinFeatures(config).fit(train)
    data=dict(x=train.npx,lod=train.lod,y=train.y)
    bc=dict(assay_missing_max=1-config.assay_coverage,minimum_training_class_observations_per_assay=config.min_per_class,
            lod_coverage_min=config.lod_coverage,lod_detection_min=config.detection_min)
    _,strict,binary=three_class_masks(data,np.arange(len(train)),bc,dict(binary_minimum=config.binary_minimum))
    assert np.array_equal(encoded.blocks[1],np.flatnonzero(binary))
    assert set(encoded.blocks[0])<=set(np.flatnonzero(strict))
    assert set(encoded.blocks[2])<=set(np.flatnonzero(binary&~strict))
    assert not set(encoded.blocks[0])&set(encoded.blocks[2])


def test_query_values_and_labels_cannot_refit_features(cohort,config):
    tr,te=holdout(cohort,71750)
    encoder=ProteinFeatures(config).fit(cohort.take(tr))
    saved=[(f.copy(),s.mean_.copy() if s else None,s.scale_.copy() if s else None) for f,s in zip(encoder.fill_values,encoder.scalers)]
    query=cohort.take(te);query.npx[:]=1e6;query.lod[:]=-1e6;query.y[:]=2
    encoder.transform(query)
    for (fill,mean,scale),f,s in zip(saved,encoder.fill_values,encoder.scalers):
        assert np.array_equal(fill,f)
        if s:assert np.array_equal(mean,s.mean_) and np.array_equal(scale,s.scale_)


def test_unknown_detection_uses_training_frequency(cohort,config):
    tr,te=holdout(cohort,71750);encoder=ProteinFeatures(config).fit(cohort.take(tr))
    query=cohort.take(te[:1]);query.npx[:]=np.nan
    transformed=encoder.transform(query)
    start=encoder.widths[0];end=start+encoder.widths[1]
    # Frequency imputation is the training mean, hence standardized value zero.
    assert np.allclose(transformed[:,start:end],0,atol=1e-6)


def test_holdouts_and_supplied_allocations(cohort,tmp_path):
    tr,te=holdout(cohort,71750)
    assert not set(cohort.ids[tr])&set(cohort.ids[te])
    assert len(tr)+len(te)==len(cohort)
    assert np.array_equal(te,holdout(cohort,71750)[1])
    frame=pd.DataFrame(dict(PATNO=cohort.ids,seed=71750,split=np.where(np.isin(np.arange(len(cohort)),te),'test','train')))
    path=tmp_path/'split.csv';frame.to_csv(path,index=False)
    actual=read_holdout(cohort,path,71750)
    assert np.array_equal(actual[0],tr) and np.array_equal(actual[1],te)
    pd.concat([frame,frame.iloc[:1]]).to_csv(path,index=False)
    with pytest.raises(ValueError,match='exactly once'):read_holdout(cohort,path,71750)
