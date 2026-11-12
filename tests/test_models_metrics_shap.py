import itertools
import numpy as np
import pytest
import torch
from dipp_label.model import BaseModel,DiPPLabel,STRATEGIES
from dipp_label.numerics import hierarchy
from dipp_label.splits import holdout
from dipp_label.metrics import complete_metrics,random_reference
from dipp_label.analysis import AssayMasker,grouped_shap,summarize_rankings


def test_hc_ratio_identity():
    p=torch.tensor([[.2,.3,.5],[.1,.8,.1]])
    h=torch.tensor([.7,.05]);weights=torch.tensor([.4,.6])
    fused=weights[0]*p+weights[1]*hierarchy(h,p)
    assert torch.allclose(fused[:,2]/fused[:,1],p[:,2]/p[:,1])


@pytest.mark.parametrize('family',['lightgbm','xgboost'])
def test_isolated_tree_runtime(family,config):
    import importlib.util
    if importlib.util.find_spec(family) is None:
        pytest.skip(f'{family} optional dependency is not installed')
    from dipp_label.baselines import Estimator
    rng=np.random.default_rng(44)
    x=rng.normal(size=(60,10)).astype(np.float32);y=np.repeat([0,1,2],20)
    model=Estimator(family,dict(depth=2,l2=1.),config).fit(x,y)
    p=model.predict_proba(x[:6])
    assert p.shape==(6,3) and np.isfinite(p).all() and np.allclose(p.sum(1),1)


@pytest.mark.parametrize('strategy',STRATEGIES)
def test_model_predict_is_independent_of_query_labels(cohort,config,strategy):
    tr,te=holdout(cohort,71750)
    model=DiPPLabel(config,strategy).fit(cohort.take(tr))
    query=cohort.take(te)
    p=model.predict_proba(query)
    query.y[:]=-1
    assert np.array_equal(model.predict_proba(query),p)
    assert p.shape==(len(te),3) and np.isfinite(p).all() and np.allclose(p.sum(1),1)
    if strategy.startswith('lr_free'):assert not hasattr(model.base,'lr')


def test_lr_free_never_calls_teacher(cohort,config,monkeypatch):
    import dipp_label.model as module
    def forbidden(*args):raise AssertionError('LR used by LR-free branch')
    monkeypatch.setattr(module,'fit_lr',forbidden)
    tr,te=holdout(cohort,71750)
    BaseModel(config,'lr_free_hc').fit(cohort.take(tr)).predict_branches(cohort.take(te))


def test_metrics_pairwise_auc_matches_pair_count():
    y=np.array([0,0,1,1,2,2]);p=np.array([[.6,.2,.2],[.4,.4,.2],[.2,.7,.1],[.3,.5,.2],[.1,.2,.7],[.3,.3,.4]])
    metrics=complete_metrics(y,p)
    score=np.log(p[:,2]/p[:,1]);pd=score[y==2];prod=score[y==1]
    expected=np.mean([(a>b)+.5*(a==b) for a,b in itertools.product(pd,prod)])
    assert metrics['pair_PD_vs_Prodromal_auc_logratio']==expected
    assert metrics['macro_recall']==metrics['balanced_accuracy']
    assert metrics['macro_f1']==np.mean([metrics[n+'_f1'] for n in ('HC','PD','Prodromal')])
    assert random_reference(np.repeat([0,1,2],[10,32,25]))['macro_f1']==pytest.approx(.31468703765037004)


def test_two_group_shap_matches_exact_shapley():
    shap=pytest.importorskip('shap')
    background=np.array([[0.,1.,1.],[1.,0.,2.]])
    query=np.array([2.,3.,4.]);groups=[np.array([0,1]),np.array([2])]
    masker=AssayMasker(background,query,groups)
    def function(x):return x[:,0]+2*x[:,1]+x[:,0]*x[:,2]
    values={}
    for mask in ([False,False],[True,False],[False,True],[True,True]):
        values[tuple(mask)]=function(masker(np.array(mask),None)[0]).mean()
    expected=np.array([.5*(values[True,False]-values[False,False]+values[True,True]-values[False,True]),
                       .5*(values[False,True]-values[False,False]+values[True,True]-values[True,False])])
    actual=shap.PermutationExplainer(function,masker,seed=7)(np.ones((1,2)),max_evals=20,silent=True)
    assert np.allclose(actual.values[0],expected)


def test_full_model_shap_additivity_and_top50(cohort,config):
    pytest.importorskip('shap')
    tr,te=holdout(cohort,71750)
    model=DiPPLabel(config).fit(cohort.take(tr))
    rows,checks=grouped_shap(model,cohort.take(tr),cohort.take(te),config)
    assert len(checks)==3 and max(c['maximum_additivity_error'] for c in checks)<1e-4
    frame,stability=summarize_rankings([rows])
    assert frame.boundary.nunique()==3
    assert np.allclose(frame.groupby('boundary').normalized_percent.max(),100)
    assert not stability


def test_cross_split_ranking_uses_zero_for_absent_assays():
    row=dict(boundary='PD_vs_Prodromal',assay_id='A',protein='SyntheticA',mean_abs_shap=2.,direction_contrasts=[1.],channels='NPX')
    other=dict(row,assay_id='B',protein='SyntheticB',mean_abs_shap=1.)
    frame,stability=summarize_rankings([[row,other],[other]])
    value=frame.set_index('assay_id')
    assert value.loc['A','mean_abs_shap']==1.
    assert value.loc['A','present_models']==1
    assert stability[0]['spearman'] is None
