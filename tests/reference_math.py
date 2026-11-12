"""Frozen data-free numerical reference functions extracted for migration tests."""

from types import SimpleNamespace

import numpy as np

import torch

from torch.nn import functional as F

deadline=lambda *a: None

base=SimpleNamespace(deadline=deadline)

def fit_lr(x,y,C):
    """Same accepted multinomial objective; expose fitted weights in RAM only."""
    deadline();target=torch.as_tensor(y,device=x.device,dtype=torch.long)
    counts=F.one_hot(target,3).sum(0)
    if bool((counts==0).any().cpu()):raise ValueError('lr_class_gate')
    sample_weights=(len(y)/3)/counts
    weight=torch.zeros((3,x.shape[1]),device=x.device,dtype=x.dtype,requires_grad=True)
    bias=torch.zeros(3,device=x.device,dtype=x.dtype,requires_grad=True)
    optimizer=torch.optim.LBFGS([weight,bias],lr=1.,max_iter=400,max_eval=700,
        tolerance_grad=1e-8,tolerance_change=1e-12,history_size=50,line_search_fn='strong_wolfe')
    calls=[0]
    def closure():
        if calls[0]%50==0:deadline()
        calls[0]+=1;optimizer.zero_grad()
        logits=x@weight.T+bias;maximum=logits.max(1).values.detach()
        log_mean_exp=maximum+torch.log1p(torch.expm1(logits-maximum[:,None]).mean(1))
        chosen=logits.gather(1,target[:,None]).squeeze(1);weights=sample_weights[target]
        loss=((log_mean_exp-chosen)*weights).sum()/weights.sum()+weight.square().sum()/(2*C*len(y))
        loss.backward();return loss
    optimizer.step(closure)
    with torch.no_grad():
        if not bool((torch.isfinite(weight).all() & torch.isfinite(bias).all()).cpu()):raise ValueError('nonfinite_lr')
    receipt=dict(optimizer_steps=int(optimizer.state[weight].get('n_iter',0)),training_parameter_count=weight.numel()+bias.numel(),closure_evaluations=calls[0])
    return (weight.detach().clone(),bias.detach().clone()),receipt

def gaussian(query,means,variances,model):
    pooled=variances.mean(1,keepdim=True)
    value=(1-model.metric_shrinkage)*variances+model.metric_shrinkage*pooled
    floor=pooled.mean(-1,keepdim=True)*model.metric_variance_floor+1e-6
    value=torch.maximum(value,floor)
    difference=query[:,:,None,:]-means[:,None,:,:]
    return -.5*(difference.square()/value[:,None,:,:]+value[:,None,:,:].log()).mean(-1)

def episode_rank(logits,query):
    """Six directed class pairs, compared within the same support episode."""
    logp=logits.log_softmax(-1).reshape(-1,3*query,3);terms=[]
    for c in range(3):
        positive=logp[:,c*query:(c+1)*query,c]
        for d in range(3):
            if c==d:continue
            negative=logp[:,d*query:(d+1)*query,c]
            terms.append(F.softplus(negative[:,None,:]-positive[:,:,None]).mean())
    return torch.stack(terms).mean()

def episodes(y, support, query, count, rng, primary_count=None, auxiliary_weight=1.):
    size=support+query
    picks=np.empty((count,3,size),dtype=np.int64)
    for label in range(3):
        candidates=np.flatnonzero(y==label)
        if len(candidates)<size:
            raise ValueError('sampling_class_gate')
        weight=np.ones(len(candidates)) if primary_count is None else np.where(candidates<primary_count,1.,auxiliary_weight)
        if not np.all(weight>0):
            raise ValueError('sampling_weights')
        # Ordering exponential clocks yields the same sequential weighted
        # without-replacement law as repeated categorical draws.
        uniform=rng.random((count,len(candidates)))
        clocks=-np.log(np.maximum(uniform,np.finfo(np.float64).tiny))/weight[None]
        selected=np.argpartition(clocks,size-1,axis=1)[:,:size]
        order=np.argsort(np.take_along_axis(clocks,selected,axis=1),axis=1)
        selected=np.take_along_axis(selected,order,axis=1)
        picks[:,label]=candidates[selected]
    return np.concatenate((picks[:,:,:support].reshape(count,-1),picks[:,:,support:].reshape(count,-1)),axis=1)

def three_class_masks(d,tr,bc,fc):
    x,lod=d['x'][tr],d['lod'][tr]
    valid=np.isfinite(x);paired=valid&np.isfinite(lod)
    broad=valid.mean(0)>=1-bc['assay_missing_max']
    for label in (0,1,2):
        broad &= valid[d['y'][tr]==label].sum(0)>=bc['minimum_training_class_observations_per_assay']
    detected=(paired&(x>=lod)).sum(0);below=(paired&(x<lod)).sum(0)
    strict=broad&(paired.mean(0)>=bc['lod_coverage_min'])&(detected/np.maximum(paired.sum(0),1)>=bc['lod_detection_min'])
    binary=broad&(paired.mean(0)>=bc['lod_coverage_min'])&(detected>=fc['binary_minimum'])&(below>=fc['binary_minimum'])
    return broad,strict,binary

def conjugate_gradient(matrix,target,iterations=192):
    value=torch.zeros_like(target);diagonal=matrix.diagonal().clamp_min(1e-8)[:,None]
    for _ in range(2):
        residual=target-matrix@value;preconditioned=residual/diagonal;direction=preconditioned.clone();product=(residual*preconditioned).sum(0)
        for step in range(iterations//2):
            if step%50==0:base.deadline(300)
            projected=matrix@direction;denominator=(direction*projected).sum(0).clamp_min(1e-20)
            alpha=product/denominator;value=value+direction*alpha
            residual=residual-projected*alpha;preconditioned=residual/diagonal
            updated=(residual*preconditioned).sum(0)
            direction=preconditioned+direction*(updated/product.clamp_min(1e-20));product=updated
    relative=(target-matrix@value).square().sum().sqrt()/target.square().sum().sqrt().clamp_min(1e-8)
    return value,relative
