from types import SimpleNamespace
import numpy as np
import torch
import reference_math as original
from dipp_label import numerics as new


def test_frozen_gaussian_values_and_gradients():
    rng=np.random.default_rng(17)
    q=torch.tensor(rng.normal(size=(2,7,16)),dtype=torch.float32,requires_grad=True)
    means=torch.tensor(rng.normal(size=(2,3,16)),dtype=torch.float32)
    variance=torch.tensor(rng.uniform(0,.2,size=(2,3,16)),dtype=torch.float32)
    variance[:,:,0]=0
    a=original.gaussian(q,means,variance,SimpleNamespace(metric_shrinkage=.5,metric_variance_floor=.01))
    b=new.gaussian(q,means,variance)
    assert torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a.sum(),q,retain_graph=True)[0],torch.autograd.grad(b.sum(),q)[0])


def test_frozen_ranking_and_sampling():
    logits=torch.randn(60,3,requires_grad=True)
    a=original.episode_rank(logits,5);b=new.episode_rank(logits,5)
    assert torch.equal(a,b)
    y=np.repeat([0,1,2],30)
    reference=original.episodes(y,5,5,12,np.random.default_rng(9))
    actual=new.episodes(y,5,5,12,np.random.default_rng(9))
    assert np.array_equal(reference,actual)
    assert all(len(set(row))==30 for row in actual)
    assert np.array_equal(y[actual[0,15:]],np.repeat([0,1,2],5))


def test_frozen_lr_and_kernel_solver():
    rng=np.random.default_rng(22);x=torch.tensor(rng.normal(size=(36,8)),dtype=torch.float32)
    y=np.repeat([0,1,2],12)
    expected,_=original.fit_lr(x,y,.003)
    actual=new.fit_lr(x,y,.003)
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    k=torch.exp(-new.distances(x,x)/8)+torch.eye(len(x))
    target=torch.tensor(rng.normal(size=(len(x),1)),dtype=torch.float32)
    a,e=original.conjugate_gradient(k,target);b,f=new.conjugate_gradient(k,target)
    assert torch.equal(a,b) and torch.equal(e,f)
    assert torch.allclose(b,torch.linalg.solve(k,target),atol=2e-6,rtol=2e-6)


def test_pca_energy_and_projection():
    rng=np.random.default_rng(7);x=torch.tensor(rng.normal(size=(45,9)),dtype=torch.float32)
    pca=new.TorchPCA();z=pca.fit_transform(x,6)
    assert torch.allclose(z,pca.transform(x),atol=2e-6,rtol=2e-6)
    assert abs(float(z.square().sum(1).mean())-1)<2e-6
    assert torch.allclose(pca.components@pca.components.T,torch.eye(6),atol=2e-6)
