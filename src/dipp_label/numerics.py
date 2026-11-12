"""Numerical primitives ported from the frozen experiment; no data readers."""
import numpy as np
import torch
from torch.nn import functional as F


def device_for(name="cpu"):
    if name != "cpu":
        raise ValueError("This release supports CPU training only")
    return torch.device("cpu")


def fit_lr(x, y, c):
    target = torch.as_tensor(y, device=x.device, dtype=torch.long)
    counts = F.one_hot(target, 3).sum(0)
    if bool((counts == 0).any().cpu()):
        raise ValueError("LR training requires all classes")
    sample_weights = (len(y) / 3) / counts
    weight = torch.zeros((3, x.shape[1]), device=x.device, dtype=x.dtype, requires_grad=True)
    bias = torch.zeros(3, device=x.device, dtype=x.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=1., max_iter=400, max_eval=700,
                                  tolerance_grad=1e-8, tolerance_change=1e-12, history_size=50,
                                  line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        logits = x @ weight.T + bias
        maximum = logits.max(1).values.detach()
        log_mean_exp = maximum + torch.log1p(torch.expm1(logits - maximum[:, None]).mean(1))
        chosen = logits.gather(1, target[:, None]).squeeze(1)
        weights = sample_weights[target]
        loss = ((log_mean_exp - chosen) * weights).sum() / weights.sum() + weight.square().sum() / (2 * c * len(y))
        loss.backward()
        return loss
    optimizer.step(closure)
    if not bool((torch.isfinite(weight).all() & torch.isfinite(bias).all()).cpu()):
        raise ValueError("Nonfinite LR parameters")
    return weight.detach(), bias.detach()


class TorchPCA:
    """Randomized PCA with the experiment's NumPy sketch, QR and sign convention.

    All linear algebra runs on CPU.
    """
    def fit_transform(self, x, rank, seed=101):
        if min(len(x) - 1, x.shape[1]) < rank:
            raise ValueError(f"PCA block needs at least {rank} variable features and {rank + 1} training rows")
        with torch.no_grad():
            self.mean = x.mean(0)
            centered = x - self.mean
            transpose = centered.shape[0] < centered.shape[1]
            matrix = centered.T if transpose else centered
            sketch_rank = min(rank + 10, min(matrix.shape))
            iterations = 7 if rank < .1 * min(matrix.shape) else 4
            omega = np.random.RandomState(seed).normal(size=(matrix.shape[1], sketch_rank)).astype(np.float32)
            basis = torch.as_tensor(omega, device=x.device)
            for _ in range(iterations):
                basis = torch.linalg.qr(matrix @ basis, mode="reduced").Q
                basis = torch.linalg.qr(matrix.T @ basis, mode="reduced").Q
            basis = torch.linalg.qr(matrix @ basis, mode="reduced").Q
            u_small, singular, vh = torch.linalg.svd(basis.T @ matrix, full_matrices=False)
            u = basis @ u_small
            left, components = (vh.T, u.T) if transpose else (u, vh)
            left, singular, components = left[:, :rank], singular[:rank], components[:rank]
            largest = components.abs().argmax(1)
            signs = torch.where(components[torch.arange(rank, device=x.device), largest] < 0, -1., 1.)
            self.components = components * signs[:, None]
            scores = left * (singular * signs)
            self.pca_scale = scores.square().mean().sqrt().clamp_min(1e-6)
            scores = scores / self.pca_scale
            self.block_center = scores.mean(0)
            self.block_scale = (scores - self.block_center).square().sum(1).mean().sqrt().clamp_min(1e-6)
            return (scores - self.block_center) / self.block_scale

    def transform(self, x):
        scores = (x - self.mean) @ self.components.T / self.pca_scale
        return (scores - self.block_center) / self.block_scale


def gaussian(query, means, variances, shrinkage=.5, variance_floor=.01):
    # query: B x Q x D; means/variances: B x C x D.
    pooled = variances.mean(1, keepdim=True)
    value = (1 - shrinkage) * variances + shrinkage * pooled
    value = torch.maximum(value, pooled.mean(-1, keepdim=True) * variance_floor + 1e-6)
    return -.5 * ((query[:, :, None, :] - means[:, None, :, :]).square() / value[:, None] + value[:, None].log()).mean(-1)


def episode_density(query, support, shrinkage=.5, variance_floor=.01):
    means = support.mean(2)
    variance = (support - means[:, :, None, :]).square().mean(2)
    return gaussian(query, means, variance, shrinkage, variance_floor)


def moments(x, y):
    groups = [x[torch.as_tensor(np.flatnonzero(y == c), device=x.device)] for c in range(3)]
    means = torch.stack([g.mean(0) for g in groups])
    variances = torch.stack([(g - means[c]).square().mean(0) for c, g in enumerate(groups)])
    return means[None], variances[None]


def episodes(y, support, query, count, rng):
    size = support + query
    picks = np.empty((count, 3, size), dtype=np.int64)
    for label in range(3):
        candidates = np.flatnonzero(y == label)
        if len(candidates) < size:
            raise ValueError(f"Class {label} needs {size} training participants; support/query are not silently reduced")
        uniform = rng.random((count, len(candidates)))
        clocks = -np.log(np.maximum(uniform, np.finfo(np.float64).tiny))
        selected = np.argpartition(clocks, size - 1, axis=1)[:, :size]
        order = np.argsort(np.take_along_axis(clocks, selected, axis=1), axis=1)
        picks[:, label] = candidates[np.take_along_axis(selected, order, axis=1)]
    return np.concatenate((picks[:, :, :support].reshape(count, -1), picks[:, :, support:].reshape(count, -1)), axis=1)


def episode_rank(logits, query):
    logp = logits.log_softmax(-1).reshape(-1, 3 * query, 3)
    terms = []
    for c in range(3):
        positive = logp[:, c * query:(c + 1) * query, c]
        for d in range(3):
            if c != d:
                negative = logp[:, d * query:(d + 1) * query, c]
                terms.append(F.softplus(negative[:, None, :] - positive[:, :, None]).mean())
    return torch.stack(terms).mean()


def conjugate_gradient(matrix, target, iterations=192):
    value = torch.zeros_like(target)
    diagonal = matrix.diagonal().clamp_min(1e-8)[:, None]
    for _ in range(2):
        residual = target - matrix @ value
        preconditioned = residual / diagonal
        direction = preconditioned.clone()
        product = (residual * preconditioned).sum(0)
        for _ in range(iterations // 2):
            projected = matrix @ direction
            alpha = product / (direction * projected).sum(0).clamp_min(1e-20)
            value = value + direction * alpha
            residual = residual - projected * alpha
            preconditioned = residual / diagonal
            updated = (residual * preconditioned).sum(0)
            direction = preconditioned + direction * (updated / product.clamp_min(1e-20))
            product = updated
    relative = (target - matrix @ value).square().sum().sqrt() / target.square().sum().sqrt().clamp_min(1e-8)
    return value, relative


def distances(a, b):
    return (a.square().sum(1)[:, None] + b.square().sum(1)[None] - 2 * a @ b.T).clamp_min(0)


def hierarchy(hc, p):
    h = hc.clamp(1e-8, 1 - 1e-8)
    conditional = p[:, 1:] / p[:, 1:].sum(1, keepdim=True).clamp_min(1e-10)
    return torch.cat((h[:, None], (1 - h[:, None]) * conditional), 1)
