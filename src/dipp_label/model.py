"""Final single-member model and the four manuscript component ablations."""
import numpy as np
import torch
from torch.nn import functional as F
from .features import ProteinFeatures
from .numerics import (TorchPCA, device_for, fit_lr, gaussian, episode_density, moments,
                       episodes, episode_rank, conjugate_gradient, distances, hierarchy)
from .splits import folds

STRATEGIES = ("dipp_label", "without_hc", "lr_hc", "lr_free_hc", "lr_free")


class BaseModel:
    def __init__(self, config, strategy="dipp_label", seed=None):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.config, self.strategy = config, strategy
        self.seed = config.model_seed if seed is None else seed
        self.device = device_for(config.device)
        self.guided = strategy not in ("lr_free_hc", "lr_free")
        self.has_prototype = strategy != "lr_hc"
        self.has_hc = strategy in ("dipp_label", "lr_hc", "lr_free_hc")

    def _tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    def fit(self, cohort):
        c = self.config
        self.features = ProteinFeatures(c)
        encoded = self.features.fit_transform(cohort)
        self.y = cohort.y.copy()
        raw = self._tensor(encoded)
        self.widths = self.features.widths
        self.kernel_width = sum(self.widths[:2])
        self.factor = torch.cat((torch.ones(self.widths[0], device=self.device),
                                 torch.full((self.widths[1],), c.detection_scale, device=self.device)))
        if self.guided:
            self.lr = fit_lr(raw[:, :self.kernel_width] * self.factor, self.y, c.teacher_c)
        self.pcas = []
        if self.has_prototype:
            with torch.no_grad():
                ranks = [6, 4, 4] if self.guided else [8, 4, 4]
                start, parts = 0, []
                for width, rank in zip(self.widths, ranks):
                    pca = TorchPCA()
                    parts.append(pca.fit_transform(raw[:, start:start + width], rank))
                    self.pcas.append(pca)
                    start += width
                projection = torch.cat(parts, 1) / (3 ** .5)
                if self.guided:
                    teacher = self._lr_logits(raw)
                    contrasts = torch.stack((teacher[:, 0] - teacher[:, 2], teacher[:, 1] - teacher[:, 2]), 1)
                    self.contrast_center = contrasts.mean(0)
                    self.contrast_scale = (contrasts - self.contrast_center).square().sum(1).mean().sqrt().clamp_min(1e-6)
                    reference = torch.cat(((contrasts - self.contrast_center) / self.contrast_scale, projection), 1)
                else:
                    reference = projection * (2 ** .5)
                    teacher = torch.zeros((len(cohort), 3), device=self.device)
            self._train_prototype(reference, teacher)
        if self.has_hc:
            with torch.no_grad():
                self.kernel_training = raw[:, :self.kernel_width].clone()
                dist = distances(self.kernel_training, self.kernel_training)
                self.bandwidth = (dist.sum() / (len(cohort) * (len(cohort) - 1))).clamp_min(1e-6)
                target = self._tensor((self.y == 0).astype(float))
                counts = torch.stack(((1 - target).sum(), target.sum()))
                sw = len(cohort) / (2 * counts[target.long()])
                matrix = torch.exp(-dist / self.bandwidth) + torch.diag(1 / sw)
                coefficients, error = conjugate_gradient(matrix, (2 * target - 1)[:, None])
                self.kernel_residual = float(error.cpu())
                if self.kernel_residual > 1e-4 or not np.isfinite(self.kernel_residual):
                    raise ValueError("HC kernel solver residual exceeds 1e-4")
                self.alpha = coefficients[:, 0]
        return self

    def _lr_logits(self, raw):
        return (raw[:, :self.kernel_width] * self.factor) @ self.lr[0].T + self.lr[1]

    def _reference(self, raw):
        start, parts = 0, []
        for width, pca in zip(self.widths, self.pcas):
            parts.append(pca.transform(raw[:, start:start + width]))
            start += width
        projected = torch.cat(parts, 1) / (3 ** .5)
        if not self.guided:
            return projected * (2 ** .5)
        teacher = self._lr_logits(raw)
        contrasts = torch.stack((teacher[:, 0] - teacher[:, 2], teacher[:, 1] - teacher[:, 2]), 1)
        return torch.cat(((contrasts - self.contrast_center) / self.contrast_scale, projected), 1)

    def _train_prototype(self, reference, teacher):
        c = self.config
        seed = self.seed + c.neural_seed_offset
        torch.manual_seed(seed)
        self.update = torch.nn.Linear(16, 16, bias=False).to(self.device)
        torch.nn.init.zeros_(self.update.weight)
        optimizer = torch.optim.AdamW(self.update.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay,
                                      foreach=False, fused=False)
        ix = episodes(self.y, c.support, c.query, c.prototype_steps * c.episodes_per_step,
                      np.random.default_rng(seed + 400001))
        ix = torch.as_tensor(ix.reshape(c.prototype_steps, c.episodes_per_step, -1), device=self.device)
        target = torch.arange(3, device=self.device).repeat_interleave(c.query).repeat(c.episodes_per_step)
        finite = torch.ones((), device=self.device, dtype=torch.bool)
        for batch in ix:
            optimizer.zero_grad(set_to_none=True)
            ref = reference[batch.flatten()].reshape(c.episodes_per_step, -1, 16)
            z = ref + self.update(ref)
            split = 3 * c.support
            logits = episode_density(z[:, split:], z[:, :split].reshape(-1, 3, c.support, 16), c.shrinkage, c.variance_floor)
            if self.guided:
                initial = episode_density(ref[:, split:], ref[:, :split].reshape(-1, 3, c.support, 16), c.shrinkage, c.variance_floor)
                logits = teacher[batch[:, split:].flatten()] + (logits - initial).reshape(-1, 3)
            else:
                logits = logits.reshape(-1, 3)
            loss = F.cross_entropy(logits, target) + c.ranking_weight * episode_rank(logits, c.query)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.update.parameters(), 1., foreach=False)
            finite &= torch.isfinite(loss.detach()) & torch.isfinite(norm)
            optimizer.step()
        if not bool(finite.cpu()):
            raise ValueError("Nonfinite prototype training")
        with torch.no_grad():
            self.reference_moments = moments(reference, self.y)
            self.learned_moments = moments(reference + self.update(reference), self.y)
        self.update.eval()

    def predict_encoded(self, encoded):
        with torch.no_grad():
            raw = self._tensor(encoded)
            if not self.has_prototype:
                score = self._lr_logits(raw)
            else:
                z = self._reference(raw)
                c = self.config
                learned = gaussian((z + self.update(z))[None], *self.learned_moments, c.shrinkage, c.variance_floor)[0]
                if self.guided:
                    initial = gaussian(z[None], *self.reference_moments, c.shrinkage, c.variance_floor)[0]
                    score = self._lr_logits(raw) + c.correction_strength * (learned - initial)
                else:
                    score = learned
            p = score.softmax(1)
            if self.has_hc:
                dist = distances(raw[:, :self.kernel_width], self.kernel_training)
                hc = (2 * (torch.exp(-dist / self.bandwidth) @ self.alpha)).sigmoid()
                return torch.stack((p, hierarchy(hc, p)), 1)
            return p[:, None, :]

    def predict_branches(self, cohort):
        return self.predict_encoded(self.features.transform(cohort))


class DiPPLabel:
    def __init__(self, config, strategy="dipp_label", seed=None):
        self.config, self.strategy = config, strategy
        self.seed = config.model_seed if seed is None else seed

    def fit(self, cohort):
        c = self.config
        paired = self.strategy in ("dipp_label", "lr_hc", "lr_free_hc")
        device = device_for(c.device)
        if paired:
            oof = np.zeros((len(cohort), 2, 3), np.float32)
            seen = np.zeros(len(cohort), int)
            for k, (tr, va) in enumerate(folds(cohort, c.inner_folds, c.inner_seed)):
                member = BaseModel(c, self.strategy, self.seed + 200003 * (k + 1)).fit(cohort.take(tr))
                oof[va] = member.predict_branches(cohort.take(va)).cpu().numpy()
                seen[va] += 1
                del member
            if not np.all(seen == 1):
                raise ValueError("Fusion OOF predictions must cover every training participant once")
            x = torch.as_tensor(oof, device=device).clamp_min(1e-8)
            target = torch.as_tensor(cohort.y, device=device, dtype=torch.long)
            sample_weight = len(cohort) / (3 * torch.bincount(target, minlength=3)[target])
            theta = torch.nn.Parameter(torch.zeros((2, 1), device=device))
            optimizer = torch.optim.Adam([theta], lr=c.fusion_lr, foreach=False, fused=False)
            for _ in range(c.fusion_steps):
                p = (x * theta.softmax(0)[None]).sum(1)
                p = p / p.sum(1, keepdim=True)
                loss = -(p.gather(1, target[:, None]).flatten().clamp_min(1e-10).log() * sample_weight).mean() + c.fusion_penalty * theta.square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([theta], 2., foreach=False)
                optimizer.step()
            self.weights = theta.detach().softmax(0).flatten()
        else:
            self.weights = torch.ones(1, device=device)
        self.base = BaseModel(c, self.strategy, self.seed).fit(cohort)
        return self

    def predict_encoded(self, encoded, chunk_size=512):
        values = []
        for start in range(0, len(encoded), chunk_size):
            branches = self.base.predict_encoded(encoded[start:start + chunk_size]).clamp_min(1e-8)
            p = (branches * self.weights[None, :, None]).sum(1)
            p = p / p.sum(1, keepdim=True)
            values.append(p.cpu().numpy().astype(float))
        return np.concatenate(values)

    def predict_proba(self, cohort):
        return self.predict_encoded(self.base.features.transform(cohort))
