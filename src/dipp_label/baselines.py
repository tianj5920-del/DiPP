"""Manuscript classifier controls, selected using training-only OOF OvO AUC."""
import itertools
import pickle
import subprocess
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .features import ProteinFeatures
from .numerics import fit_lr, device_for
from .splits import folds
from .metrics import complete_metrics, normalize

FAMILIES = ("lr", "linear_svm", "rbf_svm", "random_forest", "extra_trees", "lightgbm", "xgboost", "mlp", "residual_mlp")


def parameter_grid(family, config):
    if family == "lr":
        grid = [dict(C=c) for c in [1e-4, 1e-3, 1e-2, 1e-1, 1.]]
    elif family.endswith("svm"):
        grid = [dict(C=c) for c in [.001, .01, .1, 1., 10.]]
    elif family in ("random_forest", "extra_trees"):
        grid = [dict(depth=d, leaf=l) for d in (4, None) for l in (2, 5)]
    elif family in ("lightgbm", "xgboost"):
        grid = [dict(depth=d, l2=l) for d in (2, 3) for l in (1., 10.)]
    elif family in ("mlp", "residual_mlp"):
        grid = [dict(weight_decay=w, dropout=d, steps=s) for w, d, s in
                itertools.product([.001, .01], [.1, .3], config.baseline_neural_steps)]
    else:
        raise ValueError(f"Unknown classifier {family}")
    return grid if config.baseline_grid == "paper" else grid[:1]


class ResidualMLP(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.input = nn.Linear(width, 64)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 64),
                                    nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 64)) for _ in range(2)])
        self.output = nn.Linear(64, 3)

    def forward(self, x):
        x = self.input(x)
        for block in self.blocks:
            x = x + block(x)
        return self.output(x)


class Estimator:
    def __init__(self, family, parameters, config):
        self.family, self.parameters, self.config = family, parameters, config

    def fit(self, x, y):
        f, p, c = self.family, self.parameters, self.config
        self.device = device_for(c.device)
        if f == "lr":
            self.lr = fit_lr(torch.as_tensor(x, device=self.device), y, p["C"])
        elif f in ("mlp", "residual_mlp"):
            xx = torch.as_tensor(x, device=self.device)
            yy = torch.as_tensor(y, device=self.device, dtype=torch.long)
            weight = torch.as_tensor(len(y)/(3*np.bincount(y, minlength=3)), dtype=xx.dtype, device=self.device)
            self.members = []
            for i in range(c.baseline_neural_members):
                seed = 101 + 1009 * i
                torch.manual_seed(seed)
                if f == "mlp":
                    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Dropout(p["dropout"]),
                                          nn.Linear(64, 32), nn.ReLU(), nn.Dropout(p["dropout"]), nn.Linear(32, 3))
                else:
                    model = ResidualMLP(x.shape[1], p["dropout"])
                model = model.to(self.device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=p["weight_decay"], foreach=False, fused=False)
                for _ in range(p["steps"]):
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(xx), yy, weight=weight)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., foreach=False)
                    optimizer.step()
                model.eval()
                self.members.append(model)
        else:
            self.model_bytes = self._worker(dict(action="fit", x=x, y=y, family=f, parameters=p, threads=c.threads))
        return self

    def predict_proba(self, x):
        if self.family == "lr":
            with torch.no_grad():
                xx = torch.as_tensor(x, device=self.device)
                return normalize((xx @ self.lr[0].T + self.lr[1]).softmax(1).cpu().numpy())
        if self.family in ("mlp", "residual_mlp"):
            with torch.no_grad():
                xx = torch.as_tensor(x, device=self.device)
                return normalize(np.mean([m(xx).softmax(1).cpu().numpy() for m in self.members], axis=0))
        result = self._worker(dict(action="predict", model=self.model_bytes, x=x))
        return normalize(pickle.loads(result))

    @staticmethod
    def _worker(request):
        result = subprocess.run([sys.executable, "-m", "dipp_label.classical_worker"],
                                input=pickle.dumps(request, protocol=5), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=600)
        if result.returncode:
            raise RuntimeError(f"Isolated classical worker failed (exit {result.returncode})")
        return result.stdout


class TunedClassifier:
    def __init__(self, config, family):
        self.config, self.family = config, family

    def fit(self, cohort):
        c = self.config
        cached = []
        for tr, va in folds(cohort, c.baseline_folds, 101):
            encoder = ProteinFeatures(c).fit(cohort.take(tr))
            width = sum(encoder.widths[:2])
            cached.append((tr, va, encoder.transform(cohort.take(tr))[:, :width], encoder.transform(cohort.take(va))[:, :width]))
        candidates = []
        for index, parameters in enumerate(parameter_grid(self.family, c)):
            oof = np.zeros((len(cohort), 3))
            for tr, va, x, q in cached:
                model = Estimator(self.family, parameters, c).fit(x, cohort.y[tr])
                oof[va] = model.predict_proba(q)
            score = complete_metrics(cohort.y, oof)["macro_auc_ovo"]
            candidates.append(dict(parameters=parameters, auc=score, index=index))
        choice = max(candidates, key=lambda r: (r["auc"], -r["parameters"].get("steps", 0), -r["index"]))
        self.selection = choice
        self.features = ProteinFeatures(c).fit(cohort)
        self.width = sum(self.features.widths[:2])
        self.estimator = Estimator(self.family, choice["parameters"], c).fit(self.features.transform(cohort)[:, :self.width], cohort.y)
        return self

    def predict_proba(self, cohort):
        return self.estimator.predict_proba(self.features.transform(cohort)[:, :self.width])
