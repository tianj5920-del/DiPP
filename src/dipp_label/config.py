from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class Config:
    device: str = "cpu"
    threads: int = 2
    seeds: list[int] = field(default_factory=lambda: [71750, 71761, 71832])
    test_fraction: float = 0.2
    test_counts: dict = field(default_factory=dict)
    outer_folds: int = 5
    inner_folds: int = 5
    outer_seed: int = 58111
    inner_seed: int = 49101
    model_seed: int = 101
    neural_seed_offset: int = 7699
    prototype_steps: int = 800
    episodes_per_step: int = 4
    support: int = 5
    query: int = 5
    learning_rate: float = 0.0003
    weight_decay: float = 0.001
    teacher_c: float = 0.003
    detection_scale: float = 0.5
    ranking_weight: float = 0.3
    correction_strength: float = 1.5
    shrinkage: float = 0.5
    variance_floor: float = 0.01
    fusion_steps: int = 300
    fusion_lr: float = 0.03
    fusion_penalty: float = 0.1
    assay_coverage: float = 0.8
    lod_coverage: float = 0.8
    detection_min: float = 0.5
    min_per_class: int = 5
    binary_minimum: int = 10
    baseline_folds: int = 3
    baseline_grid: str = "paper"
    baseline_neural_steps: list[int] = field(default_factory=lambda: [100, 300, 800])
    baseline_neural_members: int = 5
    shap_background_per_class: int = 12
    shap_cycles: int = 16
    shap_max_per_class: int | None = None
    shap_seed: int = 812120

    def __post_init__(self):
        if self.device != "cpu":
            raise ValueError("This release uses CPU training only")
        for k in ("threads", "outer_folds", "inner_folds", "prototype_steps", "episodes_per_step", "support", "query", "fusion_steps", "shap_cycles"):
            if getattr(self, k) < 1:
                raise ValueError(f"{k} must be positive")
        if min(self.outer_folds, self.inner_folds, self.baseline_folds) < 2:
            raise ValueError("Cross-validation needs at least two folds")
        if not 0 < self.test_fraction < 1 or not 0 < self.teacher_c:
            raise ValueError("Invalid split fraction or LR regularization")
        if self.baseline_grid not in ("paper", "demo"):
            raise ValueError("baseline_grid must be paper or demo")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Partition seeds must be unique")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)
