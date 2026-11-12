from pathlib import Path
import pytest
import torch
from dipp_label.config import Config
from dipp_label.data import load_cohort


@pytest.fixture(scope="session", autouse=True)
def threads():
    torch.set_num_threads(1)


@pytest.fixture(scope="session")
def cohort():
    return load_cohort(Path(__file__).resolve().parents[1]/'dataset/example')


@pytest.fixture
def config():
    return Config(seeds=[71750], threads=1, outer_folds=3, inner_folds=3, prototype_steps=3,
                  fusion_steps=4, baseline_grid='demo', baseline_neural_members=1,
                  baseline_neural_steps=[3], shap_cycles=2, shap_background_per_class=3, shap_max_per_class=2)
