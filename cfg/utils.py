import yaml
from pathlib import Path
from cfg.configs import *


def load_config(path: Path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig(
        exp_name=raw["exp_name"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        loss=LossConfig(**raw["loss"]),
        ablation=AblationConfig(**raw["ablation"]),
    )