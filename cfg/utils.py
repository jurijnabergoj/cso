import yaml
from pathlib import Path
from cfg.configs import *
from cfg.experiments import GROUP


PROJECT_ROOT = Path("/d/hpc/home/jn16867/cso")
CFG_DIR = PROJECT_ROOT / "cfg"
BASE_CONFIG = CFG_DIR / "baseline.yaml"


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


def set_nested(d: dict, dotkey: str, value) -> None:
    keys = dotkey.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def generate_config(experiment: dict) -> Path:
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    name = experiment["name"]
    cfg["exp_name"] = name
    cfg["train"]["output_dir"] = str(PROJECT_ROOT / "outputs" / GROUP / name)

    for key, value in experiment.items():
        if key == "name":
            continue
        set_nested(cfg, key, value)

    out_cfg_path = CFG_DIR / f"{name}.yaml"
    with open(out_cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"  Written: {out_cfg_path}")
    return out_cfg_path
