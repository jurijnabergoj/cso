from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    train_dirs: List[Path]
    val_dir: Path
    batch_size: int
    num_workers: int


@dataclass
class ModelConfig:
    shape_latent_dim: int
    slat_dim: int
    use_hybrid: bool
    d_model: int
    d_ff: int
    h: int
    dropout: float
    geometric_feature_dim: int = 19
    image_feat_dim: int = 2048  # dim of precomputed image features (ResNet50 avgpool)
    image_feat_file: str = "image_feats.pt"  # filename inside each scene dir
    container_image_feat_file: str = ""  # masked container crop feats; empty = disabled
    object_image_feat_file: str = ""  # masked object crop feats; empty = disabled


@dataclass
class TrainConfig:
    epochs: int
    lr: int
    weight_decay: float
    load_best_model: bool
    best_model_dir: Path
    output_dir: Path
    patience: int = 0  # early stopping for val checks; 0 = disabled
    save_checkpoints: bool = True

    def __post_init__(self):
        self.best_model_dir = Path(self.best_model_dir)
        self.output_dir = Path(self.output_dir)


@dataclass
class LossConfig:
    use_l1: bool
    use_mse: bool
    log_scale: bool


@dataclass
class AblationConfig:
    use_geom: bool
    use_shape_latent: bool
    use_slat: bool
    use_attention: bool
    use_probabilistic: bool
    use_stats_encoder: bool = False  # mean/std/max over tokens instead of flatten
    use_conv3d_encoder: bool = False  # 3D CNN over the 16^3 spatial volume
    use_image_encoder: bool = False  # precomputed scene-level image features
    use_masked_image_encoder: bool = (
        False  # separate masked crops for container + object
    )


@dataclass
class ExperimentConfig:
    exp_name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    loss: LossConfig
    ablation: AblationConfig
