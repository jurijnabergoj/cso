"""
Experiment definitions
Change GROUP and EXPERIMENTS
at the bottom to activate a new round.
"""

# Round 1: architecture ablations
# Baseline: flat shape-latent encoder, trying attention vs flatten, hybrid vs not.
R1 = {
    "group": "exp1_arch",
    "experiments": [
        {
            "name": "cross_attn_hybrid_logloss",
            "ablation.use_attention": True,
            "model.use_hybrid": True,
            "loss.log_scale": True,
        },
        {
            "name": "cross_attn_hybrid_nolog",
            "ablation.use_attention": True,
            "model.use_hybrid": True,
            "loss.log_scale": False,
        },
        {
            "name": "cross_attn_nohybrid_logloss",
            "ablation.use_attention": True,
            "model.use_hybrid": False,
            "loss.log_scale": True,
        },
        {
            "name": "flatten_hybrid_logloss",
            "ablation.use_attention": False,
            "model.use_hybrid": True,
            "loss.log_scale": True,
        },
        {
            "name": "flatten_nohybrid_logloss",
            "ablation.use_attention": False,
            "model.use_hybrid": False,
            "loss.log_scale": True,
        },
        {
            "name": "flatten_nohybrid_nolog",
            "ablation.use_attention": False,
            "model.use_hybrid": False,
            "loss.log_scale": False,
        },
    ],
}

# Round 2: regularization + early stopping
# Best R1 experiment: flatten_nohybrid_nolog. Address severe train/val gap.
R2 = {
    "group": "exp2_reg",
    "experiments": [
        {
            "name": "flatten_nohybrid_logloss_v2",
            "ablation.use_attention": False,
            "model.use_hybrid": False,
            "loss.log_scale": True,
            "train.patience": 5,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "flatten_highreg",
            "ablation.use_attention": False,
            "model.use_hybrid": False,
            "loss.log_scale": False,
            "train.weight_decay": 0.1,
            "model.dropout": 0.4,
            "train.patience": 5,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "flatten_small",
            "ablation.use_attention": False,
            "model.use_hybrid": False,
            "loss.log_scale": False,
            "model.d_model": 128,
            "train.weight_decay": 0.05,
            "model.dropout": 0.3,
            "train.patience": 5,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "flatten_hybrid_highreg",
            "ablation.use_attention": False,
            "model.use_hybrid": True,
            "loss.log_scale": False,
            "train.weight_decay": 0.1,
            "model.dropout": 0.4,
            "train.patience": 5,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "cross_attn_lowlr",
            "ablation.use_attention": True,
            "model.use_hybrid": False,
            "loss.log_scale": False,
            "train.lr": 0.0001,
            "train.weight_decay": 0.05,
            "train.patience": 5,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "cross_attn_hybrid_lowlr",
            "ablation.use_attention": True,
            "model.use_hybrid": True,
            "loss.log_scale": False,
            "train.lr": 0.0001,
            "train.weight_decay": 0.05,
            "train.patience": 5,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "cross_attn_logloss_lowlr",
            "ablation.use_attention": True,
            "model.use_hybrid": False,
            "loss.log_scale": True,
            "train.lr": 0.0001,
            "train.weight_decay": 0.1,
            "train.patience": 5,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
    ],
}

# Round 3: feature ablation
# Goal: understand which features actually help against overfitting.
# Result: shape latents (4096 tokens) add no value over slat+geom alone.
R3 = {
    "group": "exp3_features",
    "experiments": [
        {
            "name": "geom_only",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": False,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.01,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "slat_geom",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": True,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.01,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "flatten_hybrid_v2",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": True,
            "ablation.use_slat": True,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.05,
            "model.dropout": 0.3,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "slat_geom_highreg",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": True,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.05,
            "model.dropout": 0.3,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
    ],
}

# Round 4: stats encoder
# Replace 65536-dim flatten with mean/std/max aggregation (48-dim).
# Result: worse than flatten (226 vs 174) — spatial structure matters.
R4 = {
    "group": "exp4_stats",
    "experiments": [
        {
            "name": "stats_hybrid",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": True,
            "ablation.use_slat": True,
            "ablation.use_stats_encoder": True,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.01,
            "model.dropout": 0.1,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "stats_noslat",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": True,
            "ablation.use_slat": False,
            "ablation.use_stats_encoder": True,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.01,
            "model.dropout": 0.1,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "stats_hybrid_highreg",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": True,
            "ablation.use_slat": True,
            "ablation.use_stats_encoder": True,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.05,
            "model.dropout": 0.2,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "stats_hybrid_smalld",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": True,
            "ablation.use_slat": True,
            "ablation.use_stats_encoder": True,
            "model.use_hybrid": True,
            "model.d_model": 64,
            "loss.log_scale": True,
            "train.weight_decay": 0.01,
            "model.dropout": 0.1,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
    ],
}

# Round 5: 3D CNN encoder
# Shape latent is a 16^3 spatial volume
# Result: 194 MAE.
R5 = {
    "group": "exp4_conv3d",
    "experiments": [
        {
            "name": "conv3d_hybrid",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": True,
            "ablation.use_slat": True,
            "ablation.use_conv3d_encoder": True,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.01,
            "model.dropout": 0.1,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "conv3d_hybrid_highreg",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": True,
            "ablation.use_slat": True,
            "ablation.use_conv3d_encoder": True,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.05,
            "model.dropout": 0.2,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "conv3d_noslat",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": True,
            "ablation.use_slat": False,
            "ablation.use_conv3d_encoder": True,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.01,
            "model.dropout": 0.1,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
    ],
}

# Round 6: geom fix + ResNet50 image features
# 1. geom_estimate fixed: use real-world scale (~0.64).
# 2. Use ResNet50 scene-level image features (2048-dim)
# Run: sbatch slurm/generate_image_embeddings.slurm
# Result: 167 MAE (image_geom_highreg).
R6 = {
    "group": "exp5_image",
    "experiments": [
        {
            "name": "conv3d_correctgeom",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": True,
            "ablation.use_slat": True,
            "ablation.use_conv3d_encoder": True,
            "ablation.use_image_encoder": False,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.01,
            "model.dropout": 0.1,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "image_geom_only",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": False,
            "ablation.use_image_encoder": True,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.01,
            "model.dropout": 0.1,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "image_conv3d_full",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": True,
            "ablation.use_slat": True,
            "ablation.use_conv3d_encoder": True,
            "ablation.use_image_encoder": True,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.01,
            "model.dropout": 0.1,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "image_geom_highreg",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": False,
            "ablation.use_image_encoder": True,
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.05,
            "model.dropout": 0.2,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
    ],
}

# Round 7: DINOv2 scene-level features
# DINOv2-small (ViT-S/14, 384-dim CLS token) vs ResNet50 (2048-dim avgpool).
# Run: sbatch --export=ALL,BACKBONE=dinov2_s slurm/generate_image_embeddings.slurm
# Result: 137 MAE (dinov2_slat_geom).
R7 = {
    "group": "exp6_dinov2",
    "experiments": [
        {
            "name": "dinov2_geom_highreg",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": False,
            "ablation.use_image_encoder": True,
            "model.image_feat_dim": 384,
            "model.image_feat_file": "dinov2_s_feats.pt",
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.05,
            "model.dropout": 0.2,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "dinov2_geom_only",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": False,
            "ablation.use_image_encoder": True,
            "model.image_feat_dim": 384,
            "model.image_feat_file": "dinov2_s_feats.pt",
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.01,
            "model.dropout": 0.1,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "dinov2_slat_geom",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": True,
            "ablation.use_image_encoder": True,
            "model.image_feat_dim": 384,
            "model.image_feat_file": "dinov2_s_feats.pt",
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.05,
            "model.dropout": 0.2,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
    ],
}

# Round 8: DINOv2 masked crops
# Container and object encoded separately from their masked regions.
# Run: sbatch --export=ALL,BACKBONE=dinov2_s,MASKED=1 slurm/generate_image_embeddings.slurm
# Result: 128 MAE (masked_scene_dinov2_geom).
# Scene-level (whole-image CLS token) beats masked crops alone,
# but combining both is best (128 MAE).
R8 = {
    "group": "exp7_masked_crops",
    "experiments": [
        {
            "name": "masked_dinov2_geom",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": False,
            "ablation.use_image_encoder": False,
            "ablation.use_masked_image_encoder": True,
            "model.image_feat_dim": 384,
            "model.container_image_feat_file": "dinov2_s_container_feats.pt",
            "model.object_image_feat_file": "dinov2_s_object_feats.pt",
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.01,
            "model.dropout": 0.1,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "masked_dinov2_slat_geom",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": True,
            "ablation.use_image_encoder": False,
            "ablation.use_masked_image_encoder": True,
            "model.image_feat_dim": 384,
            "model.container_image_feat_file": "dinov2_s_container_feats.pt",
            "model.object_image_feat_file": "dinov2_s_object_feats.pt",
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.05,
            "model.dropout": 0.2,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "masked_scene_dinov2_geom",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": False,
            "ablation.use_image_encoder": True,
            "ablation.use_masked_image_encoder": True,
            "model.image_feat_dim": 384,
            "model.image_feat_file": "dinov2_s_feats.pt",
            "model.container_image_feat_file": "dinov2_s_container_feats.pt",
            "model.object_image_feat_file": "dinov2_s_object_feats.pt",
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.05,
            "model.dropout": 0.2,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
    ],
}

# Round 9: DINOv2 patch pooling
# Patch pooling works. Blackened backgrounds seem to put DINOv2 out of distribution.
# Encoding the full image and then pooling only the masked patch tokens is significantly better.
# This round also includes the geom_features fix (6 metric scale features added) so some gain could come from that.
# Result: 130.6 MAE (masked_dinov2_slat_geom)
R9 = {
    "group": "exp8_masked_patch_pooling",
    "experiments": [
        {
            "name": "masked_dinov2_geom",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": False,
            "ablation.use_image_encoder": False,
            "ablation.use_masked_image_encoder": True,
            "model.image_feat_dim": 384,
            "model.container_image_feat_file": "dinov2_s_container_feats.pt",
            "model.object_image_feat_file": "dinov2_s_object_feats.pt",
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.01,
            "model.dropout": 0.1,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "masked_dinov2_slat_geom",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": True,
            "ablation.use_image_encoder": False,
            "ablation.use_masked_image_encoder": True,
            "model.image_feat_dim": 384,
            "model.container_image_feat_file": "dinov2_s_container_feats.pt",
            "model.object_image_feat_file": "dinov2_s_object_feats.pt",
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.05,
            "model.dropout": 0.2,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
        {
            "name": "masked_scene_dinov2_geom",
            "ablation.use_attention": False,
            "ablation.use_shape_latent": False,
            "ablation.use_slat": False,
            "ablation.use_image_encoder": True,
            "ablation.use_masked_image_encoder": True,
            "model.image_feat_dim": 384,
            "model.image_feat_file": "dinov2_s_feats.pt",
            "model.container_image_feat_file": "dinov2_s_container_feats.pt",
            "model.object_image_feat_file": "dinov2_s_object_feats.pt",
            "model.use_hybrid": True,
            "loss.log_scale": True,
            "train.weight_decay": 0.05,
            "model.dropout": 0.2,
            "train.patience": 10,
            "train.epochs": 200,
            "train.save_checkpoints": False,
        },
    ],
}

# Active round
# Change when starting a new round.
ACTIVE = R8
GROUP = ACTIVE["group"]
EXPERIMENTS = ACTIVE["experiments"]
