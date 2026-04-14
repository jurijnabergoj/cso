# CSO — Container Scene Object Counting

Predict how many objects fit inside a container from a single RGB image of the scene.

**Target:** ~79 MAE (paper baseline) | **Current best:** 127.43 MAE (R12 `pf_noslat_l10`)

---

## Problem Description

Given a single photograph of a container filled with objects (e.g. a box of pasta, a bin of parts), predict the object count. The problem decomposes into two components:

```
count ≈ volume_ratio × packing_factor
```

- **Volume ratio** — how many object-volumes fit geometrically inside the container (depends on their relative sizes)
- **Packing factor** — what fraction of the container is actually occupied (depends on object shape, arrangement, physics)

Both are hard to estimate from a single image. The key findings from this project are:

1. Volume ratio from 3D reconstruction (SAM3D) is nearly useless (r=0.068 with count) due to monocular depth scale ambiguity
2. DINOv2 visual features implicitly learn both components from appearance and are the dominant signal
3. Explicit decomposition into vr×pf predictions competes against the oracle; the model is better off predicting count directly with auxiliary pf supervision
4. The 48 MAE gap to the paper target comes from volume estimation quality, not packing factor

---

## Dataset

| Split | Path | Scenes |
|---|---|---|
| Train part 1 | `/d/hpc/projects/FRI/jn16867/3d-counting/scenes_part1` | ~5,750 |
| Train part 2 | `/d/hpc/projects/FRI/jn16867/3d-counting/scenes_part2` | ~5,750 |
| Val | `/d/hpc/projects/FRI/jn16867/3d-counting/scenes_val` | 90 |

Val count range: mean=653, min=30, max=2343.

**Scene directory structure:**

```
scene_dir/
├── images/          RGB{id}.jpg   — multi-view RGB frames
├── box_seg/         Box_Mask{id}.png  — container wall masks
├── obj_seg/         Objects_Mask{id}.png  — filled pile mask (container interior)
├── floor_seg/       Ground_Mask{id}.png
├── sam_data/        (generated)
│   ├── frame_id.txt         — best frame index
│   ├── best_instance_mask.png
│   ├── dinov2_s_feats.pt          (384,)
│   ├── dinov2_s_container_feats.pt (384,)
│   ├── dinov2_s_instance_feats.pt  (k, 384)
│   ├── instance_px.pt              (k,)
│   ├── dinov2_b_*.pt               — DINOv2-B variants
│   └── ...
├── sam3d_data/      (generated)
│   └── embeddings.pt
├── simulation_results.json  — ground truth packing factor
├── gt_count.json            — true object count
└── transforms.json          — camera intrinsics / poses
```

---

## Architecture

The model (`cso/models/count_predictor.py`) concatenates features from multiple branches and passes them through an MLP count head:

```
DINOv2-S scene (384)   ──────────────────────────────────────────┐
DINOv2-S container (384) ──[Linear→256, LN, ReLU] ────────────── │
DINOv2-S instance  (384) ──[Linear→256, LN, ReLU] ────────────── │
SAM3D slat pool (16) ──[MLP→128] ──────────────────────────────── ├─ cat → MLP → count
SAM3D slat cross-attn (64) ─[CrossAttn 256 voxels]────────────── │
Geom features (6 or 16) ──────────────────────────────────────── ┘
                                                    └─ pf_head (aux) → packing_factor
```

**Key flags (ablation config):**

| Flag | Default | Effect |
|---|---|---|
| `use_image_encoder` | false | Scene-level DINOv2 CLS token |
| `use_masked_image_encoder` | false | Masked container + object DINOv2 crops |
| `use_slat` | true | SAM3D slat max+mean pool → 16-dim |
| `use_slat_cross_attn` | false | Cross-attention between 256 subsampled voxels |
| `use_shape_latent` | true | SAM3D global (4096,8) shape fingerprint (consistently hurts) |
| `use_packing_factor_head` | false | Auxiliary sigmoid head supervised by `simulation_results.json` |
| `use_image_in_pf_head` | false | Give pf_head DINOv2 container+obj features |
| `use_clean_geom` | false | 6-dim clean geom features instead of legacy 13-dim |
| `use_hybrid` | true | Append `log1p(geom_estimate)` to features |

---

## SAM3D Feature Reference

Each `sam3d_data/embeddings.pt` contains:

| Key | Shape | Signal | Notes |
|---|---|---|---|
| `container/object.coords` | (N, 4) int32 | r=0.42 via voxel count ratio | Active voxels in 64³ grid; col 0 is batch idx |
| `container/object.slat_features` | (N, 8) float | moderate | Per-voxel local geometry; max+mean pooled to 16-dim |
| `container/object.shape_latent` | (4096, 8) float | weak (hurts) | Global shape fingerprint; consistently degrades MAE |
| `container/object.scale` | (3,) float | r=0.07, useless | Uniform scale, monocular depth ambiguity |
| `container/object.translation` | (3,) float | weak | Camera-frame position |
| `container/object.rotation` | (4,) float quaternion | weak | Orientation |
| `geom_features` | (13,) float | mixed | Precomputed: see table below |
| `geom_estimate` | scalar | r=0.07, noise | 0.64×(cont_scale/obj_scale)³ |
| `true_count` | int | — | Ground truth |

**Legacy `geom_features` (13-dim):**

| Index | Formula | Notes |
|---|---|---|
| 0 | obj_bbox_vol / cont_bbox_vol | bbox volume ratio |
| 1 | cont_voxels / cont_bbox_vol | container fill density |
| 2 | obj_voxels / obj_bbox_vol | object fill density |
| 3 | obj_density / cont_density | relative density |
| 4 | cont_voxels / 64³ | container occupancy |
| 5 | obj_voxels / 64³ | object occupancy |
| 6 | obj_translation.norm() | camera distance proxy |
| 7 | obj_scale / cont_scale | 1D scale ratio |
| 8 | cont_bbox_max / cont_bbox_min | container aspect ratio |
| 9 | obj_bbox_max / obj_bbox_min | object aspect ratio |
| 10–12 | log/sqrt/scaled variants of 0 | redundant |

**Clean `geom_features` (6-dim, `use_clean_geom=True`):**

| Index | Formula | Notes |
|---|---|---|
| 0 | log(N_cont_vox + 1) | container voxel count |
| 1 | log(N_obj_vox + 1) | object voxel count |
| 2 | log(N_cont_vox / N_obj_vox + 1) | voxel count ratio — r=0.42 with true count |
| 3 | cont_bbox_max / cont_bbox_min | container aspect ratio |
| 4 | obj_bbox_max / obj_bbox_min | object aspect ratio |
| 5 | log(cont_scale / obj_scale) | 1D log scale ratio |

---

## Experiment History

All val MAE results (90 scenes, mean count ≈ 653).

### R1–R5: Shape latents only (174–283 MAE)

SAM3D shape_latent + slat + geom features only. Broken `geom_estimate` (always 1.0). Best: 174 MAE.

> **Finding:** Shape latents consistently hurt. The (4096,8) global fingerprint overfits to geometry that doesn't predict count well.

### R6 — ResNet50 scene image (167 MAE)

Fixed `geom_estimate` to use real-world scale: `0.64 × (cont_scale/obj_scale)³`. Added ResNet50 (2048-dim) scene-level features.

| Experiment | MAE |
|---|---|
| image_geom_highreg | **167.06** |
| image_geom_only | 169.89 |

### R7 — DINOv2-S scene-level (137 MAE)

Replaced ResNet50 with DINOv2-Small CLS token (384-dim). Large improvement.

| Experiment | MAE |
|---|---|
| dinov2_slat_geom | **136.90** |
| dinov2_geom_only | 139.56 |

> **Finding:** DINOv2 is significantly better than ResNet50 for this task. Adding slat pooling gives ~3 MAE benefit.

### R8 — Masked DINOv2 crops (128–156 MAE)

Added separate DINOv2 features for masked container and object regions (blackened background). Blackening the background degrades DINOv2 (out of distribution).

| Experiment | MAE |
|---|---|
| masked_scene_dinov2_geom | **127.91** |
| masked_dinov2_geom | 149.21 |

### R9 — Masked patch pooling (131 MAE)

Switched from blackened crops to patch-token pooling over masked regions within the full image. DINOv2 sees the full image context but only the masked patch tokens are pooled.

| Experiment | MAE |
|---|---|
| masked_dinov2_slat_geom | **130.62** |
| masked_dinov2_geom | 133.83 |

### R10 — SAM pipeline with consistent frames (130 MAE)

All features (scene, container, instance) now come from the same best frame selected by `prepare_sam_data.py`. No frame inconsistency.

| Experiment | MAE |
|---|---|
| sam_all_dinov2_slat_geom | **130.12** |
| sam_all_dinov2_geom | 137.39 |
| sam_masked_dinov2_geom | 192.43 |

> **Finding:** Using only masked crops (no scene context) is catastrophically worse (192 MAE). Scene-level DINOv2 is essential.

### R11 — Shape latents + new pipeline (146–150 MAE)

Re-tested shape_latent with the new consistent-frame pipeline. Still hurts.

> **Finding:** shape_latent consistently hurts regardless of pipeline. Abandoned.

### R12 — Auxiliary packing factor supervision (127 MAE) ⭐ Current best

Added auxiliary sigmoid `pf_head` on SAM3D slat features, supervised by `simulation_results.json` packing factor. `pf_noslat`: slat only feeds pf_head, not count head. `pf_slat`: slat feeds both.

| Experiment | λ | MAE |
|---|---|---|
| **pf_noslat_l10** | 1.0 | **127.43** |
| pf_slat_l10 | 1.0 | 129.76 |
| pf_noslat_l05 | 0.5 | 130.32 |
| pf_slat_l01 | 0.1 | 135.30 |
| pf_noslat_l01 | 0.1 | 142.84 |
| pf_slat_l05 | 0.5 | 143.74 |

> **Finding:** Separating slat features for pf_head only (not count head) is better — slat can specialise for packing without interfering with the count gradient. λ=1.0 is optimal.

### R13 — Lambda sweep (127–138 MAE)

Extended sweep to higher lambda values.

| Experiment | λ | MAE |
|---|---|---|
| pf_noslat_l50 | 5.0 | 129.31 |
| pf_noslat_l100 | 10.0 | 136.60 |
| pf_noslat_l20 | 2.0 | 138.45 |

> **Finding:** λ=1.0 is definitively optimal. Non-monotone at high values; over-weighting pf hurts.

### R14 — DINOv2-B backbone (141–143 MAE)

Replaced DINOv2-Small (384-dim) with DINOv2-Base (768-dim, 4× parameters).

| Experiment | MAE |
|---|---|
| dinov2b_pf_slat_l10 | 141.65 |
| dinov2b_all_slat_geom | 142.20 |
| dinov2b_pf_noslat_l10 | 143.21 |

> **Finding:** Larger backbone is worse. DINOv2-B overfits at this data scale (~11,500 scenes). DINOv2-S remains the best backbone.

### R15 — Clean geom + slat cross-attention + image in pf_head (in progress)

Three improvements over the R12 best:

1. **`clean_geom`**: Replace 13-dim legacy features (many redundant) with 6 clean features. Voxel count ratio has r=0.42 vs r=0.07 for scale³.
2. **`slat_cross_attn`**: Cross-attention between 256 subsampled voxels of container and object, producing a 64-dim geometric summary.
3. **`image_in_pf_head`**: pf_head receives DINOv2 container+object crops alongside slat features.

| Experiment | Changes vs R12 best |
|---|---|
| clean_geom_pf | clean_geom only |
| slat_cross_pf | clean_geom + slat cross-attn |
| imgpf_clean_geom | clean_geom + DINOv2 in pf_head |
| full_package | all three combined |

---

## Oracle Analysis

What MAE would you get with perfect knowledge of one component?

| Signal | What it means | MAE |
|---|---|---|
| Predict mean count always | No model | 379.7 |
| Perfect pf, mean vr | Perfect packing, worst volume | 851.2 |
| Mean pf, perfect vr | Worst packing, perfect volume | 682.1 |
| geom_est × perfect pf | SAM3D scale with true packing | 532.6 |
| **Best model (R12)** | **DINOv2 + pf supervision** | **127.4** |

Key insight: **decomposing into vr×pf is worse than predicting count directly**, even with oracle components, because vr and pf are anti-correlated (big containers tend to have lower packing density) and vr has enormous variance (CV=1.41). Their product is more predictable than either alone, which is what the DINOv2 model implicitly learns.

---

## Scripts

### `scripts/prepare_sam_data.py`
**CPU only. Run first.**
Selects the best camera frame for each scene (frame with maximum visible object pixels). Writes `sam_data/frame_id.txt`. No data is duplicated — subsequent scripts read images/masks from the original directories using this frame ID.

```bash
python scripts/prepare_sam_data.py --data-dir /path/to/scenes
```

---

### `scripts/generate_instance_features.py`
**Requires GPU. Run after `prepare_sam_data.py`.**
Generates all DINOv2 features from the best frame:
1. Scene-level DINOv2 CLS token → `sam_data/dinov2_s_feats.pt`
2. Container masked crop (box_seg walls) → `sam_data/dinov2_s_container_feats.pt`
3. SAM2 instance segmentation → up to N instance masks → per-instance DINOv2 features → `sam_data/dinov2_s_instance_feats.pt`, `instance_px.pt`
4. Saves `sam_data/best_instance_mask.png` (used by shape embedding script)

```bash
# DINOv2-S (default)
sbatch slurm/generate_instance_features.slurm

# DINOv2-B
sbatch --export=ALL,BACKBONE=dinov2_b slurm/generate_instance_features.slurm

# Overwrite existing
sbatch --export=ALL,OVERWRITE=1 slurm/generate_instance_features.slurm
```

Array job: one task per data split (part1, part2, val).

---

### `scripts/generate_shape_embeddings.py`
**Requires GPU. Run after `generate_instance_features.py`.**
Runs SAM3D on each scene to produce 3D shape embeddings. Uses the best frame and masks from the SAM pipeline:
- Container: `box_seg/Box_Mask{fid}.png`
- Object: `sam_data/best_instance_mask.png`

Output: `sam3d_data/embeddings.pt` containing `container`, `object`, `geom_features`, `geom_estimate`, `true_count`.

```bash
sbatch slurm/generate_shape_embeddings.slurm

# Overwrite existing
sbatch --export=ALL,OVERWRITE=1 slurm/generate_shape_embeddings.slurm
```

Array job: 24 tasks (3 splits × 8 shards each).

> **Note:** After fixing the SAM3D Inf-depth scale bug (`img_and_mask_transforms.py`), re-run with `OVERWRITE=1` to get corrected scale estimates.

---

### `scripts/generate_image_embeddings.py`
**Legacy script.** Generates scene-level or masked crop DINOv2 features from GECO2 masks (not the SAM best-frame pipeline). Kept for reproducing older experiments. For new experiments, use `generate_instance_features.py` instead.

```bash
# Scene-level ResNet50
sbatch slurm/generate_image_embeddings.slurm

# DINOv2-S scene-level
sbatch --export=ALL,BACKBONE=dinov2_s slurm/generate_image_embeddings.slurm

# Masked crops (GECO2 masks, not recommended)
sbatch --export=ALL,BACKBONE=dinov2_s,MASKED=1 slurm/generate_image_embeddings.slurm
```

---

### `scripts/run_training.py`
Main training script. Distributed training via `torchrun` across 2 GPUs. Reads config from a YAML file, trains with DDP, validates every 10 epochs, saves best checkpoint (if `save_checkpoints: true`).

```bash
# Direct
torchrun --nproc_per_node=2 scripts/run_training.py cfg/pf_noslat_l10.yaml

# Via SLURM
sbatch --export=ALL,CONFIG_PATH=/path/to/cfg.yaml slurm/run_training.slurm
```

---

### `scripts/run_experiments.py`
Submits a batch of experiments to SLURM. Reads the active experiment round from `cfg/experiments.py` (`ACTIVE` variable). Generates per-experiment YAML configs and submits one SLURM job per experiment.

```bash
# Edit cfg/experiments.py: set ACTIVE = R15 (or whichever round)
python scripts/run_experiments.py
```

---

### `scripts/collect_results.py`
Prints a summary table of all completed experiment results from `outputs/`.

```bash
python scripts/collect_results.py
```

---

## Configuration

Configs use a YAML + dataclass system. `cfg/baseline.yaml` is the base; experiment scripts generate per-experiment YAMLs by applying overrides.

**Key config fields:**

```yaml
model:
  geometric_feature_dim: 16   # 16 for legacy (13 geom + 3 pixel), 6 for use_clean_geom
  image_feat_dim: 384          # 384 for DINOv2-S, 768 for DINOv2-B
  image_feat_file: "sam_data/dinov2_s_feats.pt"
  container_image_feat_file: "sam_data/dinov2_s_container_feats.pt"
  object_image_feat_file: "sam_data/dinov2_s_instance_feats.pt"

loss:
  log_scale: true              # train on log1p(count) — important for high counts
  pf_lambda: 1.0               # weight for auxiliary packing factor MSE loss

ablation:
  use_image_encoder: true      # scene-level DINOv2
  use_masked_image_encoder: true  # masked container + object DINOv2
  use_slat: false              # slat pooling in count head
  use_packing_factor_head: true
  use_clean_geom: false        # 6-dim clean geom features
  use_slat_cross_attn: false   # cross-attention between voxel sequences
  use_image_in_pf_head: false  # DINOv2 features in pf_head
```

---

## Full Pipeline (from scratch)

```bash
conda activate sam3d-objects

# 1. Select best frames (CPU, fast)
python scripts/prepare_sam_data.py --data-dir /path/to/scenes_part1
python scripts/prepare_sam_data.py --data-dir /path/to/scenes_part2
python scripts/prepare_sam_data.py --data-dir /path/to/scenes_val

# 2. Generate DINOv2 features + SAM2 instance masks (GPU, ~6h per split)
sbatch slurm/generate_instance_features.slurm  # runs all 3 splits as array job

# 3. Generate SAM3D 3D embeddings (GPU, ~24h, 24 array tasks)
sbatch slurm/generate_shape_embeddings.slurm

# 4. Run experiments (edit cfg/experiments.py first)
python scripts/run_experiments.py

# 5. Check results
python scripts/collect_results.py
```

---

## Key Known Issues

1. **SAM3D `geom_estimate` is broken**: The precomputed `geom_estimate` in `embeddings.pt` is always ~1.0 (voxel bbox ratio in normalised space). `run_training.py` recomputes it from real-world `scale`: `0.64 × (cont_scale/obj_scale)³`. Even so, this is noisy (r=0.07) due to monocular depth ambiguity.

2. **SAM3D Inf-depth scale contamination (fixed)**: Depth pixels out of sensor range produce Inf values that contaminate the scale estimate via `nanmedian`. Fixed in `ext/sam-3d-objects/.../img_and_mask_transforms.py`. Re-run shape embeddings with `OVERWRITE=1` to apply the fix.

3. **Race condition on `.pt` files**: If SAM3D embedding jobs and DINOv2 generation jobs run simultaneously on the same data directories, one can write a corrupted `.pt` file that crashes the other. Run embedding generation jobs sequentially, not in parallel.

4. **`use_clean_geom` disables pixel_feats**: When `use_clean_geom=True`, the 3-dim pixel area features are not appended (the 6 clean features replace the full 16-dim legacy block). Set `model.geometric_feature_dim: 6` in experiment configs accordingly.

---

## Environment

```bash
conda activate sam3d-objects

# SLURM cluster: Arnes HPC, 2× V100 per training job
# Module: CUDA/12.1.1

# Key dependencies (see environment.yml):
# - PyTorch 2.x + torchvision
# - transformers (HuggingFace DINOv2)
# - SAM2 (ext/GECO2)
# - SAM3D (ext/sam-3d-objects)
```

PYTHONPATH must include `ext/sam-3d-objects` for SAM3D imports and `ext/GECO2` for SAM2 imports. The SLURM scripts set this automatically.
