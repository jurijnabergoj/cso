#!/bin/bash
#SBATCH --job-name=inst-feats
#SBATCH --partition=gpu
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/inst_feats/%x_%A_%a.out
#SBATCH --error=logs/inst_feats/%x_%A_%a.err

# environment
module load CUDA/12.1.1
source ~/.bashrc
conda activate sam3d-objects

set -e
set -o pipefail

echo "Array task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Job started on $(hostname)"
nvidia-smi

PROJECT_ROOT=/d/hpc/home/jn16867/cso
SAM2_REPO=$PROJECT_ROOT/ext/GECO2
export PYTHONPATH=$PROJECT_ROOT:$SAM2_REPO:$PYTHONPATH

DATA_DIRS=(
    /d/hpc/projects/FRI/jn16867/3d-counting/scenes_part1
    /d/hpc/projects/FRI/jn16867/3d-counting/scenes_part2
    /d/hpc/projects/FRI/jn16867/3d-counting/scenes_val
)
DATA_DIR=${DATA_DIRS[$SLURM_ARRAY_TASK_ID]}
echo "Data dir: $DATA_DIR"

# Path to SAM checkpoint — update this if the checkpoint lives elsewhere
SAM_CHECKPOINT=/d/hpc/projects/FRI/jn16867/checkpoints/sam2_hiera_base_plus.pt

echo "SAM checkpoint: $SAM_CHECKPOINT"

OVERWRITE_FLAG=""
if [ "${OVERWRITE:-0}" = "1" ]; then
    OVERWRITE_FLAG="--overwrite"
    echo "Overwrite mode"
fi

BACKBONE=${BACKBONE:-dinov2_s}
echo "Backbone: $BACKBONE"

python scripts/generate_instance_features.py \
    --data-dir "$DATA_DIR" \
    --sam-checkpoint "$SAM_CHECKPOINT" \
    --backbone "$BACKBONE" \
    --n-instances 5 \
    --min-distance 15 \
    $OVERWRITE_FLAG