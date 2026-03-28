#!/bin/bash
#SBATCH --job-name=img-embed
#SBATCH --partition=gpu
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# environment
module load CUDA/12.1.1
source ~/.bashrc
conda activate sam3d-objects

# safety
set -e
set -o pipefail

echo "Array task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Job started on $(hostname)"
nvidia-smi

PROJECT_ROOT=/d/hpc/home/jn16867/cso
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

DATA_DIRS=(
    /d/hpc/projects/FRI/jn16867/3d-counting/scenes_part1
    /d/hpc/projects/FRI/jn16867/3d-counting/scenes_part2
    /d/hpc/projects/FRI/jn16867/3d-counting/scenes_val
)
DATA_DIR=${DATA_DIRS[$SLURM_ARRAY_TASK_ID]}
echo "Data dir: $DATA_DIR"

export SLURM_ARRAY_TASK_ID=0
export SLURM_ARRAY_TASK_COUNT=1

# To override backbone use: 
#   sbatch --export=ALL,BACKBONE=dinov2_s ...
BACKBONE=${BACKBONE:-resnet50}
echo "Backbone: $BACKBONE"

# Set MASKED=1 to extract per-object masked crops instead of the full scene image:
#   sbatch --export=ALL,BACKBONE=dinov2_s,MASKED=1 slurm/generate_image_embeddings.slurm
MASKED_FLAG=""
if [ "${MASKED:-0}" = "1" ]; then
    MASKED_FLAG="--masked"
    echo "Mode: masked crops"
else
    echo "Mode: scene-level"
fi

python scripts/generate_image_embeddings.py --data-dir "$DATA_DIR" --backbone "$BACKBONE" $MASKED_FLAG