#!/bin/bash
#SBATCH --job-name=sam3d-embed
#SBATCH --partition=gpu
#SBATCH --array=0-23
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/sam3d-embed/%x_%A_%a.out
#SBATCH --error=logs/sam3d-embed/%x_%A_%a.err

# environment
module load CUDA/12.1.1
source ~/.bashrc
conda activate sam3d-objects

# safety
set -e
set -o pipefail

echo "Array task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

PROJECT_ROOT=/d/hpc/home/jn16867/cso
export PYTHONPATH=$PROJECT_ROOT/ext/sam-3d-objects:$PYTHONPATH

DATA_DIRS=(
    /d/hpc/projects/FRI/jn16867/3d-counting/scenes_part1
    /d/hpc/projects/FRI/jn16867/3d-counting/scenes_part2
    /d/hpc/projects/FRI/jn16867/3d-counting/scenes_val
)

SHARDS_PER_DIR=8
DIR_IDX=$((SLURM_ARRAY_TASK_ID / SHARDS_PER_DIR))
SHARD_IDX=$((SLURM_ARRAY_TASK_ID % SHARDS_PER_DIR))
DATA_DIR=${DATA_DIRS[$DIR_IDX]}

echo "Data dir: $DATA_DIR"
echo "Shard: $SHARD_IDX / $SHARDS_PER_DIR"

OVERWRITE_FLAG=""
if [ "${OVERWRITE:-0}" = "1" ]; then
    OVERWRITE_FLAG="--overwrite"
    echo "Overwrite mode"
fi

# run
python scripts/generate_shape_embeddings.py \
    --data-dir "$DATA_DIR" \
    --tag hf \
    --shard-id "$SHARD_IDX" \
    --num-shards "$SHARDS_PER_DIR" \
    $OVERWRITE_FLAG