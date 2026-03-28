#!/bin/bash
#SBATCH --job-name=sam3d-embed
#SBATCH --partition=gpu
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- environment ---
module load CUDA/12.1.1
source ~/.bashrc
conda activate sam3d-objects

# --- safety ---
set -e
set -o pipefail

echo "Job $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

PROJECT_ROOT=/d/hpc/home/jn16867/cso
export PYTHONPATH=$PROJECT_ROOT/ext/sam-3d-objects:$PYTHONPATH

DATA_DIR=/d/hpc/projects/FRI/jn16867/3d-counting/scenes_part2

# --- run ---
python scripts/generate_embeddings.py --data-dir "$DATA_DIR" --tag hf
