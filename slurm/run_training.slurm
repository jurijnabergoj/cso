#!/bin/bash
#SBATCH --job-name=run-training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
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

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

PROJECT_ROOT=/d/hpc/home/jn16867/cso
export PYTHONPATH=$PROJECT_ROOT/ext/sam-3d-objects:$PYTHONPATH

EXP_NAME=${EXP_NAME:-default}
echo "Experiment name: $EXP_NAME"

# --- run ---
python scripts/run_training.py --exp-name "$EXP_NAME"

# Example:
# sbatch --export=ALL,EXP_NAME=train1_300epochs slurm/run_training.slurm
