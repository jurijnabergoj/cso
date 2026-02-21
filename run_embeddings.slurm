#!/bin/bash
#SBATCH --job-name=sam3d-embed
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

# --- run ---
python prepare_embeddings.py
