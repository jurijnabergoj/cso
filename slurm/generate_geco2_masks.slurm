#!/bin/bash
#SBATCH --job-name=geco2-masks
#SBATCH --partition=gpu
#SBATCH --constraint=v100s
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------
# Safety & debug
# ---------------------------
set -e
set -o pipefail

echo "Job started on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# ---------------------------
# Paths
# ---------------------------
WORKDIR=$HOME/cso
CONTAINER=$HOME/cso/hpc/geco2/geco2.sif
export PYTHONPATH=$WORKDIR/ext/GECO2:$PYTHONPATH

DATA_DIR=/d/hpc/projects/FRI/jn16867/3d-counting/scenes_part2

# ---------------------------
# Run GECO2 inside container
# ---------------------------
singularity exec --nv \
  -B $HOME/mamba:/opt/conda \
  $CONTAINER \
  bash -c "
    unset SSL_CERT_FILE
    unset SSL_CERT_DIR
    cd $WORKDIR
    micromamba run -n geco2 python scripts/generate_geco2_masks.py --data-dir "$DATA_DIR" --tag hf
  "

echo "Job finished"
