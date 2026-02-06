# HPC Running instructions

## Copy checkpoints via SSH
`
scp -r hf \
  jn16867@hpc-login3.arnes.si:/d/hpc/home/jn16867/cso/sam-3d-objects/checkpoints/
`

## Create a seperate environment

`
export MAMBA_ROOT_PREFIX=$HOME/mamba
micromamba create -n sam3d python=3.10 -y
`

## Fix possible installation issues

`
singularity exec --nv \
  -B $HOME/mamba-sam3d:/opt/conda \
  -B $PWD:/workspace \
  sam3d.sif bash -c "
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
./patching/hydra
"
`
### Install missing dependencies:

`
singularity exec --nv -B $HOME/mamba-sam3d:/opt/conda \
-B $HOME/mamba-sam3d:/opt/conda sam3d.sif \
micromamba run -n sam3d pip install \
    numpy \
    Pillow \
    torch \
    kaolin==0.17.0 \
    seaborn==0.13.2 \
    gradio==5.49.0 \
    pytest \
    findpydeps \
    pipdeptree \
    lovely_tensors \
    flash_attn==2.8.3
`

`
singularity exec --nv -B $HOME/mamba-sam3d:/opt/conda \
-B $HOME/mamba-sam3d:/opt/conda sam3d.sif \
micromamba run -n sam3d pip install git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7

`

## Run with Jupyter notebook

Login node:

`
srun --partition=gpu --gres=gpu:2 --mem=64G --time=12:00:00 --pty bash -lc "
module load CUDA/12.1.1 &&
source ~/.bashrc &&
conda activate sam3d-objects &&
jupyter lab --ip=0.0.0.0 --no-browser --port=8888"
`

