# HPC Running instructions

## Build the singularity image

Execute into a gpu node:
`
srun --mem=64G --time=12:00:00 --pty bash

singularity build geco2.sif geco2.def
`

#### Create a seperate environment
`
export MAMBA_ROOT_PREFIX=$HOME/mamba
micromamba create -n geco2 python=3.10 -y
`

#### Install cuda 12.6
`
micromamba run -n geco2 pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
`

#### Install dependencies and Deformable-DETR
`
singularity exec --nv \
-B $HOME/mamba:/opt/conda \
-B $PWD:/workspace \
geco2.sif bash -c "
export TORCH_CUDA_ARCH_LIST=7.0 &&
export MAX_JOBS=1 &&
cd /workspace/Deformable-DETR/models/ops &&
micromamba run -n geco2 python -m pip install --no-build-isolation . &&
cd ../../.. &&
cp -r ./Deformable-DETR/models/ops ./models/ops &&
micromamba run -n geco2 pip install \
    gradio \
    gradio_image_prompter \
    hydra-core \
    scikit-image \
    pycocotools \
    einops \
    "numpy<2" \
    huggingface-hub==0.34.3 \
    "pydantic<2.11"
"
`

## Run the demo

Execute into a gpu node:

*Note*: DETR was compiled under a v100 GPU and will not work on a newer one (e.g. H100)

`
srun \
  --partition=gpu \
  --constraint=v100s \
  --gres=gpu:1 \
  --time=12:00:00 \
  --pty bash
`

`
singularity exec --nv -B $PWD:/workspace geco2.sif \
micromamba run -n geco2 bash -c "
cd /workspace/Deformable-DETR/models/ops &&
ls &&
pip install -v --no-build-isolation . &&
cd ../../.. &&
python demo_gradio.py
"
`

or simply 

`
singularity exec --nv \
  -B $HOME/mamba:/opt/conda \
  -B $PWD:/workspace \
  geco2.sif bash -c "
  unset SSL_CERT_FILE &&
  cd /workspace &&
  micromamba run -n geco2 python demo_gradio.py"
`

## Run with Jupyter notebook

Login node:
`
srun \
  --partition=gpu \
  --constraint=v100s \
  --gres=gpu:1 \
  --time=12:00:00 \
  --pty bash

cd GECO2
singularity exec --nv \
  -B $HOME/mamba:/opt/conda \
  -B $PWD:/workspace \
  geco2.sif \
  bash

micromamba run -n geco2 jupyter lab \
  --ip=0.0.0.0 \
  --no-browser \
  --port=8888
`

## Install ipykernel

`
micromamba run -n geco2 pip install jupyterlab ipykernel

micromamba run -n geco2 python -m ipykernel install \
  --user \
  --name geco2 \
  --display-name "Python (geco2)"
`