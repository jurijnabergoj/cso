# HPC Running instructions

## Copy weights via SSH

Depth Anything v2: 
`
scp weights/depth_anything_v2_vitl.pth jn16867@hpc-login3.arnes.si:/d/hpc/home/jn16867/3d-counting/weights/
`

Dino v2:
`
scp weights/dinov2_vitb14_pretrain.pth jn16867@hpc-login3.arnes.si:/d/hpc/home/jn16867/3d-counting/weights/
`

Density Net:
`
scp weights/density_net_1_0.pth jn16867@hpc-login3.arnes.si:/d/hpc/home/jn16867/3d-counting/weights/
`

## Build the singularity image
`
srun --mem=64G --time=12:00:00 --pty bash

singularity build env.sif environment.def
`

## Run the experiments
Execute into a gpu node:

`
srun --partition=gpu --gres=gpu:1 --pty bash
`

`
singularity exec --nv -B $PWD:/workspace env.sif \
micromamba run -n counting3d bash -c "
cd /workspace &&
pip install -e . &&
bash process_scene.sh /workspace/data/pasta
"
`

## Run with Jupyter notebook

Login node 1:
`
srun --partition=gpu --gres=gpu:2 --time=12:00:00 --pty bash
`

`
singularity exec --nv \
-B $PWD:/workspace env.sif \
micromamba run -n counting3d \
jupyter lab --ip=0.0.0.0 --no-browser --port=8888
`

Login node 2:
`
ssh -N -L 8888:localhost:8888 jn16867@wn208
`

In terminal/wsl:
`
ssh -N -L 8888:localhost:8888 jn16867@hpc-login3.arnes.si
`

In browser: http://127.0.0.1:8888/lab/workspaces/auto-q

Or: http://127.0.0.1:8888/lab
