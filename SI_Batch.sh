#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=1000_Cifar10_nodist_train
#SBATCH --cpus-per-task=18
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Execute Program
module load 2022
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
# module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
source $HOME/.bashrc
conda activate openai

cd $Home/thesis/distill_diffusion
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/openai/lib

python scripts/image_train.py --data_dir datasets $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS 

