#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=re_done
#SBATCH --cpus-per-task=18
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00


# Execute Program
module load 2022
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
# module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
source $HOME/.bashrc
conda activate openai

cd $Home/thesis/distill_diffusion
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/openai/lib
# export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
# export DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
# export TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

python scripts/image_train.py --data_dir datasets $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --

