#!/bin/bash
#SBATCH -p gpu-a100-8
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --job-name=swin_t
#SBATCH --output=xx/logs_lau/swin_t_%j.out
#SBATCH --error=xx/logs_lau/swin_t_%j.err
#SBATCH --mem=64G

cd xx/Datasetforpt/train_1and3

python train_swin.py --model swin_t
