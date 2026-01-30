#!/bin/bash
#SBATCH -p gpu-a100-8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --job-name=effnet_v2s
#SBATCH --output=xx/logs_lau/effnet_v2s_%j.out
#SBATCH --error=xx/logs_lau/effnet_v2s_%j.err
#SBATCH --mem=64G

cd xx/Datasetforpt/train_1and3

python train_efficientnet.py --model efficientnet_v2_s
