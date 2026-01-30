#!/bin/bash
#SBATCH -p gpu-a100-8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --job-name=mobnet_v3_large
#SBATCH --output=xx/logs_lau/mobnet_v3_large_%j.out
#SBATCH --error=xx/logs_lau/mobnet_v3_large_%j.err
#SBATCH --mem=64G

cd xx/Datasetforpt/train_1and3

python train_mobilenet.py --model mobilenet_v3_large
