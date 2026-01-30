#!/bin/bash
#SBATCH -p gpu-a100-8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --job-name=effnet_b0_ue
#SBATCH --output=xx/logs_multi/effnet_b0_ue_%j.out
#SBATCH --error=xx/logs_multi/effnet_b0_ue_%j.err
#SBATCH --mem=64G

cd xx/Datasetforpt/dataset_origin/duzixunlian

python train_efficientnet.py --model efficientnet_b0 --dataset ue
