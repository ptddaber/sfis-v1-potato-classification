#!/bin/bash
#SBATCH -p gpu-a100-8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --job-name=mobnet_v2_pv
#SBATCH --output=xx/logs_multi/mobnet_v2_pv_%j.out
#SBATCH --error=xx/logs_multi/mobnet_v2_pv_%j.err
#SBATCH --mem=64G

cd xx/Datasetforpt/dataset_origin/duzixunlian

python train_mobilenet.py --model mobilenet_v2 --dataset pv
