#!/bin/bash
#SBATCH -p gpu-a100-8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --job-name=mobnet_v2_pld
#SBATCH --output=xx/logs_multi/mobnet_v2_pld_%j.out
#SBATCH --error=xx/logs_multi/mobnet_v2_pld_%j.err
#SBATCH --mem=64G

cd xx/Datasetforpt/dataset_origin/duzixunlian

python train_mobilenet.py --model mobilenet_v2 --dataset pld
