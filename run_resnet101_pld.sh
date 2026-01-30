#!/bin/bash
#SBATCH -p gpu-a100-8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --job-name=resnet101_pld
#SBATCH --output=xx/logs_multi/resnet101_pld_%j.out
#SBATCH --error=xx/logs_multi/resnet101_pld_%j.err
#SBATCH --mem=64G

cd xx/Datasetforpt/dataset_origin/duzixunlian

python train_resnet101_pld.py
