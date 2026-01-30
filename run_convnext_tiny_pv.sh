#!/bin/bash
#SBATCH -p gpu-a100-8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --job-name=convnext_t_pv
#SBATCH --output=xx/logs_multi/convnext_t_pv_%j.out
#SBATCH --error=xx/logs_multi/convnext_t_pv_%j.err
#SBATCH --mem=64G

cd xx/Datasetforpt/dataset_origin/duzixunlian

python train_convnext.py --model convnext_tiny --dataset pv
