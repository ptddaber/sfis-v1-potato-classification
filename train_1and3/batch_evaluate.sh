#!/bin/bash
#SBATCH -p gpu-a100-8
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --job-name=batch_evaluate
#SBATCH --mem=64G              # Memory request
#SBATCH --output=xx/logs_lau/batch_evl_%j.out
#SBATCH --error=xx/logs_lau/batch_evl_%j.err

cd xx/Datasetforpt/train_1and3

echo "Job Start at $(date)"
echo "Running on node: $(hostname)"

# Run Python script
python batch_evaluate.py

echo "Job End at $(date)"
