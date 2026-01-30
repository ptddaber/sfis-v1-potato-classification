#!/bin/bash
# Batch submit all UE (Uncontrolled Environment) dataset training jobs

echo "Submitting all UE training jobs..."

sbatch run_resnet101_ue.sh
sbatch run_efficientnet_b0_ue.sh
sbatch run_efficientnet_v2s_ue.sh
sbatch run_mobilenet_v2_ue.sh
sbatch run_mobilenet_v3_ue.sh
sbatch run_convnext_tiny_ue.sh
sbatch run_swin_t_ue.sh

echo "All jobs submitted! Use 'squeue -u \$USER' to check status."
