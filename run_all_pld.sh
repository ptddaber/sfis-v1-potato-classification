#!/bin/bash
# Batch submit all PLD dataset training jobs

echo "Submitting all PLD training jobs..."

sbatch run_resnet101_pld.sh
sbatch run_efficientnet_b0_pld.sh
sbatch run_efficientnet_v2s_pld.sh
sbatch run_mobilenet_v2_pld.sh
sbatch run_mobilenet_v3_pld.sh
sbatch run_convnext_tiny_pld.sh
sbatch run_swin_t_pld.sh

echo "All jobs submitted! Use 'squeue -u \$USER' to check status."
