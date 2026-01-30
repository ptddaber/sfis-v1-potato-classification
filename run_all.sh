#!/bin/bash
# Batch submit all dataset and model training jobs

echo "Submitting ALL training jobs (PLD + UE + PV)..."

# PLD dataset
sbatch run_resnet101_pld.sh
sbatch run_efficientnet_b0_pld.sh
sbatch run_efficientnet_v2s_pld.sh
sbatch run_mobilenet_v2_pld.sh
sbatch run_mobilenet_v3_pld.sh
sbatch run_convnext_tiny_pld.sh
sbatch run_swin_t_pld.sh

# UE dataset
sbatch run_resnet101_ue.sh
sbatch run_efficientnet_b0_ue.sh
sbatch run_efficientnet_v2s_ue.sh
sbatch run_mobilenet_v2_ue.sh
sbatch run_mobilenet_v3_ue.sh
sbatch run_convnext_tiny_ue.sh
sbatch run_swin_t_ue.sh

# PV dataset
sbatch run_resnet101_pv.sh
sbatch run_efficientnet_b0_pv.sh
sbatch run_efficientnet_v2s_pv.sh
sbatch run_mobilenet_v2_pv.sh
sbatch run_mobilenet_v3_pv.sh
sbatch run_convnext_tiny_pv.sh
sbatch run_swin_t_pv.sh

echo "All 21 jobs submitted! Use 'squeue -u \$USER' to check status."
