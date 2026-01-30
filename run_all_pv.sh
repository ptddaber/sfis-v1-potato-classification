#!/bin/bash
# Batch submit all PV (Plant Village) dataset training jobs

echo "Submitting all PV training jobs..."

sbatch run_resnet101_pv.sh
sbatch run_efficientnet_b0_pv.sh
sbatch run_efficientnet_v2s_pv.sh
sbatch run_mobilenet_v2_pv.sh
sbatch run_mobilenet_v3_pv.sh
sbatch run_convnext_tiny_pv.sh
sbatch run_swin_t_pv.sh

echo "All jobs submitted! Use 'squeue -u \$USER' to check status."
