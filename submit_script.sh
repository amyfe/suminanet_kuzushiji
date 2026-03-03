#!/bin/bash
# SLURM submission script for training on MLO GPU cluster
# Usage: From gpu25/gpu26/gpu27 prototyping node, run: sbatch submit_training.sh
# This submits to the gpu-unlimited partition (gpu28-33)

#SBATCH --gres=gpu:1                  # IMPORTANT: Request 1 GPU (gpu:2 for 2 GPUs, etc)
#SBATCH --time=0-01:20:00             # Max runtime: 1 hour 20 minutes [D]-HH:MM:SS
#SBATCH --job-name=kuzushiji_train    # Job name (optional)
#SBATCH --output=logs/training-%j.out # Output log file (jobid=%j)
#SBATCH --error=logs/training-%j.err  # Error log file (optional)
# NOTE: Memory is NOT pre-allocated by SLURM - your job gets what it needs
# NOTE: 15 vCPUs automatically allocated per GPU - increase only if needed

# Create logs directory if not exists
mkdir -p logs

echo "=========================================="
echo "Kuzushiji Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_GPU MB"
echo "Start time: $(date)"
echo "=========================================="

# Load conda
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate kuzushiji

# Navigate to project
cd /home/afelic/projects/kuzushiji_transcription_and_translation

# Show GPU info
nvidia-smi

echo ""
echo "Starting scripts..."
echo ""

# Run training
# python prepare_codh_annotations.py
# python cleanup_annotations.py
python validate_stage1.py \
    --checkpoint checkpoints/stage1_detection/detector_best.pt \
    --num_samples 20 \
    --split val

# python diagnose_stage1.py
echo ""
echo "=========================================="
echo "Scripts finished!"
echo "End time: $(date)"
echo "=========================================="
