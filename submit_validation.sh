#!/bin/bash
# SLURM submission script for validation on MLO GPU cluster
# Usage: sbatch submit_validation.sh [checkpoint_path] [split] [num_samples]
# Example: sbatch submit_validation.sh checkpoints/stage1_detection/detector_best.pt val 50

#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --time=0-02:00:00             # Max runtime: 2 hours (validation is fast)
#SBATCH --job-name=kuzushiji_validate # Job name
#SBATCH --output=logs/validation-%j.out # Output log file
#SBATCH --error=logs/validation-%j.err  # Error log file

# Create logs directory if not exists
mkdir -p logs

echo "=========================================="
echo "Kuzushiji Validation Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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
echo "Starting validation..."
echo ""

# Default values
CHECKPOINT="${1:-checkpoints/stage1_detection/detector_best.pt}"
SPLIT="${2:-val}"
NUM_SAMPLES="${3:-0}"  # 0 = full split

# Run validation
python validate_stage1.py \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT" \
    --num_samples "$NUM_SAMPLES" \
    --job_id "$SLURM_JOB_ID" \
    --confidence 0.3 \
    --top_k 400 \
    --nms_iou 0.3 \
    --iou_thr 0.5 \
    --min_box_size 4


echo ""
echo "=========================================="
echo "Validation finished!"
echo "End time: $(date)"
echo "=========================================="
