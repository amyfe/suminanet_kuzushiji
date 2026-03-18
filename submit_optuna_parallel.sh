#!/bin/bash
# Launch multiple Optuna workers that share one study/database.
# Usage:
#   sbatch submit_optuna_parallel.sh train 3
#   sbatch submit_optuna_parallel.sh val 3 checkpoints/stage1_detection/detector_best.pt

#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=kuzushiji_optuna
#SBATCH --output=logs/optuna-%j.out
#SBATCH --error=logs/optuna-%j.err
#8450
set -euo pipefail

mkdir -p logs

echo "=========================================="
echo "Kuzushiji Optuna Parallel Launcher"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local-shell}"
echo "Start time: $(date)"
echo "=========================================="

MODE="${1:-train}"
WORKERS="${2:-3}"
CHECKPOINT="${3:-checkpoints/stage1_detection/detector_best.pt}"

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [[ "$WORKERS" -lt 1 ]]; then
  echo "WORKERS must be a positive integer" >&2
  exit 1
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-checkpoints/optuna_stage1_parallel/${RUN_TAG}}"
mkdir -p "$BASE_DIR"

if [[ "$MODE" == "val" ]]; then
  DB_FILE="$BASE_DIR/optuna_val.db"
  STUDY_NAME="${STUDY_NAME:-stage1_detector_optuna_val_parallel_${RUN_TAG}}"
else
  DB_FILE="$BASE_DIR/optuna_train.db"
  STUDY_NAME="${STUDY_NAME:-stage1_detector_optuna_train_parallel_${RUN_TAG}}"
fi

# Use absolute path for sqlite URL so every node resolves the same DB path.
STORAGE="${STORAGE:-sqlite:///${PWD}/${DB_FILE}}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_DIR}"
GLOBAL_MAX_TRIALS="${GLOBAL_MAX_TRIALS:-30}"
N_TRIALS="${N_TRIALS:-1000000}"

printf 'Launching %s workers for mode=%s\n' "$WORKERS" "$MODE"
printf 'study_name=%s\n' "$STUDY_NAME"
printf 'storage=%s\n' "$STORAGE"
printf 'output_dir=%s\n' "$OUTPUT_DIR"
printf 'global_max_trials=%s\n\n' "$GLOBAL_MAX_TRIALS"

for worker in $(seq 1 "$WORKERS"); do
  SEED="$((42 + worker))"

  sbatch \
    --job-name="kuzu_optuna_${MODE}_w${worker}" \
    --export=ALL,N_TRIALS="$N_TRIALS",GLOBAL_MAX_TRIALS="$GLOBAL_MAX_TRIALS",STUDY_NAME="$STUDY_NAME",STORAGE="$STORAGE",OUTPUT_DIR="$OUTPUT_DIR",SEED="$SEED" \
    submit_optuna.sh "$MODE" "$CHECKPOINT"
done

echo "All workers submitted."
echo "=========================================="
echo "Parallel launcher finished"
echo "End time: $(date)"
echo "=========================================="
