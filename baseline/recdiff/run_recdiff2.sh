#!/bin/bash

# Run RecDiff2 training and evaluation using existing codebase utilities

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate idgenrec

DATASET=${DATASET:-lastfm}
DATA_PATH=${DATA_PATH:-../../rec_datasets}
CUDA=${CUDA:--1}

# Create log and models directories if they don't exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/log"
MODELS_DIR="${SCRIPT_DIR}/models"
mkdir -p "${LOG_DIR}"
mkdir -p "${MODELS_DIR}"

# Generate unique log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/recdiff2_${DATASET}_${TIMESTAMP}.log"
MODEL_FILE="${MODELS_DIR}/recdiff2_${DATASET}_${TIMESTAMP}.pth"

echo "=========================================="
echo "Starting RecDiff2 Training and Evaluation"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Data Path: ${DATA_PATH}"
echo "CUDA Device: ${CUDA}"
echo "Log File: ${LOG_FILE}"
echo "Model File: ${MODEL_FILE}"
echo "=========================================="

# Run training and redirect all output to log file
# Use -u flag for unbuffered output so it shows in real-time
# Use stdbuf to ensure unbuffered output for tee
python -u train.py \
  --data_path ${DATA_PATH} \
  --datasets ${DATASET} \
  --cuda ${CUDA} \
  --n_hid 64 \
  --n_layers 2 \
  --s_layers 2 \
  --emb_size 16 \
  --steps 20 \
  --noise_scale 0.1 \
  --noise_min 0.0001 \
  --noise_max 0.01 \
  --lr 0.001 \
  --difflr 0.001 \
  --reg 0.0001 \
  --batch_size 2048 \
  --n_epoch 150 \
  --topk 20 \
  --patience 6 \
  --eval_every 5 \
  --save_path "${MODEL_FILE}" \
  --reweight \
  2>&1 | stdbuf -oL -eL tee "${LOG_FILE}"

echo ""
echo "=========================================="
echo "Training and Evaluation Completed!"
echo "Model saved to: ${MODEL_FILE}"
echo "Log saved to: ${LOG_FILE}"
echo "=========================================="
