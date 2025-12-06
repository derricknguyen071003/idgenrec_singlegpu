#!/bin/bash

# Run GDMSR training and evaluation using existing codebase utilities

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
LOG_FILE="${LOG_DIR}/gdmsr_${DATASET}_${TIMESTAMP}.log"
MODEL_FILE="${MODELS_DIR}/gdmsr_${DATASET}_${TIMESTAMP}.pth"

echo "=========================================="
echo "Starting GDMSR Training and Evaluation"
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
python -u run_gdmsr.py \
  --data_path ${DATA_PATH} \
  --datasets ${DATASET} \
  --cuda ${CUDA} \
  --n_factor 64 \
  --gcn_layer 3 \
  --l2_reg 1e-4 \
  --lr 0.001 \
  --batch_size 2048 \
  --num_neg 1 \
  --n_epoch 150 \
  --topk 20 \
  --patience 6 \
  --eval_every 5 \
  --save_path "${MODEL_FILE}" \
  --eval_sequential \
  --max_history 20 \
  --sequential_batch_size 1 \
  --seed 2023 \
  --D 10 \
  --beta 0.5 \
  --gamma 1.0 \
  --R 0.5 \
  --epsilon 5 \
  2>&1 | stdbuf -oL -eL tee "${LOG_FILE}"

echo ""
echo "=========================================="
echo "Training and Evaluation Completed!"
echo "Model saved to: ${MODEL_FILE}"
echo "Log saved to: ${LOG_FILE}"
echo "=========================================="
