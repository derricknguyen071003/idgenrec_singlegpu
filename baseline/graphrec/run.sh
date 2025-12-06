#!/bin/bash

# GraphRec2 Training Script
# Usage: ./run.sh [dataset] [optional_args]

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate idgenrec

DATASET=${1:-lastfm}
DATA_PATH=${DATA_PATH:-../../../rec_datasets}
EMBED_DIM=${EMBED_DIM:-64}
BATCH_SIZE=${BATCH_SIZE:-2048}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-2000}
EVAL_EVERY=${EVAL_EVERY:-5}
LR=${LR:-0.001}
EPOCHS=${EPOCHS:-100}
TOPK_LIST=${TOPK_LIST:-5,10,20}
DEVICE=${DEVICE:-cpu}
SEED=${SEED:-2023}

# Create log and models directories if they don't exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/log"
MODELS_DIR="${SCRIPT_DIR}/models"
mkdir -p "${LOG_DIR}"
mkdir -p "${MODELS_DIR}"

# Generate unique log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/graphrec2_${DATASET}_${TIMESTAMP}.log"
MODEL_FILE="${MODELS_DIR}/graphrec2_${DATASET}_${TIMESTAMP}.pth"

echo "=========================================="
echo "Starting GraphRec2 Training and Evaluation"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Data Path: ${DATA_PATH}"
echo "Device: ${DEVICE}"
echo "Log File: ${LOG_FILE}"
echo "Model File: ${MODEL_FILE}"
echo "=========================================="

cd "$(dirname "$0")"

# Run training and redirect all output to log file
# Use -u flag for unbuffered output so it shows in real-time
# Use stdbuf to ensure unbuffered output for tee
python -u run_GraphRec.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --embed_dim ${EMBED_DIM} \
    --batch_size ${BATCH_SIZE} \
    --test_batch_size ${TEST_BATCH_SIZE} \
    --eval_every ${EVAL_EVERY} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --topk_list ${TOPK_LIST} \
    --device ${DEVICE} \
    --seed ${SEED} \
    --save_path "${MODEL_FILE}" \
    2>&1 | stdbuf -oL -eL tee "${LOG_FILE}"

echo ""
echo "=========================================="
echo "Training and Evaluation Completed!"
echo "Model saved to: ${MODEL_FILE}"
echo "Log saved to: ${LOG_FILE}"
echo "=========================================="
