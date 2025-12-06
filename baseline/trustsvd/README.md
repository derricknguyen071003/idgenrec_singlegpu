# TrustSVD Implementation

A PyTorch implementation of TrustSVD, a trust-based matrix factorization algorithm for social recommendation. This implementation integrates with the existing idgenrec codebase by reusing existing data loading utilities.

## Overview

TrustSVD is a trust-based matrix factorization algorithm that incorporates social trust information into collaborative filtering. It learns user and item embeddings while regularizing user embeddings to be similar to their trusted friends.

**Reference**: Guo et al. "TrustSVD: Collaborative Filtering with Both the Explicit and Implicit Influence of User Trust and of Item Ratings" (AAAI 2015)

## Files

- `train_trustsvd_simple.py`: Complete implementation including:
  - TrustSVD model class (~150 lines)
  - Training script with data loading (~540 lines)
  - Ranking evaluation (HR@K, NDCG@K)
  - Sequential evaluation (next-item prediction)
- `run_trustsvd_simple.sh`: Single script that runs training and both evaluations automatically
- `log/`: Directory for training logs (created automatically)
- `models/`: Directory for saved models (created automatically)

## Dependencies

- PyTorch
- NumPy
- SciPy
- Existing idgenrec codebase (for data loading utilities: `utils.ReadLineFromFile`, `indexing.construct_user_sequence_dict`)

## Usage

### Training and Evaluation (All-in-One)

The simplest way to run TrustSVD with automatic training and evaluation:

```bash
# Set dataset name (lastfm, lastfm4i, etc.)
export DATASET=lastfm
export DATA_PATH=../../rec_datasets  # Path to rec_datasets directory
export CUDA=-1  # Use -1 for CPU, or 0, 1, etc. for GPU

# Run training and both evaluations automatically
bash run_trustsvd_simple.sh
```

This will:
1. Train the TrustSVD model
2. Perform ranking evaluation (HR@5, HR@10, NDCG@5, NDCG@10)
3. Perform sequential evaluation (predicting next item in sequence)
4. Save model to `models/trustsvd_{DATASET}_{TIMESTAMP}.pth`
5. Save log to `log/trustsvd_{DATASET}_{TIMESTAMP}.log`

Each run creates unique timestamped files, so you can track multiple experiments.

### Manual Usage

You can also run the training script directly with custom arguments:

```bash
python train_trustsvd_simple.py \
  --data_path ../../rec_datasets \
  --datasets lastfm \
  --cuda -1 \
  --n_factor 64 \
  --lambda_u 0.01 \
  --lambda_v 0.01 \
  --lambda_t 0.01 \
  --lr 0.001 \
  --batch_size 2048 \
  --n_epoch 150 \
  --topk 20 \
  --eval_every 5 \
  --save_path "models/trustsvd_lastfm.pth" \
  --eval_sequential \
  --max_history 20 \
  --sequential_batch_size 1
```

To skip sequential evaluation, simply omit the `--eval_sequential` flag.

## Key Features

- **Self-Contained**: Model and training code in a single file
- **Matrix Factorization**: User and item embeddings with biases
- **Trust Regularization**: Users are regularized to be similar to their trusted friends
- **Integration with Existing Codebase**: Uses `utils.ReadLineFromFile` and `indexing.construct_user_sequence_dict` from idgenrec
- **Efficient Implementation**: Pure PyTorch implementation with sparse matrix support
- **Sequential Split**: Uses sequential (temporal) train/test split by default
- **Dual Evaluation**: Supports both ranking evaluation and sequential evaluation
- **Automatic Logging**: Each run creates timestamped log files
- **Organized Output**: Models and logs saved in separate directories

## Model Architecture

The TrustSVD model consists of:
- User embeddings `P` (n_user × n_factor)
- Item embeddings `Q` (n_item × n_factor)
- User biases `b_u`
- Item biases `b_i`
- Global bias `μ`

Rating prediction: `r_ui = μ + b_u + b_i + P_u^T · Q_i`

## Loss Function

The loss function combines:
1. **Explicit Feedback Loss**: MSE between predicted and observed ratings
2. **Trust Regularization**: L2 distance between user embeddings and weighted average of trusted friends' embeddings
3. **L2 Regularization**: Regularization on user and item embeddings

## Data Format

The script expects data in the same format as the idgenrec codebase:
- `user_sequence.txt`: User sequences in format `user_id item1 item2 ...`
- `friend_sequence.txt`: Social trust network in format `user_id friend1 friend2 ...`

The script automatically:
- Loads sequences using existing utilities
- Creates consecutive ID mappings (0-indexed)
- Performs sequential train/test split (80/20 by default)
- Builds sparse matrices for efficient training

## Hyperparameters

- `--n_factor`: Dimension of latent factors (default: 64)
- `--lambda_u`: User embedding regularization (default: 0.01)
- `--lambda_v`: Item embedding regularization (default: 0.01)
- `--lambda_t`: Trust regularization weight (default: 0.01)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 2048)
- `--n_epoch`: Number of training epochs (default: 150)
- `--topk`: Top-K for evaluation (default: 20)
- `--eval_every`: Evaluate every N epochs (default: 5)
- `--eval_sequential`: Flag to enable sequential evaluation after training (default: False)
- `--max_history`: Maximum history length for sequential evaluation (default: 20)
- `--sequential_batch_size`: Batch size for sequential evaluation (default: 1)

## Evaluation

The model is evaluated using:
- **Hit Rate (HR)@K**: Binary metric (1 if at least one relevant item in top-K, 0 otherwise)
- **NDCG@K**: Normalized Discounted Cumulative Gain at K

Evaluation is performed at K=5 and K=10 by default.

## Output Files

When running the script:
- **Models**: Saved to `models/trustsvd_{DATASET}_{TIMESTAMP}.pth`
  - Example: `models/trustsvd_lastfm_20231205_054800.pth`
- **Logs**: Saved to `log/trustsvd_{DATASET}_{TIMESTAMP}.log`
  - Example: `log/trustsvd_lastfm_20231205_054800.log`

Each run creates unique timestamped files, allowing you to track multiple experiments without overwriting previous results.

## Train/Test Split

The default split is **sequential (temporal)**:
- Training: First 80% of each user's sequence
- Test: Last 20% of each user's sequence

This preserves temporal order, which is appropriate for sequential recommendation tasks.
