# DiffNet2: Neural Influence Diffusion Model for Social Recommendation

This is a PyTorch reimplementation of DiffNet following the format used in TrustSVD and RecDiff. It uses sequential recommendation splits and reuses utilities from the main codebase.

## Overview

DiffNet is a neural influence diffusion model for social recommendation that learns user and item embeddings by diffusing influence through:
1. **Social graph** (user-user connections): Models how social influence propagates
2. **Information graph** (user-item interactions): Models how item consumption patterns influence users

Reference: Wu et al. "A Neural Influence Diffusion Model for Social Recommendation" (SIGIR 2019)

## Key Features

- **Sequential Recommendation Split**: Uses the same split logic as RecDiff/TrustSVD:
  - Last item = test
  - Second to last = validation
  - Rest = training
- **Dual Graph Diffusion**: 
  - 2-layer GCN on social graph for influence diffusion
  - Information graph aggregation for item consumption patterns
- **Oversmoothing Mitigation**: Combines all GCN layers to prevent oversmoothing (following original DiffNet line 115)
- **Reuses Existing Utilities**: Leverages data loading, evaluation, and indexing utilities from the main codebase

## Model Architecture

1. **Initialization**: User and item embeddings (optionally combined with review features)
2. **Information Graph**: Aggregates item embeddings to get user embeddings from consumed items
3. **Social Graph**: 2-layer GCN diffusion through user-user connections
4. **Fusion**: Combines social influence and item consumption patterns
5. **Prediction**: Sigmoid of dot product between user and item embeddings

## Usage

### Basic Training

```bash
bash run_diffnet2.sh
```

### Custom Configuration

```bash
python train_diffnet2.py \
  --data_path ../../rec_datasets \
  --datasets lastfm \
  --cuda 0 \
  --n_factor 64 \
  --lambda_u 0.01 \
  --lambda_v 0.01 \
  --lr 0.001 \
  --batch_size 2048 \
  --n_epoch 150 \
  --topk 20 \
  --patience 6 \
  --eval_every 5 \
  --eval_sequential \
  --max_history 20
```

### Parameters

- `--data_path`: Path to rec_datasets directory
- `--datasets`: Dataset name (e.g., lastfm, yelp)
- `--cuda`: CUDA device ID (-1 for CPU, 0+ for GPU)
- `--n_factor`: Dimension of latent factors (default: 64)
- `--lambda_u`: User embedding regularization (default: 0.01)
- `--lambda_v`: Item embedding regularization (default: 0.01)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 2048)
- `--n_epoch`: Number of epochs (default: 150)
- `--topk`: Top-K for evaluation (default: 20)
- `--patience`: Early stopping patience (default: 6)
- `--eval_every`: Evaluate every N epochs (default: 5)
- `--eval_sequential`: Run sequential evaluation after training
- `--max_history`: Maximum history length for sequential evaluation (default: 20)

## Differences from Original DiffNet

1. **PyTorch Implementation**: Converted from TensorFlow v1 to PyTorch
2. **Sequential Split**: Uses sequential recommendation split instead of random split
3. **Simplified Review Features**: Review feature support is included but disabled by default (can be enabled if review data is available)
4. **Unified Format**: Follows the same structure as TrustSVD for consistency

## Evaluation

The model supports two evaluation modes:

1. **Standard Ranking**: Evaluates on validation/test sets using HR@K and NDCG@K
2. **Sequential Evaluation**: Evaluates using sequential recommendation setting (recommended for sequential splits)

Sequential evaluation masks items in the user's history and predicts the next item.

## Files

- `train_diffnet2.py`: Main training script with DiffNet model implementation
- `run_diffnet2.sh`: Shell script for running training with default parameters
- `README.md`: This file

## Dependencies

- PyTorch
- scipy (for sparse matrices)
- Existing codebase utilities (from `src/utils/`)

## Notes

- The model requires a `friend_sequence.txt` file for social graph. If not available, it will run without social connections.
- Sequential split is the default and recommended setting for this implementation.
