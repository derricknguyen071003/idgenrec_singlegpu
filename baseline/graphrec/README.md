# GraphRec2

Reimplementation of GraphRec for social recommendation using `common_utils.py` with sequential data splitting.

## Overview

This implementation follows the pattern established by other baseline models (TrustSVD, DiffNet2, etc.) and uses the `common_utils.py` module for:
- Data loading with sequential split (last item = test, second to last = validation, rest = training)
- Sequential evaluation using `SequentialTestDataset` and `evaluate_sequential`
- Consistent evaluation metrics (Hit@K, NDCG@K)

## Features

- **Graph Neural Network Architecture**: Uses attention mechanisms to aggregate embeddings from:
  - User-item interactions (items rated by user)
  - Social connections (friends of user)
  - Item-item relationships (items rated by same users)
  
- **Sequential Recommendation**: Evaluates using sequential split where only the last item per user is used as test target

- **Social Integration**: Incorporates social adjacency matrix when available

## File Structure

```
graphrec2/
├── model.py          # GraphRec model implementation
├── run_GraphRec.py   # Main training/evaluation script
├── config.py         # Command-line argument parser
└── README.md         # This file
```

## Usage

### Basic Training

```bash
cd command/baseline/graphrec2
python run_GraphRec.py \
    --dataset lastfm \
    --data_path ../../rec_datasets \
    --embed_dim 64 \
    --batch_size 128 \
    --lr 0.001 \
    --epochs 100 \
    --topk_list 5,10,20 \
    --device cuda
```

### Arguments

- `--dataset`: Dataset name (e.g., 'lastfm', 'ciao')
- `--data_path`: Path to rec_datasets directory (default: '../../rec_datasets')
- `--embed_dim`: Embedding dimension (default: 64)
- `--batch_size`: Training batch size (default: 128)
- `--test_batch_size`: Evaluation batch size (default: 1000)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 100)
- `--topk_list`: Top-K values for evaluation, comma-separated (default: '5,10,20')
- `--device`: Device to use ('cuda' or 'cpu', default: 'cuda')
- `--seed`: Random seed (default: 2023)

## Data Format

The implementation expects data in the same format as other baselines:
- `user_sequence.txt`: User interaction sequences
- `friend_sequence.txt`: Social network connections (optional)

Data is automatically split using sequential split:
- Users with 1 item: skipped (no history)
- Users with 2 items: first = train, second = validation
- Users with 3+ items: rest = train, second to last = validation, last = test

## Model Architecture

The GraphRec model consists of:

1. **User Embedding Layer**: Base user embeddings
2. **Item Embedding Layer**: Base item embeddings
3. **Attention Networks**:
   - User-Item Attention: Aggregates items rated by user
   - Social Attention: Aggregates friends' embeddings
   - Item-Item Attention: Aggregates items rated by same users
4. **Prediction Layer**: Combines user and item embeddings for rating prediction

## Evaluation

The model is evaluated using:
- **Hit Rate@K**: Fraction of users whose test item appears in top-K recommendations
- **NDCG@K**: Normalized Discounted Cumulative Gain at K

Evaluation follows the sequential recommendation setting where:
- Training items are masked during evaluation
- Only the last item per user is used as the test target
- Items in the user's history are excluded from recommendations

## Differences from Original GraphRec

1. **Sequential Split**: Uses sequential data splitting instead of random split
2. **Sequential Evaluation**: Evaluates on last item per user with history masking
3. **Common Utilities**: Uses shared utilities for data loading and evaluation
4. **No Explicit Ratings**: Uses binary interactions (ratings default to 1.0 if not provided)

## References

- Original GraphRec paper: "Graph Neural Networks for Social Recommendation" (WWW 2019)
- Implementation follows patterns from TrustSVD and DiffNet2 baselines
