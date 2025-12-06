# GDMSR: Graph Denoising for Sequential Recommendation

This is a reimplementation of the Graph-Denoising-SocialRec framework adapted for sequential recommendation tasks.

## Overview

GDMSR (Graph Denoising for Social Recommendation) is a framework that learns to denoise graphs by periodically removing noisy/redundant edges. The original paper focused on social recommendation (denoising user-user social graphs), while this implementation adapts it for sequential recommendation by denoising user-item interaction graphs.

### Key Features

- **Graph Denoising**: Periodically removes edges from the user-item interaction graph based on learned interaction scores
- **LightGCN Backbone**: Built on top of LightGCN for efficient graph convolution
- **Progressive Edge Removal**: Removes edges every D epochs using exponential moving average of interaction scores
- **Sequential Evaluation**: Supports sequential recommendation evaluation where the last item per user is used as the test target

## Architecture

The model consists of:

1. **LightGCN Encoder**: Multi-layer graph convolution to learn user and item embeddings
2. **Interaction Scoring MLP**: Learns scores for each user-item interaction edge
3. **Periodic Edge Removal**: Every D epochs, removes edges with lowest scores using adaptive drop rate
4. **BPR Loss**: Standard Bayesian Personalized Ranking loss for recommendation

## Usage

### Quick Start

```bash
# Run with default settings (lastfm dataset)
./run.sh

# Or with custom parameters
DATASET=lastfm CUDA=0 ./run.sh
```

### Training Script

```bash
python run_gdmsr.py \
  --data_path ../../rec_datasets \
  --datasets lastfm \
  --cuda 0 \
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
  --save_path models/gdmsr_lastfm.pth \
  --eval_sequential \
  --max_history 20 \
  --sequential_batch_size 1 \
  --seed 2023 \
  --D 10 \
  --beta 0.5 \
  --gamma 1.0 \
  --R 0.5 \
  --epsilon 5
```

### Parameters

- `--data_path`: Path to rec_datasets directory
- `--datasets`: Dataset name (default: lastfm)
- `--cuda`: CUDA device ID (-1 for CPU, 0+ for GPU)
- `--n_factor`: Embedding dimension (default: 64)
- `--gcn_layer`: Number of GCN layers (default: 3)
- `--l2_reg`: L2 regularization coefficient (default: 1e-4)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Training batch size (default: 2048)
- `--num_neg`: Number of negative samples per positive (default: 1)
- `--n_epoch`: Number of training epochs (default: 150)
- `--topk`: Top-K for evaluation (default: 20)
- `--eval_every`: Evaluate every N epochs (default: 5)
- `--save_path`: Path to save model
- `--eval_sequential`: Enable sequential evaluation
- `--max_history`: Maximum history length for sequential evaluation (default: 20)
- `--sequential_batch_size`: Batch size for sequential evaluation (default: 1)
- `--patience`: Early stopping patience (default: 10)
- `--seed`: Random seed (default: 2023)

**GDMSR-specific parameters:**
- `--D`: Denoise every D epochs (default: 10)
- `--beta`: Exponential moving average coefficient for scores (default: 0.5)
- `--gamma`: Scaling factor for drop rate calculation (default: 1.0)
- `--R`: Base drop rate (default: 0.5)
- `--epsilon`: Minimum interactions per user to consider for denoising (default: 5)

## Model Saving and Logging

The implementation follows the same pattern as other baselines:

- **Model Saving**: Models are saved using `torch.save(model.state_dict(), args.save_path)` when validation performance improves
- **Logging**: Uses Python's `logging` module with INFO level, consistent with other baselines
- **Early Stopping**: Stops training if validation performance doesn't improve for `patience` evaluations
- **Best Model Loading**: Automatically loads the best model (based on validation HR@10) for final evaluation

## Graph Denoising Mechanism

The denoising process works as follows:

1. **Score Learning**: The model learns interaction scores for each user-item edge using an MLP
2. **Score Aggregation**: Scores are aggregated using exponential moving average across epochs
3. **Edge Removal**: Every D epochs, edges with lowest scores are removed
4. **Adaptive Drop Rate**: The number of edges removed per user is calculated as:
   ```
   drop_num = log2(num_interactions)^gamma * R
   ```
   This ensures users with more interactions have proportionally more edges removed

## Output

The training script outputs:
- Training loss (BPR, regularization) at each epoch
- Validation metrics (HR@K, NDCG@K) every `eval_every` epochs
- Edge removal statistics during denoising
- Sequential evaluation results on test set (if `--eval_sequential` is enabled)
- Model checkpoints saved to the specified path

## Differences from Original GDMSR

1. **Graph Type**: Original denoises social graph (user-user), this version denoises interaction graph (user-item)
2. **Framework**: Original uses DGL HeteroGCN, this uses LightGCN (simpler, more consistent with other baselines)
3. **Evaluation**: Adapted for sequential recommendation evaluation
4. **Integration**: Uses common utilities from the baseline codebase for data loading and evaluation

## Reference

Original paper: Graph Denoising for Social Recommendation
