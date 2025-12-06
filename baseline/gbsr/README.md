# GBSR: Graph Bottlenecked Social Recommendation for Sequential Recommendation

This is a reimplementation of the KDD'24 paper "Graph Bottlenecked Social Recommendation" (GBSR) adapted for sequential recommendation tasks.

## Overview

GBSR is a model-agnostic framework that learns to denoise graphs by removing redundant edges while preserving the minimal yet efficient structure needed for recommendation. The original paper focused on social recommendation (denoising user-user social graphs), while this implementation adapts it for sequential recommendation by denoising user-item interaction graphs.

### Key Features

- **Graph Denoising**: Learns to remove noisy/redundant edges from the user-item interaction graph
- **HSIC Bottleneck**: Uses Hilbert-Schmidt Independence Criterion to minimize information between original and denoised graphs while maximizing information between denoised graph and recommendation labels
- **LightGCN Backbone**: Built on top of LightGCN for efficient graph convolution
- **Sequential Evaluation**: Supports sequential recommendation evaluation where the last item per user is used as the test target

## Architecture

The model consists of:

1. **Graph Reconstruction Module**: Learns masks for interaction edges using preference-guided refinement
2. **LightGCN Encoder**: Multi-layer graph convolution to learn user and item embeddings
3. **HSIC Bottleneck Loss**: Regularizes the model to learn minimal graph structure
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
python run_gbsr.py \
  --data_path ../../rec_datasets \
  --datasets lastfm \
  --cuda 0 \
  --n_factor 64 \
  --gcn_layer 3 \
  --beta 5.0 \
  --sigma 0.25 \
  --edge_bias 0.5 \
  --l2_reg 1e-4 \
  --lr 0.001 \
  --batch_size 2048 \
  --num_neg 1 \
  --n_epoch 150 \
  --topk 20 \
  --patience 6 \
  --eval_every 5 \
  --save_path models/gbsr_lastfm.pth \
  --eval_sequential \
  --max_history 20 \
  --sequential_batch_size 1 \
  --seed 2023
```

### Parameters

- `--data_path`: Path to rec_datasets directory
- `--datasets`: Dataset name (default: lastfm)
- `--cuda`: CUDA device ID (-1 for CPU, 0+ for GPU)
- `--n_factor`: Embedding dimension (default: 64)
- `--gcn_layer`: Number of GCN layers (default: 3)
- `--beta`: Coefficient for HSIC regularization (default: 5.0)
- `--sigma`: Kernel parameter for HSIC (default: 0.25)
- `--edge_bias`: Observation bias for interaction edges (default: 0.5)
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

## Model Saving and Logging

The implementation follows the same pattern as other baselines:

- **Model Saving**: Models are saved using `torch.save(model.state_dict(), args.save_path)` when validation performance improves
- **Logging**: Uses Python's `logging` module with INFO level, consistent with other baselines
- **Early Stopping**: Stops training if validation performance doesn't improve for `patience` evaluations
- **Best Model Loading**: Automatically loads the best model (based on validation HR@10) for final evaluation

## Output

The training script outputs:
- Training loss (BPR, regularization, HSIC) at each epoch
- Validation metrics (HR@K, NDCG@K) every `eval_every` epochs
- Sequential evaluation results on test set (if `--eval_sequential` is enabled)
- Model checkpoints saved to the specified path

## Differences from Original GBSR

1. **Graph Type**: Original denoises social graph (user-user), this version denoises interaction graph (user-item)
2. **Framework**: Original uses TensorFlow, this uses PyTorch
3. **Evaluation**: Adapted for sequential recommendation evaluation
4. **Integration**: Uses common utilities from the baseline codebase for data loading and evaluation

## Reference

Original paper: [Graph Bottlenecked Social Recommendation](https://arxiv.org/abs/2406.08214)

```
@article{GBSR2024,
  title={Graph Bottlenecked Social Recommendation},
  author={Yonghui Yang, Le Wu, Zihan Wang, Zhuangzhuang He, Richang Hong, and Meng Wang},
  journal={30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```
