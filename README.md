# IDGenRec with Discrete Diffusion

A single-GPU implementation of IDGenRec (ID Generation and Recommendation) enhanced with discrete diffusion for sequential recommendation tasks. This codebase integrates discrete diffusion processes to improve recommendation quality through noise prediction and cross-view alignment.

## Overview

This project implements an alternating training framework for ID generation and recommendation models, with optional discrete diffusion mechanisms. The system supports both item recommendation and social (friend) recommendation tasks, with the ability to leverage cross-view information when both views are available.

### Key Features

- **ID Generation**: Generates user/item IDs using generative models (T5-based)
- **Recommendation**: Sequential recommendation using T5 encoder-decoder architecture
- **Discrete Diffusion**: Optional diffusion-based training with noise prediction and cross-view alignment
- **Multi-View Learning**: Supports item and social (friend) recommendation views with cross-view enhancement
- **Hyperparameter Tuning**: Integrated Optuna support for automated hyperparameter optimization

## Discrete Diffusion Configuration

The discrete diffusion mechanism is implemented to enhance the recommendation model's robustness and enable cross-view alignment. When enabled, the system applies a forward diffusion process during training.

### Diffusion Process

The discrete diffusion follows a forward corruption process where tokens in the input sequence are progressively corrupted over timesteps $t \in \{0, 1, \ldots, T-1\}$.

#### Scheduler Configuration

The diffusion scheduler uses a linear schedule for the corruption probability:

$$\beta_t = \frac{t}{T-1} \cdot \beta_{\text{max}}$$

where:
- $T$ is the number of diffusion timesteps (`--diffusion_timesteps`, default: 100)
- $\beta_{\text{max}}$ is the maximum corruption probability (`--diffusion_beta_max`, default: 0.1)
- $\beta_t$ is the corruption probability at timestep $t$

The survival probability (probability of keeping a token uncorrupted) is computed as:

$$\alpha_t = 1 - \beta_t$$

$$\bar{\alpha}_t = \prod_{s=0}^{t} \alpha_s$$

The corruption probability at timestep $t$ is:

$$P(\text{corrupt}) = 1 - \bar{\alpha}_t$$

#### Corruption Mechanism

For each sequence at timestep $t$:

1. **Corruption Mask Sampling**: Each token position is corrupted with probability $1 - \bar{\alpha}_t$:
   $$\text{noise\_mask}[i] \sim \text{Bernoulli}(1 - \bar{\alpha}_t)$$

2. **Noise Injection**: Corrupted positions are replaced with:
   - **Cross-view tokens** (with probability `--diffusion_cross_prob`, default: 0.5): Tokens from the other view (e.g., social tokens when training item recommendation) that are not present in the current view
   - **Random vocabulary tokens** (with probability $1 - \text{cross\_prob}$): Uniformly sampled from the vocabulary

3. **Timestep Conditioning**: The timestep $t$ is prepended as a special token `[TIMESTEP_t]` to the input sequence to condition the model on the diffusion timestep.

### Loss Functions

When discrete diffusion is enabled, the training objective combines three loss components:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{mask}} \cdot \mathcal{L}_{\text{BCE}} + \lambda_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}$$

#### 1. Cross-Entropy Loss (Next-ID Prediction)

The standard language modeling loss for predicting the next token:

$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{N} \log P(y_i | x_{\text{corrupted}}, t)$$

where:
- $x_{\text{corrupted}}$ is the corrupted input sequence at timestep $t$
- $y_i$ is the target token at position $i$
- $N$ is the sequence length

#### 2. Binary Cross-Entropy Loss (Noise Mask Prediction)

A binary classification head predicts which positions were corrupted:

$$\mathcal{L}_{\text{BCE}} = \frac{1}{N} \sum_{i=1}^{N} \text{BCE}(\text{noise\_logits}_i, \text{noise\_mask}_i)$$

where:
- $\text{noise\_logits}_i$ is the logit from the noise prediction head for position $i$
- $\text{noise\_mask}_i \in \{0, 1\}$ indicates whether position $i$ was corrupted

The noise prediction head is a 2-layer MLP:
$$\text{noise\_logits} = \text{Linear}_2(\text{GELU}(\text{Dropout}(\text{Linear}_1(h_{\text{encoder}}))))$$

where $h_{\text{encoder}}$ are the encoder hidden states from T5.

#### 3. KL Divergence Loss (Cross-View Alignment)

When both item and social views are available, this loss aligns the probability distributions:

$$\mathcal{L}_{\text{KL}} = \text{KL}(P_{\text{item}} || P_{\text{social}}) = \sum_{v \in \mathcal{V}} P_{\text{item}}(v) \log \frac{P_{\text{item}}(v)}{P_{\text{social}}(v)}$$

where:
- $P_{\text{item}}$ and $P_{\text{social}}$ are the softmax probability distributions over vocabulary $\mathcal{V}$ from the item and social models, respectively
- This encourages the two views to have similar token-level preferences

### Configuration Parameters

Discrete diffusion is controlled by the following command-line arguments:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use_diffusion` | int | 0 | Enable (1) or disable (0) discrete diffusion |
| `--diffusion_timesteps` | int | 100 | Number of diffusion timesteps $T$ |
| `--diffusion_beta_max` | float | 0.1 | Maximum corruption probability $\beta_{\text{max}}$ |
| `--diffusion_cross_prob` | float | 0.5 | Probability of using cross-view tokens vs random tokens |
| `--lambda_mask` | float | 0.1 | Weight for noise mask prediction loss $\lambda_{\text{mask}}$ |
| `--lambda_kl` | float | 0.1 | Weight for KL divergence loss $\lambda_{\text{KL}}$ |
| `--noise_head_dropout` | float | 0.1 | Dropout probability for the noise prediction head |

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.7.1+ (with CUDA support)
- Transformers 4.56.2+
- See `requirements.txt` for full dependencies

### Setup

```bash
# Clone the repository
cd idgenrec_singlegpu

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
idgenrec_singlegpu/
├── src/
│   ├── main_generative.py          # Main training script
│   ├── optuna_tune.py              # Optuna hyperparameter tuning
│   ├── data/                       # Dataset classes
│   │   ├── MultiTaskDataset_gen.py
│   │   ├── MultiTaskDataset_rec.py
│   │   └── MultiTaskDataset_social.py
│   ├── runner/
│   │   └── SingleRunner.py         # Training runner with diffusion support
│   ├── processor/
│   │   └── Collator.py             # Data collator with diffusion corruption
│   └── utils/
│       ├── discrete_diffusion.py   # Diffusion scheduler and utilities
│       ├── utils.py                # Configuration and utilities
│       └── evaluate.py             # Evaluation metrics
├── template/                       # Prompt templates
├── rec_datasets/                   # Dataset directory
├── model/                          # Saved model checkpoints
└── log/                            # Training logs
```

## Usage

### Basic Training (Without Diffusion)

```bash
python src/main_generative.py \
    --datasets lastfm4i \
    --data_path ../rec_datasets \
    --run_id baseline_experiment \
    --train 1 \
    --id_epochs 10 \
    --rec_epochs 10 \
    --use_diffusion 0
```

### Training with Discrete Diffusion

```bash
python src/main_generative.py \
    --datasets lastfm4i \
    --data_path ../rec_datasets \
    --run_id diffusion_experiment \
    --train 1 \
    --id_epochs 10 \
    --rec_epochs 10 \
    --use_diffusion 1 \
    --diffusion_timesteps 100 \
    --diffusion_beta_max 0.1 \
    --diffusion_cross_prob 0.5 \
    --lambda_mask 0.1 \
    --lambda_kl 0.1 \
    --noise_head_dropout 0.1
```

### Multi-View Training (2ID2Rec with Diffusion)

For training both item and social recommendation models with cross-view alignment:

```bash
python src/main_generative.py \
    --datasets lastfm4i \
    --data_path ../rec_datasets \
    --run_id 2id2rec_diffusion \
    --train 1 \
    --run_type 2id2rec \
    --use_diffusion 1 \
    --diffusion_timesteps 100 \
    --diffusion_beta_max 0.1 \
    --diffusion_cross_prob 0.5 \
    --lambda_mask 0.1 \
    --lambda_kl 0.1 \
    --social_quantization_id 1 \
    --rounds 3
```

### Hyperparameter Tuning with Optuna

```bash
python src/optuna_tune.py \
    --datasets lastfm4i \
    --data_path ../rec_datasets \
    --run_id optuna_tune \
    --use_diffusion 1 \
    --n_trials 20 \
    --study_name idgenrec_optuna
```

Optuna will automatically tune:
- Learning rates (`id_lr`, `rec_lr`)
- Weight decay and warmup proportion
- Diffusion-specific parameters (`lambda_mask`, `lambda_kl`, `diffusion_beta_max`, `diffusion_cross_prob`, `noise_head_dropout`)

## Key Components

### DiscreteDiffusionScheduler

Located in `src/utils/discrete_diffusion.py`, this class manages the diffusion schedule:

- Computes $\beta_t$ and $\bar{\alpha}_t$ for all timesteps
- Provides `get_alpha_bar(t)` to retrieve survival probabilities

### corrupt_sequence

Function that applies corruption to a sequence:
- Samples corruption mask based on $\bar{\alpha}_t$
- Injects cross-view or random tokens
- Returns corrupted sequence and noise mask

### NoisePredictionHead

A binary classification head that predicts which positions were corrupted:
- Takes T5 encoder hidden states as input
- Outputs per-position binary logits
- Trained with BCE loss against the ground-truth noise mask

### Cross-View Alignment

When `other_view_model` is provided (e.g., in 2ID2Rec mode):
- Both models process the same corrupted input
- KL divergence is computed between their output distributions
- Encourages alignment between item and social preferences

## Evaluation

The model is evaluated using standard recommendation metrics:
- Hit Rate (Hit@K) for K ∈ {1, 5, 10, 20}
- Normalized Discounted Cumulative Gain (NDCG@K) for K ∈ {1, 5, 10, 20}

Metrics are computed during evaluation and logged to both console and WandB (if enabled).

## Logging and Monitoring

- **Console Logging**: Training progress, losses, and metrics are logged to console
- **File Logging**: Logs are saved to `log/train/{dataset}/{run_id}.log`
- **WandB Integration**: Enable with `--use_wandb 1` for experiment tracking

## Model Checkpoints

Models are saved to `model/{dataset}/{run_id}/`:
- `model_gen_item_round{N}_final.pt`: ID generator for items
- `model_rec_item_round{N}_final.pt`: Recommender for items
- `model_gen_friend_round{N}_final.pt`: ID generator for friends (if 2ID2Rec)
- `model_social_friend_round{N}_final.pt`: Social recommender (if 2ID2Rec)

## Citation

If you use this codebase, please cite the original IDGenRec paper and any relevant diffusion-based recommendation papers.

## License

[Add your license information here]

