# Discrete Diffusion Implementation Plan

## Overview
Implementing discrete diffusion for dual-view recommendation system (IDGenRec) with:
- Forward process: Corrupt input sequences with cross-view noise
- Reverse process: Train recommenders to denoise and predict clean next-IDs
- Cross-view alignment: KL divergence between item and social preferences

## Architecture
- **Forward Process (IDGenerator)**: Corrupts clean ID sequences with noise
- **Reverse Process (Recommender)**: Denoises corrupted sequences to predict clean next-IDs
- **Dual Views**: Item-view and Social-view with cross-view noise injection

## Implementation Steps

### ✅ Step 1: Core Diffusion Utilities (COMPLETED)
**File**: `src/utils/discrete_diffusion.py`

**Components**:
- `DiscreteDiffusionScheduler`: Noise scheduling (β_t, ᾱ_t computation)
- `corrupt_sequence()`: Token-level corruption with cross-view noise support
- `get_cross_view_tokens()`: Extract tokens unique to one view

**Status**: ✅ Complete and tested

---

### ✅ Step 2: Timestep Embedding (COMPLETED)
**File**: `src/utils/timestep_embedding.py`

**Components**:
- `add_timestep_tokens_to_tokenizer()`: Adds special tokens `[TIMESTEP_0]` through `[TIMESTEP_T-1]`
- `prepend_timestep_token()`: Prepends timestep token to input sequences

**Integration**:
- Modified `SingleRunner.__init__` to add timestep tokens to tokenizer
- Modified training loops to prepend timestep tokens before model forward pass
- Resizes model embeddings to accommodate new tokens

**Status**: ✅ Complete (using prepend approach)

---

### ✅ Step 3: Extend Datasets for Noise Injection (COMPLETED)
**Files**: 
- `src/data/MultiTaskDataset_rec.py`
- `src/data/MultiTaskDataset_rec_social.py`
- `src/processor/Collator.py`

**Components**:
- Datasets load cross-view token information (social tokens for item view, item tokens for social view)
- Datasets return `cross_view_tokens` field in `__getitem__`
- Collator extracts and tokenizes cross-view tokens
- Collator applies corruption with cross-view noise injection
- Collator returns timesteps and noise masks

**Status**: ✅ Complete

---

### ✅ Step 4: Update Collator for Diffusion Fields (COMPLETED)
**File**: `src/processor/Collator.py`

**Components**:
- ✅ Handle timesteps and noise masks in return values
- ✅ Support cross-view token extraction and tokenization
- ✅ Proper batching of diffusion fields
- ✅ Maintain backward compatibility when `use_diffusion=False`
- ✅ Cross-view token tokenization and filtering via `get_cross_view_tokens()`

**Status**: ✅ Complete

---

### ✅ Step 5: Add Noise Prediction Head (COMPLETED)
**File**: `src/models/noise_head.py`

**Components**:
- `NoisePredictionHead` module:
  - Takes T5 encoder hidden states as input
  - Outputs per-position binary predictions (clean/noisy)
  - Shape: `[batch_size, seq_len, hidden_dim] -> [batch_size, seq_len]`
  - Architecture: MLP with GELU activation and dropout
- `create_noise_head()`: Helper function to create head with correct hidden dimension
- Attached to `model_rec` and `model_social` in `SingleRunner.__init__`
- Head parameters initialized with Xavier uniform

**Status**: ✅ Complete

**Dependencies**: Step 2 (timestep embedding)

---

### ✅ Step 6: Modify Training Loop for Multi-Loss (COMPLETED)
**File**: `src/runner/SingleRunner.py`

**Components**:
- Updated `_train_recommender_phase()`:
  - ✅ Extracts timesteps and noise masks from batch
  - ✅ Gets encoder hidden states from T5 model
  - ✅ Computes noise mask predictions from noise head
  - ✅ Computes three loss components:
    - `L_CE`: Standard next-ID prediction (existing)
    - `L_BCE`: Noise mask prediction (new) - with proper alignment handling
    - `L_KL`: KL divergence between views (placeholder for future implementation)
  - ✅ Combines losses: `L_total = L_CE + λ_mask·L_BCE + λ_KL·L_KL`
  - ✅ Handles timestep token alignment for noise mask prediction
  - ✅ Includes noise head in gradient clipping and zeroing
  - ✅ Tracks and logs individual loss components
  - ✅ Logs diffusion metrics to wandb

**Status**: ✅ Complete (KL divergence integration pending coordination between views)

**Dependencies**: Steps 4, 5, 7

---

### ✅ Step 7: Implement KL Divergence Computation (COMPLETED)
**File**: `src/utils/discrete_diffusion.py`

**Components**:
- ✅ `compute_kl_divergence()`: Computes KL(P || Q) from logits
  - Supports temperature scaling
  - Handles 2D `[batch_size, vocab_size]` and 3D `[batch_size, seq_len, vocab_size]` inputs
  - Numerical stability with epsilon clamping
  - Multiple reduction modes: 'mean', 'sum', 'none'
- ✅ `compute_kl_divergence_from_probs()`: Computes KL divergence from probability distributions
  - Normalizes probabilities to ensure they sum to 1
  - Same numerical stability and reduction options

**Status**: ✅ Complete (ready for integration into training loop when both views are coordinated)

**Dependencies**: Step 1

---

### ⏳ Step 8: Add Command-Line Arguments
**File**: `src/utils/utils.py`

**Tasks**:
- ✅ Add `--use_diffusion` (default: 0)
- ✅ Add `--diffusion_timesteps` (default: 100)
- ✅ Add `--diffusion_beta_max` (default: 0.1)
- ✅ Add `--diffusion_cross_prob` (default: 0.5)
- ✅ Add `--lambda_mask` (default: 0.1)
- ✅ Add `--lambda_kl` (default: 0.1)

**Status**: ✅ Complete

---

### ✅ Step 9: Update Social Training Phase (COMPLETED)
**File**: `src/runner/SingleRunner.py`

**Components**:
- ✅ Applied same diffusion modifications to `_train_social_phase()` as in Step 6
- ✅ Noise prediction head (`noise_head_social`) used for social model
- ✅ Multi-loss computation (CE + BCE + KL) implemented
- ✅ Timestep token prepending and alignment handling
- ✅ Gradient clipping and zeroing for noise head
- ✅ Logging and wandb integration with `social_diffusion/` prefix
- ⏳ KL divergence computation (placeholder, requires batch coordination)

**Status**: ✅ Complete

**Dependencies**: Step 6

---

### ⏳ Step 10: Handle Cross-View Data Loading
**Files**: 
- `src/data/MultiTaskDataset_rec.py`
- `src/data/MultiTaskDataset_rec_social.py`
- `src/utils/dataset_utils.py`

**Tasks**:
- ✅ Ensure item dataset can access social view tokens (done in Step 3)
- ⏳ Ensure social dataset can access item view tokens (partially done, may need refinement)
- Handle cases where cross-view data is missing (graceful fallback)
- Optimize cross-view token loading (cache if needed)

**Status**: Partially complete, may need refinement

---

### ✅ Step 11: Update Optimizer Initialization (COMPLETED)
**File**: `src/runner/SingleRunner.py`

**Components**:
- Updated `create_optimizer_and_scheduler()` to include noise head parameters
- Updated `create_optimizer_and_scheduler_3()` to include noise head parameters for both rec and social models
- Noise head parameters follow same weight decay rules as model parameters
- Properly grouped with bias/LayerNorm parameters having no weight decay

**Status**: ✅ Complete

**Dependencies**: Step 5

---

### ✅ Step 12: Add Logging and Monitoring (COMPLETED)
**File**: `src/runner/SingleRunner.py`

**Components**:
- ✅ Basic logging implemented:
  - ✅ `diffusion/ce_loss`: Cross-entropy loss
  - ✅ `diffusion/bce_loss`: BCE loss for noise prediction
  - ✅ `diffusion/kl_loss`: KL divergence loss (when available)
  - ✅ Individual loss component logging every 100 batches
- ✅ Additional metrics implemented:
  - ✅ `diffusion/corruption_rate`: Average corruption rate per batch/epoch
  - ✅ `diffusion/avg_timestep`: Average timestep sampled per batch/epoch
  - ✅ `social_diffusion/corruption_rate`: Social view corruption rate
  - ✅ `social_diffusion/avg_timestep`: Social view average timestep
- ⏳ Cross-view usage tracking (optional, would require batch structure changes)

**Status**: ✅ Complete - all essential metrics implemented

**Dependencies**: Step 6

---

### ⏳ Step 13: Testing and Validation
**Tasks**:
- Test each component independently:
  - ✅ Discrete diffusion utilities (Step 1)
  - ✅ Timestep embedding (Step 2)
  - ✅ Dataset corruption (Step 3)
  - ✅ Noise prediction head (Step 5)
  - ✅ Multi-loss computation (Step 6)
  - ✅ KL divergence (Step 7)
  - ✅ Social training phase (Step 9)
  - ✅ Logging and monitoring (Step 12)
  - ✅ Social training phase (Step 9)
  - ✅ Logging and monitoring (Step 12)
- Validate corruption works correctly:
  - Check corruption rates match expected β̄_t
  - Verify cross-view noise is used when available
  - Ensure noise masks are correct
- Test backward compatibility:
  - Ensure `use_diffusion=0` works as before
  - Verify no breaking changes to existing functionality
- Test with different `run_type` modes:
  - `2id2rec`: Dual views with separate recommenders
  - `2id1rec`: Dual views with unified recommender
  - `1id2rec`: Single ID generator, dual recommenders
- Performance testing:
  - Check training speed impact
  - Monitor memory usage
  - Verify convergence

**Dependencies**: All previous steps

---

## Loss Function

### Final Loss Formula
```
L_total = L_CE + λ_mask·L_BCE + λ_KL·L_KL
```

Where:
- **L_CE**: Cross-entropy loss for next-ID prediction (standard recommendation loss)
- **L_BCE**: Binary cross-entropy loss for noise mask prediction
- **L_KL**: KL divergence between item and social view distributions

### Loss Components

#### L_CE (Next-ID Prediction)
```
L_CE = -∑_τ log p_R(y^0_τ | H_t, t, y^0_{<τ})
```
- Standard sequence-to-sequence cross-entropy
- Predicts clean next-ID `y^0` from noisy history `H_t`
- Conditioned on timestep `t`

#### L_BCE (Noise Mask Prediction)
```
L_BCE = -∑_ℓ [m_ℓ log m̂_ℓ + (1 - m_ℓ) log(1 - m̂_ℓ)]
```
- Binary cross-entropy per token position
- `m_ℓ`: True noise mask (1 = corrupted, 0 = clean)
- `m̂_ℓ`: Predicted noise probability

#### L_KL (Cross-View Alignment)
```
L_KL = KL(p_item(·|H_t^I, H_t^S, t) || p_social(·|H_t^S, H_t^I, t))
```
- KL divergence between item and social recommender distributions
- Aligns preferences learned by both views
- Only computed when both views are available

---

## Key Design Decisions

### 1. Forward Process (Corruption)
- **Location**: Happens in Collator after tokenization
- **Timing**: After IDGenerator phase, before Recommender training
- **Noise Types**: 
  - Cross-view tokens (preferred when available)
  - Random vocabulary tokens (fallback)

### 2. Reverse Process (Denoising)
- **Implementation**: Recommender models act as denoisers
- **Target**: Predict clean next-ID (x₀-prediction variant)
- **Conditioning**: Timestep token + cross-view context

### 3. Timestep Conditioning
- **Method**: Prepend special token `[TIMESTEP_t]` to input
- **Rationale**: Natural for T5, allows explicit attention to timestep

### 4. Cross-View Noise
- **Item View**: Corrupts with tokens from social view (friend sequences)
- **Social View**: Corrupts with tokens from item view (item sequences)
- **Fallback**: Random vocabulary tokens if cross-view unavailable

---

## Implementation Notes

### Current Status
- ✅ Steps 1-9: Complete (Core implementation fully done)
- ✅ Step 11: Complete (Optimizer initialization)
- ✅ Step 12: Complete (Logging and monitoring)
- ⏳ Steps 10, 13: Pending (Cross-view loading refinements, testing)

### Next Steps
1. ✅ Step 4 (Collator) - DONE
2. ✅ Step 5 (Noise prediction head) - DONE
3. ✅ Step 6 (Multi-loss training loop) - DONE
4. ✅ Step 7 (KL divergence computation) - DONE
5. ✅ Step 9 (Update social training phase) - DONE
6. ✅ Step 12 (Logging and monitoring) - DONE
7. Complete remaining steps (10, 13)

### Testing Strategy
- Test incrementally after each step
- Verify backward compatibility
- Monitor training metrics
- Compare with baseline (no diffusion)

---

## Hyperparameters

### Diffusion Parameters
- `diffusion_timesteps`: 100 (number of diffusion steps T)
- `diffusion_beta_max`: 0.1 (maximum corruption probability)
- `diffusion_cross_prob`: 0.5 (probability of using cross-view vs random noise)

### Loss Weights
- `lambda_mask`: 0.1 (weight for noise mask prediction loss)
- `lambda_kl`: 0.1 (weight for KL divergence loss)

### Recommended Values
- Start with small weights (0.01-0.1) and increase gradually
- Monitor individual loss components
- Adjust based on training dynamics

---

## Files Modified/Created

### New Files
- `src/utils/discrete_diffusion.py` ✅
- `src/utils/timestep_embedding.py` ✅
- `src/models/noise_head.py` ✅

### Modified Files
- `src/utils/utils.py` ✅ (command-line args)
- `src/data/MultiTaskDataset_rec.py` ✅ (cross-view token loading)
- `src/data/MultiTaskDataset_rec_social.py` ✅ (cross-view token loading)
- `src/processor/Collator.py` ✅ (diffusion corruption, timestep/noise mask handling)
- `src/runner/SingleRunner.py` ✅ (multi-loss training, noise head integration, optimizer updates)
- `src/utils/dataset_utils.py` ✅ (batch unpacking for diffusion fields)
- `src/utils/discrete_diffusion.py` ✅ (KL divergence functions added)

---

## Future Enhancements (Optional)

1. **Cosine Schedule**: Add cosine noise schedule option
2. **Importance Sampling**: Sample timesteps with higher weight on medium noise levels
3. **Token Denoising**: Predict clean token distribution for corrupted positions
4. **Adaptive Weights**: Automatically adjust loss weights during training
5. **Inference Sampling**: Implement iterative denoising for generation

---

## References

- Discrete Diffusion Models
- Dual-View Recommendation Systems
- Cross-View Alignment via KL Divergence
- T5-based Generative Recommendation



