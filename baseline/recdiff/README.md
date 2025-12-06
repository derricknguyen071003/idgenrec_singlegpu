# RecDiff Minimal Reimplementation

A minimal, clean reimplementation of RecDiff for social recommendation.

## Files

- `model.py`: Core models (GCN, SDNet, DiffusionProcess) - ~200 lines
- `train.py`: Training script with data loading and evaluation - ~200 lines
- Total: ~400 lines (vs ~1000+ in original)

## Dependencies

- PyTorch
- NumPy
- SciPy
- tqdm

**No DGL required!** Uses pure PyTorch sparse operations.

## Usage

```bash
python train.py --data_path ../RecDiff/datasets/yelp/dataset.pkl \
  --cuda 0 --n_hid 64 --steps 20 --lr 0.001 --difflr 0.001 \
  --batch_size 2048 --n_epoch 150 --topk 20 --reweight
```

## Key Simplifications

1. **No DGL dependency**: Pure PyTorch sparse matrix operations
2. **Single file models**: All model components in one file
3. **Simplified GCN**: Removed unnecessary complexity
4. **Streamlined diffusion**: Essential diffusion operations only
5. **Minimal data handling**: Direct pickle loading
6. **Clean training loop**: Straightforward BPR + diffusion loss

## Architecture

- **GCNModel**: User-item and social graph embeddings using sparse matrix ops
- **SDNet**: Denoising network for diffusion
- **DiffusionProcess**: Forward/reverse diffusion operations

