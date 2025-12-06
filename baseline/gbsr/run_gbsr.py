"""
GBSR (Graph Bottlenecked Social Recommendation) adapted for Sequential Recommendation

This implementation adapts the KDD'24 GBSR framework for sequential recommendation:
- Instead of denoising social graph (user-user), we denoise interaction graph (user-item)
- Uses LightGCN-style aggregation
- Applies HSIC bottleneck to learn minimal yet efficient graph structure
"""
import sys
import os

# Add baseline directory to path to import common_utils
script_dir = os.path.dirname(os.path.abspath(__file__))
baseline_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, baseline_path)

# Import common utilities
from common_utils import (
    setup_logging, resolve_data_path, parse_id, split_user_sequence,
    load_data_using_utils as _load_data_using_utils, load_sequences_using_utils,
    TestDataset, SequentialTestDataset,
    test_collate_fn, sequential_collate_fn,
    evaluate as _evaluate, evaluate_sequential as _evaluate_sequential,
    compute_all_predictions_batched
)

# Setup logging
setup_logging()
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix, coo_matrix
import random


def kernel_matrix(x, sigma):
    """Compute RBF kernel matrix"""
    # x: [batch_size, dim]
    # Returns: [batch_size, batch_size]
    x_norm = torch.norm(x, dim=1, keepdim=True) ** 2
    pairwise_dist = x_norm + x_norm.t() - 2 * torch.mm(x, x.t())
    return torch.exp(-pairwise_dist / (2 * sigma ** 2))


def hsic(Kx, Ky, m):
    """Compute Hilbert-Schmidt Independence Criterion"""
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / (m ** 2) + torch.mean(Kx) * torch.mean(Ky) - 2 * torch.mean(Kxy) / m
    return h * ((m / (m - 1)) ** 2)


class GBSR_LightGCN(nn.Module):
    """
    GBSR with LightGCN for Sequential Recommendation
    
    Denoises user-item interaction graph using preference-guided refinement
    and HSIC-based bottleneck learning.
    """
    def __init__(self, n_user, n_item, n_factor=64, gcn_layer=3, 
                 beta=5.0, sigma=0.25, edge_bias=0.5, l2_reg=1e-4):
        """
        Args:
            n_user: Number of users
            n_item: Number of items
            n_factor: Embedding dimension
            gcn_layer: Number of GCN layers
            beta: Coefficient for HSIC regularization
            sigma: Kernel parameter for HSIC
            edge_bias: Observation bias for interaction edges
            l2_reg: L2 regularization coefficient
        """
        super(GBSR_LightGCN, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = n_factor
        self.gcn_layer = gcn_layer
        self.beta = beta
        self.sigma = sigma
        self.edge_bias = edge_bias
        self.l2_reg = l2_reg
        
        # User and item embeddings
        self.user_emb = nn.Parameter(torch.randn(n_user, n_factor) * 0.1)
        self.item_emb = nn.Parameter(torch.randn(n_item, n_factor) * 0.1)
        
        # Mask MLPs for graph reconstruction (one per layer)
        self.mask_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_factor * 2, n_factor),
                nn.ReLU(),
                nn.Linear(n_factor, 1)
            ) for _ in range(gcn_layer)
        ])
        
        # Initialize embeddings
        nn.init.normal_(self.user_emb, std=0.01)
        nn.init.normal_(self.item_emb, std=0.01)
    
    def build_lightgcn_adj(self, train_mat):
        """
        Build LightGCN adjacency matrix from user-item interactions
        
        Returns:
            adj_matrix: Combined adjacency [n_user+n_item, n_user+n_item]
            interaction_indices: Indices of user-item edges for masking
        """
        # Create bipartite graph: [user, item] -> [user+n_item, user+n_item]
        n_total = self.n_user + self.n_item
        
        # User-item edges (from train_mat)
        coo = train_mat.tocoo()
        user_indices = coo.row
        item_indices = coo.col + self.n_user  # Offset items
        
        # Build symmetric adjacency
        rows = np.concatenate([user_indices, item_indices])
        cols = np.concatenate([item_indices, user_indices])
        data = np.ones(len(rows))
        
        # Normalize by degree
        adj = csr_matrix((data, (rows, cols)), shape=(n_total, n_total))
        degree = np.array(adj.sum(axis=1)).flatten() + 1e-8
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
        norm_data = degree_inv_sqrt[rows] * data * degree_inv_sqrt[cols]
        
        adj_normalized = csr_matrix((norm_data, (rows, cols)), shape=(n_total, n_total))
        
        # Get interaction edge indices for masking (only user->item direction)
        interaction_indices = np.arange(len(user_indices))
        
        return adj_normalized, interaction_indices, user_indices, item_indices
    
    def graph_reconstruction(self, ego_emb, layer, interaction_indices, user_indices, item_indices):
        """
        Reconstruct graph by learning masks for interaction edges
        
        Args:
            ego_emb: Ego embeddings [n_user+n_item, n_factor]
            layer: Current GCN layer index
            interaction_indices: Indices of interaction edges to mask
            user_indices: User indices for interactions
            item_indices: Item indices for interactions
        
        Returns:
            mask_gate: Learned mask values
            mean_mask: Mean mask value
        """
        # Get embeddings for interaction edges
        user_emb = ego_emb[user_indices]  # [n_interactions, n_factor]
        item_emb = ego_emb[item_indices]  # [n_interactions, n_factor]
        cat_emb = torch.cat([user_emb, item_emb], dim=1)  # [n_interactions, 2*n_factor]
        
        # Compute mask logits
        mask_mlp = self.mask_mlps[layer]
        logit = mask_mlp(cat_emb).squeeze(-1)  # [n_interactions]
        
        # Gumbel-softmax style sampling
        eps = torch.rand_like(logit)
        mask_gate_input = torch.log(eps + 1e-8) - torch.log(1 - eps + 1e-8)
        mask_gate_input = (logit + mask_gate_input) / 0.2
        mask_gate = torch.sigmoid(mask_gate_input) + self.edge_bias
        
        return mask_gate, torch.mean(mask_gate)
    
    def create_masked_lightgcn_emb(self, ego_emb, adj_sparse, masked_adj_sparse):
        """
        Create embeddings with both original and masked graphs
        Returns: (user_emb_old, item_emb_old, user_emb, item_emb)
        """
        # Original graph embeddings
        all_emb = [ego_emb]
        adj_coo = adj_sparse.tocoo()
        adj_indices = torch.LongTensor(np.vstack([adj_coo.row, adj_coo.col]))
        adj_values = torch.FloatTensor(adj_coo.data)
        adj_torch = torch.sparse_coo_tensor(adj_indices, adj_values,
                                           torch.Size(adj_coo.shape),
                                           device=ego_emb.device)
        
        # Masked graph embeddings
        all_emb_masked = [ego_emb]
        masked_coo = masked_adj_sparse.tocoo()
        masked_indices = torch.LongTensor(np.vstack([masked_coo.row, masked_coo.col]))
        masked_values = torch.FloatTensor(masked_coo.data)
        masked_torch = torch.sparse_coo_tensor(masked_indices, masked_values,
                                               torch.Size(masked_coo.shape),
                                               device=ego_emb.device)
        
        for _ in range(self.gcn_layer):
            tmp_emb = torch.sparse.mm(adj_torch, all_emb[-1])
            all_emb.append(tmp_emb)
            
            tmp_emb_masked = torch.sparse.mm(masked_torch, all_emb_masked[-1])
            all_emb_masked.append(tmp_emb_masked)
        
        # Average all layers
        all_emb = torch.stack(all_emb, dim=1)
        mean_emb = torch.mean(all_emb, dim=1)
        user_emb_old, item_emb_old = torch.split(mean_emb, [self.n_user, self.n_item])
        
        all_emb_masked = torch.stack(all_emb_masked, dim=1)
        mean_emb_masked = torch.mean(all_emb_masked, dim=1)
        user_emb, item_emb = torch.split(mean_emb_masked, [self.n_user, self.n_item])
        
        return user_emb_old, item_emb_old, user_emb, item_emb
    
    def forward(self, user_ids, item_ids, train_mat, return_embeddings=False):
        """
        Forward pass
        
        Args:
            user_ids: User indices [batch_size]
            item_ids: Item indices [batch_size]
            train_mat: Training interaction matrix
            return_embeddings: If True, return all embeddings for HSIC loss
        
        Returns:
            predictions: [batch_size] or (predictions, embeddings) if return_embeddings=True
        """
        # Build adjacency matrix (cache it if not already built)
        if not hasattr(self, '_cached_adj_data'):
            adj_sparse, interaction_indices, user_indices, item_indices = self.build_lightgcn_adj(train_mat)
            self._cached_adj_data = (adj_sparse, interaction_indices, user_indices, item_indices)
        else:
            adj_sparse, interaction_indices, user_indices, item_indices = self._cached_adj_data
        
        # Ego embeddings
        ego_emb = torch.cat([self.user_emb, self.item_emb], dim=0)
        
        # Graph reconstruction (mask layer 0)
        mask_gate, mean_mask = self.graph_reconstruction(
            ego_emb, 0, interaction_indices, user_indices, item_indices
        )
        
        # Create masked adjacency
        masked_adj_sparse = self.apply_mask_to_adj(
            adj_sparse, mask_gate, interaction_indices, user_indices, item_indices
        )
        
        # Get embeddings
        user_emb_old, item_emb_old, user_emb, item_emb = self.create_masked_lightgcn_emb(
            ego_emb, adj_sparse, masked_adj_sparse
        )
        
        # Predictions
        user_emb_batch = user_emb[user_ids]
        item_emb_batch = item_emb[item_ids]
        predictions = torch.sum(user_emb_batch * item_emb_batch, dim=1)
        
        if return_embeddings:
            return predictions, (user_emb_old, item_emb_old, user_emb, item_emb)
        return predictions
    
    def apply_mask_to_adj(self, adj_sparse, mask_gate, interaction_indices, user_indices, item_indices):
        """Apply learned masks to adjacency matrix"""
        # Convert to COO for manipulation
        adj_coo = adj_sparse.tocoo()
        rows = adj_coo.row
        cols = adj_coo.col
        data = adj_coo.data.copy()
        
        # Apply mask to user->item edges
        mask_np = mask_gate.detach().cpu().numpy()
        for idx, (u, i) in enumerate(zip(user_indices, item_indices)):
            # Find edge in adjacency
            edge_mask = (rows == u) & (cols == i)
            if np.any(edge_mask):
                data[edge_mask] *= mask_np[idx]
            
            # Also mask symmetric edge (item->user)
            edge_mask = (rows == i) & (cols == u)
            if np.any(edge_mask):
                data[edge_mask] *= mask_np[idx]
        
        masked_adj = csr_matrix((data, (rows, cols)), shape=adj_sparse.shape)
        return masked_adj
    
    def compute_bpr_loss(self, user_ids, pos_item_ids, neg_item_ids, train_mat):
        """Compute BPR loss"""
        predictions, (user_emb_old, item_emb_old, user_emb, item_emb) = self.forward(
            torch.cat([user_ids, user_ids]), 
            torch.cat([pos_item_ids, neg_item_ids]),
            train_mat,
            return_embeddings=True
        )
        
        pos_pred = predictions[:len(user_ids)]
        neg_pred = predictions[len(user_ids):]
        
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_pred - neg_pred) + 1e-8))
        
        # L2 regularization
        user_reg = torch.norm(user_emb[user_ids], p=2) ** 2
        item_reg = torch.norm(item_emb[pos_item_ids], p=2) ** 2 + torch.norm(item_emb[neg_item_ids], p=2) ** 2
        reg_loss = self.l2_reg * (user_reg + item_reg) / len(user_ids)
        
        return bpr_loss, reg_loss, (user_emb_old, item_emb_old, user_emb, item_emb)
    
    def compute_hsic_loss(self, user_ids, item_ids, embeddings):
        """Compute HSIC bottleneck loss"""
        user_emb_old, item_emb_old, user_emb, item_emb = embeddings
        
        # Get unique users and items in batch
        unique_users = torch.unique(user_ids)
        unique_items = torch.unique(item_ids)
        
        # User HSIC
        if len(unique_users) > 1:
            user_emb_old_batch = F.normalize(user_emb_old[unique_users], p=2, dim=1)
            user_emb_batch = F.normalize(user_emb[unique_users], p=2, dim=1)
            Kx_user = kernel_matrix(user_emb_old_batch, self.sigma)
            Ky_user = kernel_matrix(user_emb_batch, self.sigma)
            hsic_user = hsic(Kx_user, Ky_user, len(unique_users))
        else:
            hsic_user = torch.tensor(0.0, device=user_emb.device)
        
        # Item HSIC
        if len(unique_items) > 1:
            item_emb_old_batch = F.normalize(item_emb_old[unique_items], p=2, dim=1)
            item_emb_batch = F.normalize(item_emb[unique_items], p=2, dim=1)
            Kx_item = kernel_matrix(item_emb_old_batch, self.sigma)
            Ky_item = kernel_matrix(item_emb_batch, self.sigma)
            hsic_item = hsic(Kx_item, Ky_item, len(unique_items))
        else:
            hsic_item = torch.tensor(0.0, device=item_emb.device)
        
        return hsic_user + hsic_item
    
    def predict_all(self):
        """Predict for all user-item pairs (for evaluation)"""
        # This will be computed on-the-fly during evaluation
        return None


class GBSRDataset(Dataset):
    """Dataset for GBSR training with negative sampling"""
    def __init__(self, train_mat, num_neg=1):
        """
        Args:
            train_mat: Training interaction matrix (scipy sparse)
            num_neg: Number of negative samples per positive
        """
        coo = train_mat.tocoo()
        self.rows = coo.row
        self.cols = coo.col
        self.num_neg = num_neg
        self.n_item = train_mat.shape[1]
        
        # Get all items for negative sampling
        self.all_items = set(range(self.n_item))
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        user = self.rows[idx]
        pos_item = self.cols[idx]
        
        # Negative sampling
        neg_items = []
        for _ in range(self.num_neg):
            neg_item = random.choice(list(self.all_items - {pos_item}))
            neg_items.append(neg_item)
        
        if self.num_neg == 1:
            return user, pos_item, neg_items[0]
        else:
            return user, pos_item, neg_items


def compute_all_predictions(model, n_user, n_item, train_mat, device, batch_size=1000):
    """Wrapper for compute_all_predictions_batched with GBSR-specific forward"""
    def forward_fn(model, user_ids, item_ids, **kwargs):
        return model(user_ids, item_ids, train_mat, return_embeddings=False)
    
    return compute_all_predictions_batched(
        model, n_user, n_item, device, batch_size,
        forward_fn=forward_fn,
        train_mat=train_mat
    )


def evaluate(model, test_loader, n_user, n_item, train_mat, topk_list=[5, 10, 20], device='cpu'):
    """Evaluate model using Hit Rate@K and NDCG@K - reuses common utilities"""
    def compute_predictions_fn(model, n_user, n_item, device, **kwargs):
        return compute_all_predictions(model, n_user, n_item, train_mat, device)
    
    return _evaluate(model, test_loader, n_user, n_item, topk_list, device,
                     compute_predictions_fn=compute_predictions_fn)


def load_data_using_utils(data_path, dataset_name):
    """Wrapper for common_utils.load_data_using_utils"""
    # For GBSR, we don't need social graph, just user-item interactions
    return _load_data_using_utils(data_path, dataset_name, return_social_as_adj=False)


def evaluate_sequential(model, test_loader, n_user, n_item, train_mat, topk_list=[5, 10], device='cpu'):
    """Evaluate model using sequential setting - reuses common utilities"""
    def compute_predictions_fn(model, n_user, n_item, device, **kwargs):
        return compute_all_predictions(model, n_user, n_item, train_mat, device)
    
    return _evaluate_sequential(model, test_loader, n_user, n_item, topk_list, device,
                                compute_predictions_fn=compute_predictions_fn)


def train(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda}')
    else:
        device = torch.device('cpu')
    
    # Load data using existing utilities
    logging.info(f"Loading data from {args.data_path}/{args.datasets}...")
    train_mat, val_mat, test_mat, trust_mat, user_id_map, item_id_map, n_user, n_item = load_data_using_utils(
        args.data_path, args.datasets
    )
    
    logging.info(f"Dataset statistics:")
    logging.info(f"  Users: {n_user}")
    logging.info(f"  Items: {n_item}")
    logging.info(f"  Training interactions: {train_mat.nnz}")
    logging.info(f"  Validation interactions: {val_mat.nnz}")
    logging.info(f"  Test interactions: {test_mat.nnz}")
    
    # Create model
    logging.info("Creating model...")
    model = GBSR_LightGCN(
        n_user=n_user,
        n_item=n_item,
        n_factor=args.n_factor,
        gcn_layer=args.gcn_layer,
        beta=args.beta,
        sigma=args.sigma,
        edge_bias=args.edge_bias,
        l2_reg=args.l2_reg
    ).to(device)
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Pre-build adjacency matrix to cache it
    logging.info("Pre-building adjacency matrix (this may take a moment)...")
    model._cached_adj_data = model.build_lightgcn_adj(train_mat)
    logging.info("Adjacency matrix cached")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Data loaders
    logging.info("Creating data loaders...")
    train_loader = DataLoader(
        GBSRDataset(train_mat, num_neg=args.num_neg),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    logging.info(f"Training batches: {len(train_loader)}")
    
    # Validation loader (for eval_every and model selection)
    val_loader = DataLoader(
        TestDataset(val_mat, train_mat),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=test_collate_fn
    )
    
    # Training loop with early stopping
    best_hr_10 = 0
    best_ndcg_10 = 0
    best_epoch = 0
    patience = args.patience
    no_improve_count = 0
    
    logging.info("Starting training...")
    for epoch in range(args.n_epoch):
        model.train()
        total_loss = 0
        total_bpr_loss = 0
        total_reg_loss = 0
        total_hsic_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 0 and epoch == 0:
                logging.info(f"Processing first batch...")
            user_ids, pos_item_ids, neg_item_ids = batch
            
            # Convert to tensors if needed
            if not isinstance(user_ids, torch.Tensor):
                user_ids = torch.LongTensor(user_ids)
            if not isinstance(pos_item_ids, torch.Tensor):
                pos_item_ids = torch.LongTensor(pos_item_ids)
            if not isinstance(neg_item_ids, torch.Tensor):
                neg_item_ids = torch.LongTensor(neg_item_ids)
            
            user_ids = user_ids.to(device)
            pos_item_ids = pos_item_ids.to(device)
            neg_item_ids = neg_item_ids.to(device)
            
            # Compute loss
            bpr_loss, reg_loss, embeddings = model.compute_bpr_loss(
                user_ids, pos_item_ids, neg_item_ids, train_mat
            )
            hsic_loss = model.compute_hsic_loss(user_ids, pos_item_ids, embeddings)
            
            total_loss_batch = bpr_loss + reg_loss + args.beta * hsic_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_bpr_loss += bpr_loss.item()
            total_reg_loss += reg_loss.item()
            total_hsic_loss += hsic_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, '
                     f'BPR={total_bpr_loss/len(train_loader):.4f}, '
                     f'Reg={total_reg_loss/len(train_loader):.4f}, '
                     f'HSIC={total_hsic_loss/len(train_loader):.4f}')
        
        # Evaluate on validation set
        if (epoch + 1) % args.eval_every == 0:
            topk_list = [5, 10, args.topk] if args.topk not in [5, 10] else [5, 10]
            metrics = evaluate(model, val_loader, n_user, n_item, train_mat, topk_list, device)
            
            # Log metrics
            metric_str = []
            for k in sorted(topk_list):
                metric_str.append(f'HR@{k}={metrics[k]["hit"]:.4f}, NDCG@{k}={metrics[k]["ndcg"]:.4f}')
            logging.info(f'Epoch {epoch+1} (Validation): {", ".join(metric_str)}')
            
            # Use HR@10 for model selection on validation set
            hr_10 = metrics[10]['hit']
            ndcg_10 = metrics[10]['ndcg']
            if hr_10 > best_hr_10 or (hr_10 == best_hr_10 and ndcg_10 > best_ndcg_10):
                best_hr_10 = hr_10
                best_ndcg_10 = ndcg_10
                best_epoch = epoch + 1
                no_improve_count = 0
                torch.save(model.state_dict(), args.save_path)
                logging.info(f'Model saved with Validation HR@10={hr_10:.4f}, NDCG@10={ndcg_10:.4f} (Epoch {best_epoch})')
            else:
                no_improve_count += 1
                if best_epoch > 0 and no_improve_count >= patience:
                    logging.info(f'Early stopping at epoch {epoch+1}. No improvement for {patience} evaluations.')
                    logging.info(f'Best model was at epoch {best_epoch} with Validation HR@10={best_hr_10:.4f}, NDCG@10={best_ndcg_10:.4f}')
                    break
    
    # Load best model for final evaluation
    if best_epoch > 0:
        logging.info(f'\nLoading best model from epoch {best_epoch} for final evaluation...')
        model.load_state_dict(torch.load(args.save_path, map_location=device))
    
    logging.info(f'Best Validation HR@10: {best_hr_10:.4f}, NDCG@10: {best_ndcg_10:.4f} (Epoch {best_epoch})')
    
    # Sequential evaluation (primary evaluation for sequential splitting)
    if args.eval_sequential:
        logging.info(f'\n=== Sequential Evaluation (Test Set) ===')
        topk_list = [5, 10]
        
        # Load and remap sequences using existing utilities
        user_sequences = load_sequences_using_utils(args.data_path, args.datasets, user_id_map, item_id_map)
        
        # Create train sequences using same split logic
        train_sequences = {}
        for user_id, seq in user_sequences.items():
            train_items, _, _ = split_user_sequence(seq)
            if train_items:
                train_sequences[user_id] = train_items
        
        # Create sequential test dataset (using test items only - last item per user)
        sequential_test_dataset = SequentialTestDataset(
            user_sequences, train_sequences, max_history=args.max_history
        )
        sequential_test_loader = DataLoader(
            sequential_test_dataset,
            batch_size=args.sequential_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=sequential_collate_fn
        )
        
        logging.info(f"Sequential test samples: {len(sequential_test_dataset)}")
        
        # Evaluate sequentially
        sequential_metrics = evaluate_sequential(
            model, sequential_test_loader, n_user, n_item, train_mat, topk_list, device
        )
        
        logging.info("\nSequential Evaluation Results (Test Set):")
        for k in sorted(topk_list):
            logging.info(f"  HR@{k}={sequential_metrics[k]['hit']:.4f}, NDCG@{k}={sequential_metrics[k]['ndcg']:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train GBSR for Sequential Recommendation')
    parser.add_argument('--data_path', type=str, default='../../rec_datasets', help='Path to rec_datasets directory')
    parser.add_argument('--datasets', type=str, default='lastfm', help='Dataset name')
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device ID (-1 for CPU, 0+ for GPU)')
    parser.add_argument('--n_factor', type=int, default=64, help='Dimension of latent factors')
    parser.add_argument('--gcn_layer', type=int, default=3, help='Number of GCN layers')
    parser.add_argument('--beta', type=float, default=5.0, help='Coefficient for HSIC regularization')
    parser.add_argument('--sigma', type=float, default=0.25, help='Kernel parameter for HSIC')
    parser.add_argument('--edge_bias', type=float, default=0.5, help='Observation bias for interaction edges')
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='L2 regularization coefficient')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--num_neg', type=int, default=1, help='Number of negative samples per positive')
    parser.add_argument('--n_epoch', type=int, default=150, help='Number of epochs')
    parser.add_argument('--topk', type=int, default=20, help='Top-K for evaluation')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save_path', type=str, default='gbsr_model.pth', help='Path to save model')
    parser.add_argument('--eval_sequential', action='store_true', help='Run sequential evaluation after training')
    parser.add_argument('--max_history', type=int, default=20, help='Maximum history length for sequential evaluation')
    parser.add_argument('--sequential_batch_size', type=int, default=1, help='Batch size for sequential evaluation')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (number of evaluations without improvement)')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    args = parser.parse_args()
    
    train(args)
