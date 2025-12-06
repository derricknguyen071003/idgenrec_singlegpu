"""
GDMSR (Graph Denoising for Social Recommendation) adapted for Sequential Recommendation

This implementation adapts the Graph-Denoising-SocialRec framework for sequential recommendation:
- Instead of denoising social graph (user-user), we denoise interaction graph (user-item)
- Uses LightGCN-style aggregation
- Periodically removes noisy edges based on learned interaction scores
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
import math


class GDMSR_LightGCN(nn.Module):
    """
    GDMSR with LightGCN for Sequential Recommendation
    
    Denoises user-item interaction graph by periodically removing edges
    with low interaction scores.
    """
    def __init__(self, n_user, n_item, n_factor=64, gcn_layer=3, l2_reg=1e-4):
        """
        Args:
            n_user: Number of users
            n_item: Number of items
            n_factor: Embedding dimension
            gcn_layer: Number of GCN layers
            l2_reg: L2 regularization coefficient
        """
        super(GDMSR_LightGCN, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = n_factor
        self.gcn_layer = gcn_layer
        self.l2_reg = l2_reg
        
        # User and item embeddings
        self.user_emb = nn.Parameter(torch.randn(n_user, n_factor) * 0.1)
        self.item_emb = nn.Parameter(torch.randn(n_item, n_factor) * 0.1)
        
        # Interaction scoring MLP (to learn interaction scores)
        self.interaction_scoring = nn.Sequential(
            nn.Linear(n_factor * 2, n_factor),
            nn.ReLU(),
            nn.Linear(n_factor, 1),
            nn.Sigmoid()
        )
        
        # Initialize embeddings
        nn.init.normal_(self.user_emb, std=0.01)
        nn.init.normal_(self.item_emb, std=0.01)
    
    def build_lightgcn_adj(self, train_mat, edge_mask=None):
        """
        Build LightGCN adjacency matrix from user-item interactions
        
        Args:
            train_mat: Training interaction matrix
            edge_mask: Optional mask to remove edges (1 = keep, 0 = remove)
        
        Returns:
            adj_matrix: Combined adjacency [n_user+n_item, n_user+n_item]
            interaction_indices: Indices of user-item edges (original indices before masking)
            user_indices: User indices for interactions (after masking)
            item_indices: Item indices for interactions (after masking)
        """
        # Create bipartite graph: [user, item] -> [user+n_item, user+n_item]
        n_total = self.n_user + self.n_item
        
        # User-item edges (from train_mat)
        coo = train_mat.tocoo()
        user_indices_orig = coo.row
        item_indices_orig = coo.col
        
        # Apply edge mask if provided
        if edge_mask is not None:
            keep_mask = edge_mask == 1
            user_indices_orig = user_indices_orig[keep_mask]
            item_indices_orig = item_indices_orig[keep_mask]
        
        # Offset items for bipartite graph
        user_indices = user_indices_orig
        item_indices = item_indices_orig + self.n_user
        
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
        
        # Get interaction edge indices for scoring (original indices before offset)
        interaction_indices = np.arange(len(user_indices_orig))
        
        return adj_normalized, interaction_indices, user_indices_orig, item_indices_orig
    
    def create_lightgcn_emb(self, ego_emb, adj_sparse):
        """Create LightGCN embeddings by averaging over layers"""
        all_emb = [ego_emb]
        
        # Convert scipy sparse to torch sparse
        adj_coo = adj_sparse.tocoo()
        indices = torch.LongTensor(np.vstack([adj_coo.row, adj_coo.col]))
        values = torch.FloatTensor(adj_coo.data)
        adj_torch = torch.sparse_coo_tensor(indices, values, 
                                           torch.Size(adj_coo.shape), 
                                           device=ego_emb.device)
        
        for _ in range(self.gcn_layer):
            tmp_emb = torch.sparse.mm(adj_torch, all_emb[-1])
            all_emb.append(tmp_emb)
        
        # Average all layers
        all_emb = torch.stack(all_emb, dim=1)
        mean_emb = torch.mean(all_emb, dim=1)
        
        return mean_emb
    
    def compute_interaction_scores(self, ego_emb, user_indices, item_indices):
        """
        Compute interaction scores for all user-item edges
        
        Args:
            ego_emb: Combined embeddings [n_user+n_item, n_factor]
            user_indices: User indices (not offset)
            item_indices: Item indices (not offset)
        
        Returns:
            scores: Interaction scores [n_interactions]
        """
        user_emb = ego_emb[user_indices]  # [n_interactions, n_factor]
        item_emb = ego_emb[item_indices + self.n_user]  # [n_interactions, n_factor] (offset items)
        cat_emb = torch.cat([user_emb, item_emb], dim=1)  # [n_interactions, 2*n_factor]
        
        scores = self.interaction_scoring(cat_emb).squeeze(-1)  # [n_interactions]
        return scores
    
    def forward(self, user_ids, item_ids, train_mat, edge_mask=None, return_scores=False):
        """
        Forward pass
        
        Args:
            user_ids: User indices [batch_size]
            item_ids: Item indices [batch_size]
            train_mat: Training interaction matrix
            edge_mask: Optional mask to remove edges
            return_scores: If True, return interaction scores
        
        Returns:
            predictions: [batch_size] or (predictions, scores) if return_scores=True
        """
        # Build adjacency matrix (with optional edge mask)
        adj_sparse, interaction_indices, user_indices, item_indices = self.build_lightgcn_adj(
            train_mat, edge_mask
        )
        
        # Ego embeddings
        ego_emb = torch.cat([self.user_emb, self.item_emb], dim=0)
        
        # Get embeddings
        mean_emb = self.create_lightgcn_emb(ego_emb, adj_sparse)
        user_emb, item_emb = torch.split(mean_emb, [self.n_user, self.n_item])
        
        # Predictions
        user_emb_batch = user_emb[user_ids]
        item_emb_batch = item_emb[item_ids]
        predictions = torch.sum(user_emb_batch * item_emb_batch, dim=1)
        
        if return_scores:
            # Compute interaction scores for all edges
            scores = self.compute_interaction_scores(ego_emb, user_indices, item_indices)
            return predictions, scores
        return predictions
    
    def compute_bpr_loss(self, user_ids, pos_item_ids, neg_item_ids, train_mat, edge_mask=None):
        """Compute BPR loss"""
        predictions = self.forward(
            torch.cat([user_ids, user_ids]), 
            torch.cat([pos_item_ids, neg_item_ids]),
            train_mat,
            edge_mask=edge_mask
        )
        
        pos_pred = predictions[:len(user_ids)]
        neg_pred = predictions[len(user_ids):]
        
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_pred - neg_pred) + 1e-8))
        
        # L2 regularization
        user_reg = torch.norm(self.user_emb[user_ids], p=2) ** 2
        item_reg = torch.norm(self.item_emb[pos_item_ids], p=2) ** 2 + torch.norm(self.item_emb[neg_item_ids], p=2) ** 2
        reg_loss = self.l2_reg * (user_reg + item_reg) / len(user_ids)
        
        return bpr_loss, reg_loss
    
    def predict_all(self):
        """Predict for all user-item pairs (for evaluation)"""
        return None


class GDMSRDataset(Dataset):
    """Dataset for GDMSR training with negative sampling"""
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


def delete_interaction_edges(interaction_scores, train_mat, epsilon, gamma, R):
    """
    Delete interaction edges with low scores
    
    Args:
        interaction_scores: Learned interaction scores [n_interactions]
        train_mat: Training interaction matrix
        epsilon: Minimum number of interactions per user to consider
        gamma: Scaling factor for drop rate
        R: Base drop rate
    
    Returns:
        edge_mask: Mask for edges (1 = keep, 0 = remove)
    """
    coo = train_mat.tocoo()
    n_interactions = len(coo.row)
    edge_mask = np.ones(n_interactions, dtype=np.int32)
    
    # Group interactions by user
    user_interaction_dict = {}
    for i in range(n_interactions):
        user = coo.row[i]
        if user not in user_interaction_dict:
            user_interaction_dict[user] = []
        user_interaction_dict[user].append(i)
    
    # For each user, remove edges with lowest scores
    for user, interaction_indices in user_interaction_dict.items():
        if len(interaction_indices) < epsilon:
            continue
        
        # Calculate drop number
        drop_num = int(math.pow(int(math.log2(len(interaction_indices))), gamma) * R)
        drop_num = min(drop_num, len(interaction_indices) - 1)  # Keep at least one edge
        
        # Get scores for this user's interactions
        scores_with_indices = [(interaction_scores[i], i) for i in interaction_indices]
        scores_with_indices = sorted(scores_with_indices, key=lambda x: x[0])
        
        # Remove edges with lowest scores
        for j in range(drop_num):
            edge_mask[scores_with_indices[j][1]] = 0
    
    return edge_mask


def compute_all_predictions(model, n_user, n_item, train_mat, device, batch_size=1000, edge_mask=None):
    """Wrapper for compute_all_predictions_batched with GDMSR-specific forward"""
    def forward_fn(model, user_ids, item_ids, train_mat=train_mat, edge_mask=edge_mask, **kwargs):
        return model(user_ids, item_ids, train_mat, edge_mask=edge_mask, return_scores=False)
    
    return compute_all_predictions_batched(
        model, n_user, n_item, device, batch_size,
        forward_fn=forward_fn
    )


def evaluate(model, test_loader, n_user, n_item, train_mat, topk_list=[5, 10, 20], device='cpu', edge_mask=None):
    """Evaluate model using Hit Rate@K and NDCG@K - reuses common utilities"""
    def compute_predictions_fn(model, n_user, n_item, device, **kwargs):
        return compute_all_predictions(model, n_user, n_item, train_mat, device, edge_mask=edge_mask)
    
    return _evaluate(model, test_loader, n_user, n_item, topk_list, device,
                     compute_predictions_fn=compute_predictions_fn)


def load_data_using_utils(data_path, dataset_name):
    """Wrapper for common_utils.load_data_using_utils"""
    return _load_data_using_utils(data_path, dataset_name, return_social_as_adj=False)


def evaluate_sequential(model, test_loader, n_user, n_item, train_mat, topk_list=[5, 10], device='cpu', edge_mask=None):
    """Evaluate model using sequential setting - reuses common utilities"""
    def compute_predictions_fn(model, n_user, n_item, device, **kwargs):
        return compute_all_predictions(model, n_user, n_item, train_mat, device, edge_mask=edge_mask)
    
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
    model = GDMSR_LightGCN(
        n_user=n_user,
        n_item=n_item,
        n_factor=args.n_factor,
        gcn_layer=args.gcn_layer,
        l2_reg=args.l2_reg
    ).to(device)
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Pre-build adjacency matrix to cache it
    logging.info("Pre-building adjacency matrix...")
    model._cached_adj_data = model.build_lightgcn_adj(train_mat)
    logging.info("Adjacency matrix cached")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Data loaders
    logging.info("Creating data loaders...")
    train_loader = DataLoader(
        GDMSRDataset(train_mat, num_neg=args.num_neg),
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
    
    # Initialize edge mask (all edges kept initially)
    coo = train_mat.tocoo()
    edge_mask = np.ones(len(coo.row), dtype=np.int32)
    interaction_scores_epoch = None
    
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
        
        # Convert edge_mask to torch tensor for model
        edge_mask_torch = torch.from_numpy(edge_mask).to(device) if edge_mask is not None else None
        
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
            bpr_loss, reg_loss = model.compute_bpr_loss(
                user_ids, pos_item_ids, neg_item_ids, train_mat, edge_mask=edge_mask
            )
            
            total_loss_batch = bpr_loss + reg_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_bpr_loss += bpr_loss.item()
            total_reg_loss += reg_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, '
                     f'BPR={total_bpr_loss/len(train_loader):.4f}, '
                     f'Reg={total_reg_loss/len(train_loader):.4f}')
        
        # Compute interaction scores for denoising
        if epoch == 0 or epoch % args.D == 0:
            logging.info(f"Computing interaction scores for denoising (epoch {epoch+1})...")
            model.eval()
            with torch.no_grad():
                # Compute scores for ALL edges (not just masked ones)
                ego_emb = torch.cat([model.user_emb, model.item_emb], dim=0)
                coo = train_mat.tocoo()
                user_indices_all = coo.row
                item_indices_all = coo.col
                interaction_scores_all = model.compute_interaction_scores(ego_emb, user_indices_all, item_indices_all)
                interaction_scores_np = interaction_scores_all.cpu().numpy()
            
            # Update interaction scores (exponential moving average)
            n_interactions = len(interaction_scores_np)
            if interaction_scores_epoch is None:
                interaction_scores_epoch = interaction_scores_np.copy()
            else:
                # Only update scores for edges that still exist (if edge_mask is used)
                if edge_mask is not None:
                    keep_mask = edge_mask == 1
                    interaction_scores_epoch[keep_mask] = (
                        args.beta * interaction_scores_epoch[keep_mask] + 
                        (1 - args.beta) * interaction_scores_np[keep_mask]
                    )
                else:
                    interaction_scores_epoch = (
                        args.beta * interaction_scores_epoch + 
                        (1 - args.beta) * interaction_scores_np
                    )
            
            # Delete edges with low scores
            edge_mask = delete_interaction_edges(
                interaction_scores_epoch, train_mat, args.epsilon, args.gamma, args.R
            )
            removed_edges = np.sum(edge_mask == 0)
            logging.info(f"Removed {removed_edges} edges ({removed_edges/len(edge_mask)*100:.2f}%)")
        
        # Evaluate on validation set
        if (epoch + 1) % args.eval_every == 0:
            topk_list = [5, 10, args.topk] if args.topk not in [5, 10] else [5, 10]
            metrics = evaluate(model, val_loader, n_user, n_item, train_mat, topk_list, device, edge_mask=edge_mask)
            
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
            model, sequential_test_loader, n_user, n_item, train_mat, topk_list, device, edge_mask=edge_mask
        )
        
        logging.info("\nSequential Evaluation Results (Test Set):")
        for k in sorted(topk_list):
            logging.info(f"  HR@{k}={sequential_metrics[k]['hit']:.4f}, NDCG@{k}={sequential_metrics[k]['ndcg']:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train GDMSR for Sequential Recommendation')
    parser.add_argument('--data_path', type=str, default='../../rec_datasets', help='Path to rec_datasets directory')
    parser.add_argument('--datasets', type=str, default='lastfm', help='Dataset name')
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device ID (-1 for CPU, 0+ for GPU)')
    parser.add_argument('--n_factor', type=int, default=64, help='Dimension of latent factors')
    parser.add_argument('--gcn_layer', type=int, default=3, help='Number of GCN layers')
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='L2 regularization coefficient')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--num_neg', type=int, default=1, help='Number of negative samples per positive')
    parser.add_argument('--n_epoch', type=int, default=150, help='Number of epochs')
    parser.add_argument('--topk', type=int, default=20, help='Top-K for evaluation')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save_path', type=str, default='gdmsr_model.pth', help='Path to save model')
    parser.add_argument('--eval_sequential', action='store_true', help='Run sequential evaluation after training')
    parser.add_argument('--max_history', type=int, default=20, help='Maximum history length for sequential evaluation')
    parser.add_argument('--sequential_batch_size', type=int, default=1, help='Batch size for sequential evaluation')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (number of evaluations without improvement)')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    # GDMSR-specific parameters
    parser.add_argument('--D', type=int, default=10, help='Denoise every D epochs')
    parser.add_argument('--beta', type=float, default=0.5, help='Exponential moving average coefficient for scores')
    parser.add_argument('--gamma', type=float, default=1.0, help='Scaling factor for drop rate')
    parser.add_argument('--R', type=float, default=0.5, help='Base drop rate')
    parser.add_argument('--epsilon', type=int, default=5, help='Minimum interactions per user to consider for denoising')
    args = parser.parse_args()
    
    train(args)
