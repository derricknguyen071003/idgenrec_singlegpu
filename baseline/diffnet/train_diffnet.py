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


class DiffNet(nn.Module):
    """
    DiffNet: A Neural Influence Diffusion Model for Social Recommendation
    
    This model learns user and item embeddings by diffusing influence through:
    1. Social graph (user-user connections)
    2. Information graph (user-item interactions)
    
    Reference: Wu et al. "A Neural Influence Diffusion Model for Social Recommendation" (SIGIR 2019)
    """
    def __init__(self, n_user, n_item, n_factor=64, review_dim=0, 
                 lambda_u=0.01, lambda_v=0.01, use_review=False):
        """
        Args:
            n_user: Number of users
            n_item: Number of items
            n_factor: Dimension of latent factors
            review_dim: Dimension of review features (0 if not used)
            lambda_u: Regularization for user embeddings
            lambda_v: Regularization for item embeddings
            use_review: Whether to use review features
        """
        super(DiffNet, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = n_factor
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.use_review = use_review
        
        # User and item embeddings
        self.user_emb = nn.Parameter(torch.randn(n_user, n_factor) * 0.1)
        self.item_emb = nn.Parameter(torch.randn(n_item, n_factor) * 0.1)
        
        # Review feature processing (if available)
        if use_review and review_dim > 0:
            self.reduce_dim_layer = nn.Linear(review_dim, n_factor)
            self.user_review_features = None  # Will be set from data
            self.item_review_features = None  # Will be set from data
        else:
            self.reduce_dim_layer = None
            self.user_review_features = None
            self.item_review_features = None
        
        # Initialize embeddings
        nn.init.normal_(self.user_emb, std=0.01)
        nn.init.normal_(self.item_emb, std=0.01)
    
    def set_review_features(self, user_review_features, item_review_features):
        """Set review features (if available)"""
        if self.use_review and self.reduce_dim_layer is not None:
            self.user_review_features = user_review_features
            self.item_review_features = item_review_features
    
    def convert_distribution(self, x):
        """Normalize features to controlled distribution"""
        mean = x.mean()
        std = x.std() + 1e-8
        return (x - mean) * 0.2 / std
    
    def init_node_features(self):
        """Initialize node features by combining embeddings with review features"""
        if self.use_review and self.user_review_features is not None:
            # Process review features
            first_user_feature = self.convert_distribution(self.user_review_features)
            first_item_feature = self.convert_distribution(self.item_review_features)
            
            second_user_feature = self.reduce_dim_layer(first_user_feature)
            second_item_feature = self.reduce_dim_layer(first_item_feature)
            
            third_user_feature = self.convert_distribution(second_user_feature)
            third_item_feature = self.convert_distribution(second_item_feature)
            
            # Combine with embeddings
            init_user_feature = third_user_feature + self.user_emb
            init_item_feature = third_item_feature + self.item_emb
        else:
            # Just use embeddings
            init_user_feature = self.user_emb
            init_item_feature = self.item_emb
        
        return init_user_feature, init_item_feature
    
    def gcn_layer(self, adj_sparse, features):
        """
        GCN layer: H' = D^(-1) * A * H + H (with residual)
        
        Args:
            adj_sparse: Sparse adjacency matrix (scipy csr_matrix)
            features: Node features [n_nodes, n_dim]
        
        Returns:
            Updated features
        """
        # Convert to torch sparse COO tensor
        adj_coo = adj_sparse.tocoo()
        indices = torch.LongTensor(np.vstack([adj_coo.row, adj_coo.col]))
        values = torch.FloatTensor(adj_coo.data)
        shape = adj_coo.shape
        adj_torch = torch.sparse_coo_tensor(indices, values, torch.Size(shape), device=features.device)
        
        # Compute degree matrix (row-wise sum)
        degree = torch.sparse.sum(adj_torch, dim=1).to_dense()
        degree = degree + 1e-8  # Avoid division by zero
        degree_inv = 1.0 / degree
        degree_inv = degree_inv.unsqueeze(1)  # [n_nodes, 1]
        
        # GCN: D^(-1) * A * H
        output = torch.sparse.mm(adj_torch, features)
        output = output * degree_inv
        
        # Residual connection
        output = output + features
        
        return output
    
    def forward(self, user_ids, item_ids, social_adj=None, info_adj=None):
        """
        Forward pass through DiffNet
        
        Args:
            user_ids: Tensor of user indices
            item_ids: Tensor of item indices
            social_adj: Social adjacency matrix (user-user) [n_user, n_user]
            info_adj: Information adjacency matrix (user-item) [n_user, n_item]
        
        Returns:
            Predicted ratings
        """
        # Initialize node features
        init_user_feature, init_item_feature = self.init_node_features()
        
        # Information graph: aggregate item embeddings to get user embeddings from consumed items
        # This is done via sparse matrix multiplication: user_emb_from_items = info_adj @ item_emb
        if info_adj is not None and info_adj.nnz > 0:
            # Convert info_adj to torch sparse COO tensor and multiply with item features
            info_coo = info_adj.tocoo()
            info_indices = torch.LongTensor(np.vstack([info_coo.row, info_coo.col]))
            info_values = torch.FloatTensor(info_coo.data)
            info_shape = info_coo.shape
            info_torch = torch.sparse_coo_tensor(info_indices, info_values, torch.Size(info_shape), device=init_item_feature.device)
            
            # Compute degree (number of items per user) for normalization
            user_degree = torch.sparse.sum(info_torch, dim=1).to_dense()
            user_degree = user_degree + 1e-8
            user_degree_inv = 1.0 / user_degree
            user_degree_inv = user_degree_inv.unsqueeze(1)
            
            # Aggregate: user_emb_from_items = (info_adj @ item_emb) / degree
            user_embedding_from_items = torch.sparse.mm(info_torch, init_item_feature)
            user_embedding_from_items = user_embedding_from_items * user_degree_inv
        else:
            user_embedding_from_items = torch.zeros_like(init_user_feature)
        
        # Social graph: diffuse influence through user-user connections
        if social_adj is not None and social_adj.nnz > 0:
            # First GCN layer on social graph
            first_gcn_user_embedding = self.gcn_layer(social_adj, init_user_feature)
            # Second GCN layer on social graph
            second_gcn_user_embedding = self.gcn_layer(social_adj, first_gcn_user_embedding)
            
            # Combine: tackle oversmoothing by adding all layers (as in original DiffNet line 115)
            final_user_embedding = first_gcn_user_embedding + second_gcn_user_embedding + user_embedding_from_items
        else:
            # No social graph, just use information graph
            final_user_embedding = user_embedding_from_items
        
        # Get final item embedding (with review features if available)
        # Use the same init_item_feature computed earlier
        final_item_embedding = self.item_emb + init_item_feature
        
        # Select embeddings for batch
        user_emb = final_user_embedding[user_ids]
        item_emb = final_item_embedding[item_ids]
        
        # Predict ratings: sigmoid(dot product)
        pred = torch.sigmoid((user_emb * item_emb).sum(dim=1))
        
        return pred
    
    def predict_all(self):
        """
        Predict ratings for all user-item pairs
        
        Returns:
            Matrix of predicted ratings (n_user x n_item)
        """
        # This is expensive, so we'll compute it only when needed for evaluation
        # For now, return a placeholder that will be computed in evaluate function
        return None
    
    def compute_loss(self, user_ids, item_ids, ratings, social_adj=None, info_adj=None):
        """
        Compute MSE loss
        
        Args:
            user_ids: Tensor of user indices
            item_ids: Tensor of item indices
            ratings: Tensor of observed ratings
            social_adj: Social adjacency matrix
            info_adj: Information adjacency matrix
        
        Returns:
            Loss and components
        """
        pred = self.forward(user_ids, item_ids, social_adj, info_adj)
        
        # MSE loss
        mse_loss = ((pred - ratings) ** 2).mean()
        
        # Regularization
        reg_loss = self.lambda_u * (self.user_emb[user_ids] ** 2).sum() + \
                   self.lambda_v * (self.item_emb[item_ids] ** 2).sum()
        
        total_loss = mse_loss + reg_loss
        
        return total_loss, {
            'mse': mse_loss.item(),
            'reg': reg_loss.item()
        }


class DiffNetDataset(Dataset):
    """Simple dataset for DiffNet training"""
    def __init__(self, train_mat):
        coo = train_mat.tocoo()
        self.rows = coo.row
        self.cols = coo.col
        self.ratings = coo.data.astype(np.float32)
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.ratings[idx]


def compute_all_predictions(model, n_user, n_item, social_adj, info_adj, device, batch_size=1000):
    """Wrapper for compute_all_predictions_batched with DiffNet-specific forward"""
    def forward_fn(model, user_ids, item_ids, **kwargs):
        return model(user_ids, item_ids, social_adj, info_adj)
    
    return compute_all_predictions_batched(
        model, n_user, n_item, device, batch_size,
        forward_fn=forward_fn
    )


def evaluate(model, test_loader, n_user, n_item, social_adj, info_adj, topk_list=[5, 10, 20], device='cpu'):
    """Evaluate model using Hit Rate@K and NDCG@K - reuses common utilities"""
    def compute_predictions_fn(model, n_user, n_item, device, **kwargs):
        return compute_all_predictions(model, n_user, n_item, social_adj, info_adj, device)
    
    return _evaluate(model, test_loader, n_user, n_item, topk_list, device,
                     compute_predictions_fn=compute_predictions_fn)


def load_data_using_utils(data_path, dataset_name):
    """Wrapper for common_utils.load_data_using_utils with return_social_as_adj=True"""
    return _load_data_using_utils(data_path, dataset_name, return_social_as_adj=True)


def evaluate_sequential(model, test_loader, n_user, n_item, social_adj, info_adj, topk_list=[5, 10], device='cpu'):
    """Evaluate model using sequential setting - reuses common utilities"""
    def compute_predictions_fn(model, n_user, n_item, device, **kwargs):
        return compute_all_predictions(model, n_user, n_item, social_adj, info_adj, device)
    
    return _evaluate_sequential(model, test_loader, n_user, n_item, topk_list, device,
                                compute_predictions_fn=compute_predictions_fn)


def train(args):
    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda}')
    else:
        device = torch.device('cpu')
    
    # Load data using existing utilities
    logging.info(f"Loading data from {args.data_path}/{args.datasets}...")
    train_mat, val_mat, test_mat, social_adj, info_adj, user_id_map, item_id_map, n_user, n_item = load_data_using_utils(
        args.data_path, args.datasets
    )
    
    logging.info(f"Dataset statistics:")
    logging.info(f"  Users: {n_user}")
    logging.info(f"  Items: {n_item}")
    logging.info(f"  Training interactions: {train_mat.nnz}")
    logging.info(f"  Validation interactions: {val_mat.nnz}")
    logging.info(f"  Test interactions: {test_mat.nnz}")
    if social_adj is not None:
        logging.info(f"  Social relationships: {social_adj.nnz}")
    else:
        logging.info(f"  Social relationships: 0")
    logging.info(f"  Information graph edges: {info_adj.nnz}")
    
    # Create model
    model = DiffNet(
        n_user=n_user,
        n_item=n_item,
        n_factor=args.n_factor,
        review_dim=0,  # Not using review features for now
        lambda_u=args.lambda_u,
        lambda_v=args.lambda_v,
        use_review=False
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Data loaders
    train_loader = DataLoader(
        DiffNetDataset(train_mat),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
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
    best_epoch = 0
    patience = args.patience  
    no_improve_count = 0
    
    for epoch in range(args.n_epoch):
        model.train()
        total_loss = 0
        loss_components = {'mse': 0, 'reg': 0}
        
        for batch_idx, (user_ids, item_ids, ratings) in enumerate(train_loader):
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            
            # Compute loss
            loss, components = model.compute_loss(user_ids, item_ids, ratings, social_adj, info_adj)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += components.get(key, 0)
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, '
                     f'MSE={loss_components["mse"]/len(train_loader):.4f}, '
                     f'Reg={loss_components["reg"]/len(train_loader):.4f}')
        
        # Evaluate on validation set
        if (epoch + 1) % args.eval_every == 0:
            topk_list = [5, 10, args.topk] if args.topk not in [5, 10] else [5, 10]
            metrics = evaluate(model, val_loader, n_user, n_item, social_adj, info_adj, topk_list, device)
            
            # Log metrics
            metric_str = []
            for k in sorted(topk_list):
                metric_str.append(f'HR@{k}={metrics[k]["hit"]:.4f}, NDCG@{k}={metrics[k]["ndcg"]:.4f}')
            logging.info(f'Epoch {epoch+1} (Validation): {", ".join(metric_str)}')
            
            # Use HR@10 for model selection on validation set
            hr_10 = metrics[10]['hit']
            if hr_10 > best_hr_10:
                best_hr_10 = hr_10
                best_epoch = epoch + 1
                no_improve_count = 0
                torch.save(model.state_dict(), args.save_path)
                logging.info(f'Model saved with Validation HR@10={hr_10:.4f} (Epoch {best_epoch})')
            else:
                no_improve_count += 1
                # Only trigger early stopping if we've had at least one improvement
                if best_epoch > 0 and no_improve_count >= patience:
                    logging.info(f'Early stopping at epoch {epoch+1}. No improvement for {patience} evaluations.')
                    logging.info(f'Best model was at epoch {best_epoch} with Validation HR@10={best_hr_10:.4f}')
                    break
    
    # Load best model for final evaluation
    if best_epoch > 0:
        logging.info(f'\nLoading best model from epoch {best_epoch} for final evaluation...')
        model.load_state_dict(torch.load(args.save_path, map_location=device))
    
    logging.info(f'Best Validation HR@10: {best_hr_10:.4f} (Epoch {best_epoch})')
    
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
            model, sequential_test_loader, n_user, n_item, social_adj, info_adj, topk_list, device
        )
        
        logging.info("\nSequential Evaluation Results (Test Set):")
        for k in sorted(topk_list):
            logging.info(f"  HR@{k}={sequential_metrics[k]['hit']:.4f}, NDCG@{k}={sequential_metrics[k]['ndcg']:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train DiffNet using existing codebase utilities')
    parser.add_argument('--data_path', type=str, default='../../rec_datasets', help='Path to rec_datasets directory')
    parser.add_argument('--datasets', type=str, default='lastfm', help='Dataset name')
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device ID (-1 for CPU, 0+ for GPU)')
    parser.add_argument('--n_factor', type=int, default=64, help='Dimension of latent factors')
    parser.add_argument('--lambda_u', type=float, default=0.01, help='User embedding regularization')
    parser.add_argument('--lambda_v', type=float, default=0.01, help='Item embedding regularization')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--n_epoch', type=int, default=150, help='Number of epochs')
    parser.add_argument('--topk', type=int, default=20, help='Top-K for evaluation')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save_path', type=str, default='diffnet2_model.pth', help='Path to save model')
    parser.add_argument('--eval_sequential', action='store_true', help='Run sequential evaluation after training')
    parser.add_argument('--max_history', type=int, default=20, help='Maximum history length for sequential evaluation')
    parser.add_argument('--sequential_batch_size', type=int, default=1, help='Batch size for sequential evaluation')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (number of evaluations without improvement)')
    args = parser.parse_args()
    
    train(args)
