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
    evaluate, evaluate_sequential
)

# Setup logging
setup_logging()
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix


class TrustSVD(nn.Module):
    """
    TrustSVD: Trust-based Matrix Factorization with Social Regularization
    
    This model learns user and item embeddings while incorporating social trust
    information. It uses both explicit ratings and implicit feedback.
    
    Reference: Guo et al. "TrustSVD: Collaborative Filtering with Both the Explicit 
    and Implicit Influence of User Trust and of Item Ratings" (AAAI 2015)
    """
    def __init__(self, n_user, n_item, n_factor=64, lambda_u=0.01, 
                 lambda_v=0.01, lambda_t=0.01):
        """
        Args:
            n_user: Number of users
            n_item: Number of items
            n_factor: Dimension of latent factors
            lambda_u: Regularization for user embeddings
            lambda_v: Regularization for item embeddings
            lambda_t: Regularization for trust relationships
        """
        super(TrustSVD, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = n_factor
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_t = lambda_t
        
        # User and item embeddings
        self.user_emb = nn.Parameter(torch.randn(n_user, n_factor) * 0.1)
        self.item_emb = nn.Parameter(torch.randn(n_item, n_factor) * 0.1)
        
        # User and item biases
        self.user_bias = nn.Parameter(torch.zeros(n_user))
        self.item_bias = nn.Parameter(torch.zeros(n_item))
        
        # Global bias
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        
        # Initialize embeddings
        nn.init.normal_(self.user_emb, std=0.01)
        nn.init.normal_(self.item_emb, std=0.01)
    
    def forward(self, user_ids, item_ids):
        """
        Predict ratings for user-item pairs
        
        Args:
            user_ids: Tensor of user indices
            item_ids: Tensor of item indices
            
        Returns:
            Predicted ratings
        """
        user_emb = self.user_emb[user_ids]
        item_emb = self.item_emb[item_ids]
        user_b = self.user_bias[user_ids]
        item_b = self.item_bias[item_ids]
        
        # Rating prediction: r_ui = μ + b_u + b_i + p_u^T * q_i
        pred = self.global_bias + user_b + item_b + (user_emb * item_emb).sum(dim=1)
        return pred
    
    def predict_all(self):
        """
        Predict ratings for all user-item pairs
        
        Returns:
            Matrix of predicted ratings (n_user x n_item)
        """
        # Compute all predictions: R = μ + b_u + b_i + P * Q^T
        ratings = (self.user_emb @ self.item_emb.t()) + \
                  self.global_bias + \
                  self.user_bias.unsqueeze(1) + \
                  self.item_bias.unsqueeze(0)
        return ratings
    
    def compute_loss(self, user_ids, item_ids, ratings):
        """
        Compute basic loss for explicit feedback
        
        Args:
            user_ids: Tensor of user indices
            item_ids: Tensor of item indices
            ratings: Tensor of observed ratings
            
        Returns:
            Loss and components
        """
        # Explicit feedback loss (MSE for observed ratings)
        pred = self.forward(user_ids, item_ids)
        explicit_loss = ((pred - ratings) ** 2).mean()
        
        # Regularization terms
        reg_loss = self.lambda_u * (self.user_emb[user_ids] ** 2).sum() + \
                   self.lambda_v * (self.item_emb[item_ids] ** 2).sum()
        
        total_loss = explicit_loss + reg_loss
        
        return total_loss, {
            'explicit': explicit_loss.item(),
            'reg': reg_loss.item()
        }
    
    def compute_trust_regularization(self, user_ids, trust_mat, device):
        """
        Compute trust regularization: users should be similar to their trusted friends
        
        Args:
            user_ids: Tensor of user indices
            trust_mat: Sparse trust matrix (scipy csr_matrix)
            device: Device to place tensors
            
        Returns:
            Trust regularization loss
        """
        if trust_mat is None or trust_mat.nnz == 0:
            return torch.tensor(0.0, device=device)
        
        trust_loss = 0.0
        unique_users = torch.unique(user_ids)
        
        for u in unique_users:
            u = u.item()
            # Get trusted friends for this user
            trust_row = trust_mat[u]
            if trust_row.nnz == 0:
                continue
            
            trusted_friends = torch.tensor(trust_row.indices, dtype=torch.long, device=device)
            trust_weights = torch.tensor(trust_row.data, dtype=torch.float32, device=device)
            
            # User embedding
            u_emb = self.user_emb[u]
            # Trusted friends' embeddings
            friend_embs = self.user_emb[trusted_friends]
            
            # Trust regularization: minimize difference between user and weighted average of friends
            # Normalize trust weights
            trust_weights_norm = trust_weights / (trust_weights.sum() + 1e-8)
            friend_avg = (trust_weights_norm.unsqueeze(1) * friend_embs).sum(dim=0)
            
            # L2 distance between user and weighted average of friends
            trust_loss += ((u_emb - friend_avg) ** 2).sum()
        
        if len(unique_users) > 0:
            trust_loss = trust_loss / len(unique_users)
        
        return trust_loss

class TrustSVDDataset(Dataset):
    """Simple dataset for TrustSVD training"""
    def __init__(self, train_mat):
        coo = train_mat.tocoo()
        self.rows = coo.row
        self.cols = coo.col
        self.ratings = coo.data.astype(np.float32)
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.ratings[idx]

def load_data_using_utils(data_path, dataset_name):
    """Wrapper for common_utils.load_data_using_utils with return_social_as_adj=False"""
    return _load_data_using_utils(data_path, dataset_name, return_social_as_adj=False)


def train(args):
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
    if trust_mat is not None:
        logging.info(f"  Trust relationships: {trust_mat.nnz}")
    else:
        logging.info(f"  Trust relationships: 0")
    
    # Create model
    model = TrustSVD(
        n_user=n_user,
        n_item=n_item,
        n_factor=args.n_factor,
        lambda_u=args.lambda_u,
        lambda_v=args.lambda_v,
        lambda_t=args.lambda_t
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Data loaders
    train_loader = DataLoader(
        TrustSVDDataset(train_mat),
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
        loss_components = {'explicit': 0, 'trust': 0, 'reg': 0}
        
        for batch_idx, (user_ids, item_ids, ratings) in enumerate(train_loader):
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            
            # Compute loss
            loss, components = model.compute_loss(user_ids, item_ids, ratings)
            
            # Add trust regularization
            if trust_mat is not None and args.lambda_t > 0:
                trust_loss = model.compute_trust_regularization(user_ids, trust_mat, device)
                loss = loss + args.lambda_t * trust_loss
                components['trust'] = trust_loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += components.get(key, 0)
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, '
                     f'Explicit={loss_components["explicit"]/len(train_loader):.4f}, '
                     f'Trust={loss_components["trust"]/len(train_loader):.4f}, '
                     f'Reg={loss_components["reg"]/len(train_loader):.4f}')
        
        # Evaluate on validation set
        if (epoch + 1) % args.eval_every == 0:
            topk_list = [5, 10, args.topk] if args.topk not in [5, 10] else [5, 10]
            metrics = evaluate(model, val_loader, n_user, n_item, topk_list, device)
            
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
    # Note: With sequential splitting, each user has only 1 test item (last item),
    # so ranking evaluation is redundant - sequential evaluation is more appropriate
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
            model, sequential_test_loader, n_user, n_item, topk_list, device
        )
        
        logging.info("\nSequential Evaluation Results (Test Set):")
        for k in sorted(topk_list):
            logging.info(f"  HR@{k}={sequential_metrics[k]['hit']:.4f}, NDCG@{k}={sequential_metrics[k]['ndcg']:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train TrustSVD using existing codebase utilities')
    parser.add_argument('--data_path', type=str, default='../../rec_datasets', help='Path to rec_datasets directory')
    parser.add_argument('--datasets', type=str, default='lastfm', help='Dataset name')
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device ID (-1 for CPU, 0+ for GPU)')
    parser.add_argument('--n_factor', type=int, default=64, help='Dimension of latent factors')
    parser.add_argument('--lambda_u', type=float, default=0.01, help='User embedding regularization')
    parser.add_argument('--lambda_v', type=float, default=0.01, help='Item embedding regularization')
    parser.add_argument('--lambda_t', type=float, default=0.01, help='Trust regularization weight')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--n_epoch', type=int, default=150, help='Number of epochs')
    parser.add_argument('--topk', type=int, default=20, help='Top-K for evaluation')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save_path', type=str, default='trustsvd_model.pth', help='Path to save model')
    parser.add_argument('--eval_sequential', action='store_true', help='Run sequential evaluation after training')
    parser.add_argument('--max_history', type=int, default=20, help='Maximum history length for sequential evaluation')
    parser.add_argument('--sequential_batch_size', type=int, default=1, help='Batch size for sequential evaluation')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (number of evaluations without improvement)')
    args = parser.parse_args()
    
    train(args)
