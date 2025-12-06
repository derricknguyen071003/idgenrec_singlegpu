"""
GraphRec2: Reimplementation using common_utils with sequential split
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_uniform_
from scipy.sparse import csr_matrix

# Add common_utils to path
script_dir = os.path.dirname(os.path.abspath(__file__))
common_utils_path = os.path.join(script_dir, '..')
sys.path.insert(0, common_utils_path)

from common_utils import (
    setup_logging,
    resolve_data_path,
    load_data_using_utils,
    load_sequences_using_utils,
    SequentialTestDataset,
    sequential_collate_fn,
    evaluate_sequential,
    compute_all_predictions_batched
)

import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class GraphRec(nn.Module):
    """
    GraphRec model for social recommendation
    
    Architecture:
    - User-item graph encoder with attention
    - Social graph encoder with attention  
    - Item-item graph encoder with attention
    - Final prediction layer combining all embeddings
    """
    
    def __init__(self, n_users, n_items, embed_dim=64, social_adj=None, info_adj=None, 
                 history_u=None, history_i=None, history_ur=None, history_ir=None,
                 batch_size=128, device='cpu'):
        """
        Args:
            n_users: Number of users
            n_items: Number of items
            embed_dim: Embedding dimension
            social_adj: Social adjacency matrix (n_users x n_users)
            info_adj: Item interaction adjacency matrix (n_users x n_items)
            history_u: User interaction history (list of lists)
            history_i: Item interaction history (list of lists)
            history_ur: User ratings history (list of lists)
            history_ir: Item ratings history (list of lists)
            batch_size: Batch size for training
            device: Device to run on
        """
        super(GraphRec, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.device = device
        self.batch_size = batch_size
        
        # Embeddings
        self.embed_user = nn.Embedding(n_users, embed_dim)
        self.embed_item = nn.Embedding(n_items, embed_dim)
        
        # Initialize embeddings
        xavier_uniform_(self.embed_user.weight)
        xavier_uniform_(self.embed_item.weight)
        
        # Build history dictionaries
        self.history_u = history_u if history_u else {}
        self.history_i = history_i if history_i else {}
        self.history_ur = history_ur if history_ur else {}
        self.history_ir = history_ir if history_ir else {}
        
        # Attention networks for aggregating neighbors
        # User-item attention (for items rated by user)
        self.W1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W3 = nn.Linear(embed_dim, 1, bias=False)
        
        # Social attention (for friends)
        self.W4 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W5 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W6 = nn.Linear(embed_dim, 1, bias=False)
        
        # Item-item attention (for items rated by same users)
        self.W7 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W8 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W9 = nn.Linear(embed_dim, 1, bias=False)
        
        # Convert sparse matrices to tensors if provided
        if social_adj is not None:
            if isinstance(social_adj, csr_matrix):
                self.social_adj = self._sparse_to_tensor(social_adj).to(device)
            else:
                self.social_adj = torch.tensor(social_adj, dtype=torch.float32).to(device)
        else:
            self.social_adj = None
            
        if info_adj is not None:
            if isinstance(info_adj, csr_matrix):
                self.info_adj = self._sparse_to_tensor(info_adj).to(device)
            else:
                self.info_adj = torch.tensor(info_adj, dtype=torch.float32).to(device)
        else:
            self.info_adj = None
        
        # Final prediction layer
        self.fc1 = nn.Linear(embed_dim * 2, embed_dim)
        self.fc2 = nn.Linear(embed_dim, 1)
        
    def _sparse_to_tensor(self, sparse_mat):
        """Convert scipy sparse matrix to torch sparse tensor"""
        coo = sparse_mat.tocoo()
        # Convert to numpy array first, then to tensor for better performance
        indices = np.array([coo.row, coo.col])
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(coo.data)
        shape = coo.shape
        # Use torch.sparse_coo_tensor instead of deprecated torch.sparse.FloatTensor
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
    
    def attention_item(self, user_emb, item_embs, ratings):
        """
        Attention mechanism for aggregating item embeddings based on ratings
        Args:
            user_emb: User embedding [embed_dim]
            item_embs: Item embeddings [num_items, embed_dim]
            ratings: Rating values [num_items]
        Returns:
            Aggregated item embedding [embed_dim]
        """
        if len(item_embs) == 0:
            return torch.zeros_like(user_emb)
        
        item_embs = item_embs.to(self.device)
        ratings = ratings.to(self.device).unsqueeze(1)  # [num_items, 1]
        
        # Attention weights
        q = self.W1(user_emb.unsqueeze(0))  # [1, embed_dim]
        k = self.W2(item_embs)  # [num_items, embed_dim]
        
        # Combine with rating information
        attention_input = q + k + ratings * item_embs
        attention_weights = self.W3(attention_input).squeeze(1)  # [num_items]
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # Weighted aggregation
        aggregated = torch.sum(attention_weights.unsqueeze(1) * item_embs, dim=0)
        return aggregated
    
    def attention_social(self, user_emb, friend_embs):
        """
        Attention mechanism for aggregating social (friend) embeddings
        Args:
            user_emb: User embedding [embed_dim]
            friend_embs: Friend embeddings [num_friends, embed_dim]
        Returns:
            Aggregated social embedding [embed_dim]
        """
        if len(friend_embs) == 0:
            return torch.zeros_like(user_emb)
        
        friend_embs = friend_embs.to(self.device)
        
        # Attention weights
        q = self.W4(user_emb.unsqueeze(0))  # [1, embed_dim]
        k = self.W5(friend_embs)  # [num_friends, embed_dim]
        
        attention_input = q + k
        attention_weights = self.W6(attention_input).squeeze(1)  # [num_friends]
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # Weighted aggregation
        aggregated = torch.sum(attention_weights.unsqueeze(1) * friend_embs, dim=0)
        return aggregated
    
    def attention_itemitem(self, item_emb, neighbor_item_embs, neighbor_ratings):
        """
        Attention mechanism for item-item relationships
        Args:
            item_emb: Target item embedding [embed_dim]
            neighbor_item_embs: Neighbor item embeddings [num_neighbors, embed_dim]
            neighbor_ratings: Ratings from same users [num_neighbors]
        Returns:
            Aggregated item embedding [embed_dim]
        """
        if len(neighbor_item_embs) == 0:
            return torch.zeros_like(item_emb)
        
        neighbor_item_embs = neighbor_item_embs.to(self.device)
        neighbor_ratings = neighbor_ratings.to(self.device).unsqueeze(1)
        
        q = self.W7(item_emb.unsqueeze(0))
        k = self.W8(neighbor_item_embs)
        
        attention_input = q + k + neighbor_ratings * neighbor_item_embs
        attention_weights = self.W9(attention_input).squeeze(1)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        aggregated = torch.sum(attention_weights.unsqueeze(1) * neighbor_item_embs, dim=0)
        return aggregated
    
    def forward(self, user_ids, item_ids):
        """
        Forward pass
        Args:
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size]
        Returns:
            Predicted ratings [batch_size]
        """
        # Base embeddings
        user_emb = self.embed_user(user_ids)  # [batch_size, embed_dim]
        item_emb = self.embed_item(item_ids)  # [batch_size, embed_dim]
        
        # Aggregate user embeddings from items
        user_items_emb = []
        for i, u in enumerate(user_ids.cpu().numpy()):
            if u in self.history_u and len(self.history_u[u]) > 0:
                item_indices = torch.LongTensor(self.history_u[u]).to(self.device)
                item_embeddings = self.embed_item(item_indices)
                ratings = torch.FloatTensor(self.history_ur.get(u, [1.0] * len(self.history_u[u]))).to(self.device)
                aggregated = self.attention_item(user_emb[i], item_embeddings, ratings)
                user_items_emb.append(aggregated)
            else:
                user_items_emb.append(torch.zeros(self.embed_dim).to(self.device))
        user_items_emb = torch.stack(user_items_emb)  # [batch_size, embed_dim]
        
        # Aggregate social embeddings
        user_social_emb = []
        if self.social_adj is not None:
            for i, u in enumerate(user_ids.cpu().numpy()):
                # Get friends from adjacency matrix
                if isinstance(self.social_adj, torch.Tensor) and self.social_adj.is_sparse:
                    # For sparse tensor, extract row and get non-zero indices
                    row = self.social_adj[u].to_dense()
                    friend_indices = torch.nonzero(row > 0).squeeze(1).cpu().numpy()
                else:
                    friend_indices = torch.nonzero(self.social_adj[u] > 0).squeeze(1).cpu().numpy()
                
                # Handle scalar vs array
                if friend_indices.ndim == 0:
                    friend_indices = friend_indices.reshape(1)
                
                if len(friend_indices) > 0:
                    friend_embeddings = self.embed_user(torch.LongTensor(friend_indices).to(self.device))
                    aggregated = self.attention_social(user_emb[i], friend_embeddings)
                    user_social_emb.append(aggregated)
                else:
                    user_social_emb.append(torch.zeros(self.embed_dim).to(self.device))
        else:
            user_social_emb = [torch.zeros(self.embed_dim).to(self.device)] * len(user_ids)
        user_social_emb = torch.stack(user_social_emb).to(self.device)  # [batch_size, embed_dim]
        
        # Combine user embeddings
        user_emb_final = user_emb + user_items_emb + user_social_emb
        
        # Aggregate item embeddings from users
        item_users_emb = []
        for i, item_idx in enumerate(item_ids.cpu().numpy()):
            if item_idx in self.history_i and len(self.history_i[item_idx]) > 0:
                user_indices = torch.LongTensor(self.history_i[item_idx]).to(self.device)
                user_embeddings = self.embed_user(user_indices)
                ratings = torch.FloatTensor(self.history_ir.get(item_idx, [1.0] * len(self.history_i[item_idx]))).to(self.device)
                aggregated = self.attention_itemitem(item_emb[i], user_embeddings, ratings)
                item_users_emb.append(aggregated)
            else:
                item_users_emb.append(torch.zeros(self.embed_dim).to(self.device))
        item_users_emb = torch.stack(item_users_emb)  # [batch_size, embed_dim]
        
        # Combine item embeddings
        item_emb_final = item_emb + item_users_emb
        
        # Final prediction
        concat_emb = torch.cat([user_emb_final, item_emb_final], dim=1)  # [batch_size, 2*embed_dim]
        out = F.relu(self.fc1(concat_emb))
        out = self.fc2(out).squeeze(1)  # [batch_size]
        
        return out
    
    def loss(self, user_ids, item_ids, ratings):
        """
        Compute MSE loss
        Args:
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size]
            ratings: True ratings [batch_size]
        Returns:
            Loss value
        """
        pred = self.forward(user_ids, item_ids)
        loss = F.mse_loss(pred, ratings)
        return loss
    
    def predict_all(self, batch_size=1000):
        """
        Predict ratings for all user-item pairs (for evaluation)
        Uses batched computation for efficiency
        Returns:
            Prediction matrix [n_users, n_items]
        """
        self.eval()
        all_preds = torch.zeros(self.n_users, self.n_items).to(self.device)
        
        with torch.no_grad():
            # Process users in batches
            for u_start in range(0, self.n_users, batch_size):
                u_end = min(u_start + batch_size, self.n_users)
                user_ids = torch.arange(u_start, u_end, dtype=torch.long).to(self.device)
                
                # For each user batch, predict for all items in batches
                for i_start in range(0, self.n_items, batch_size):
                    i_end = min(i_start + batch_size, self.n_items)
                    item_ids = torch.arange(i_start, i_end, dtype=torch.long).to(self.device)
                    
                    # Create meshgrid for all user-item pairs in this batch
                    u_grid, i_grid = torch.meshgrid(user_ids, item_ids, indexing='ij')
                    u_flat = u_grid.flatten()
                    i_flat = i_grid.flatten()
                    
                    preds = self.forward(u_flat, i_flat)
                    all_preds[u_flat, i_flat] = preds
        
        return all_preds


def build_history_dicts(train_mat, user_id_map, item_id_map):
    """
    Build history dictionaries from training matrix for GraphRec model
    Returns:
        history_u: {user_id: [item_ids]}
        history_i: {item_id: [user_ids]}
        history_ur: {user_id: [ratings]}
        history_ir: {item_id: [ratings]}
    """
    history_u = {}
    history_i = {}
    history_ur = {}
    history_ir = {}
    
    train_coo = train_mat.tocoo()
    
    for u_idx, i_idx, rating in zip(train_coo.row, train_coo.col, train_coo.data):
        # User history
        if u_idx not in history_u:
            history_u[u_idx] = []
            history_ur[u_idx] = []
        history_u[u_idx].append(i_idx)
        history_ur[u_idx].append(float(rating))
        
        # Item history
        if i_idx not in history_i:
            history_i[i_idx] = []
            history_ir[i_idx] = []
        history_i[i_idx].append(u_idx)
        history_ir[i_idx].append(float(rating))
    
    return history_u, history_i, history_ur, history_ir


class TrainDataset(Dataset):
    """Dataset for training from sparse matrix"""
    def __init__(self, train_mat):
        train_coo = train_mat.tocoo()
        self.users = train_coo.row
        self.items = train_coo.col
        self.ratings = train_coo.data
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return (
            torch.LongTensor([self.users[idx]]).squeeze(),
            torch.LongTensor([self.items[idx]]).squeeze(),
            torch.FloatTensor([self.ratings[idx]]).squeeze()
        )


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (user_ids, item_ids, ratings) in enumerate(train_loader):
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.to(device)
        
        optimizer.zero_grad()
        loss = model.loss(user_ids, item_ids, ratings)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def compute_predictions_fn(model, n_user, n_item, device, **kwargs):
    """Compute predictions for all user-item pairs using model's predict_all method"""
    batch_size = kwargs.get('batch_size', 2000)  # Increase batch size for efficiency
    return model.predict_all(batch_size=batch_size)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GraphRec2: Social Recommendation with Sequential Split')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='../../rec_datasets', 
                       help='Path to rec_datasets directory')
    parser.add_argument('--dataset', type=str, default='ciao', 
                       help='Dataset name')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=64, 
                       help='Embedding dimension')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=2000, 
                       help='Batch size for testing/evaluation')
    parser.add_argument('--eval_every', type=int, default=5, 
                       help='Evaluate every N epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    
    # Evaluation arguments
    parser.add_argument('--topk_list', type=str, default='5,10,20', 
                       help='Top-K values for evaluation (comma-separated)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=2023, 
                       help='Random seed')
    parser.add_argument('--save_path', type=str, default='graphrec2_model.pth', 
                       help='Path to save model')
    
    args = parser.parse_args()
    
    # Parse topk_list
    args.topk_list = [int(k) for k in args.topk_list.split(',')]
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Embedding dimension: {args.embed_dim}")
    
    # Load data using common_utils (sequential split)
    logger.info("Loading data with sequential split...")
    # Resolve data path using common_utils
    resolved_data_path = resolve_data_path(args.data_path)
    logger.info(f"Resolved data path: {resolved_data_path}")
    train_mat, val_mat, test_mat, social_adj, info_adj, user_id_map, item_id_map, n_user, n_item = \
        load_data_using_utils(resolved_data_path, args.dataset, return_social_as_adj=True)
    
    logger.info(f"Users: {n_user}, Items: {n_item}")
    logger.info(f"Train interactions: {train_mat.nnz}")
    logger.info(f"Val interactions: {val_mat.nnz}")
    logger.info(f"Test interactions: {test_mat.nnz}")
    
    if social_adj is not None:
        logger.info(f"Social edges: {social_adj.nnz if hasattr(social_adj, 'nnz') else social_adj.sum()}")
    else:
        logger.info("No social network data found")
    
    # Build history dictionaries for model
    logger.info("Building history dictionaries...")
    history_u, history_i, history_ur, history_ir = build_history_dicts(
        train_mat, user_id_map, item_id_map
    )
    
    # Create model
    logger.info("Initializing GraphRec model...")
    model = GraphRec(
        n_users=n_user,
        n_items=n_item,
        embed_dim=args.embed_dim,
        social_adj=social_adj,
        info_adj=info_adj,
        history_u=history_u,
        history_i=history_i,
        history_ur=history_ur,
        history_ir=history_ir,
        batch_size=args.batch_size,
        device=device
    ).to(device)
    
    # Create training dataset and loader
    train_dataset = TrainDataset(train_mat)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    # Load sequences for sequential evaluation
    logger.info("Loading user sequences for sequential evaluation...")
    user_sequences = load_sequences_using_utils(resolved_data_path, args.dataset, user_id_map, item_id_map)
    
    # Create validation and test datasets (sequential)
    val_sequences = {u: seq for u, seq in user_sequences.items() if len(seq) >= 3}
    test_sequences = {u: seq for u, seq in user_sequences.items() if len(seq) >= 3}
    
    # Build train sequences (all items except last 2 for users with >=3 items)
    train_sequences = {}
    for u, seq in user_sequences.items():
        if len(seq) >= 3:
            train_sequences[u] = seq[:-2]  # Everything except last 2 (val and test)
    
    val_dataset = SequentialTestDataset(val_sequences, train_sequences, max_history=50)
    test_dataset = SequentialTestDataset(test_sequences, train_sequences, max_history=50)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=sequential_collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=sequential_collate_fn,
        num_workers=0
    )
    
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
    
    # Training loop
    best_val_metric = 0.0
    best_test_metric = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        logger.info(f"Epoch {epoch}/{args.epochs}, Train Loss: {train_loss:.4f}")
        
        # Validate (only every eval_every epochs or last epoch)
        if epoch % args.eval_every != 0 and epoch < args.epochs:
            continue
            
        logger.info("Evaluating on validation set...")
        val_metrics = evaluate_sequential(
            model, val_loader, n_user, n_item, 
            topk_list=args.topk_list, device=device,
            compute_predictions_fn=compute_predictions_fn,
            batch_size=args.test_batch_size
        )
        
        # Log validation metrics
        val_str = "Val - "
        for k in args.topk_list:
            hit = val_metrics[k]['hit']
            ndcg = val_metrics[k]['ndcg']
            val_str += f"H@{k}: {hit:.4f}, N@{k}: {ndcg:.4f}  "
        logger.info(val_str)
        
        # Test (for monitoring, but not for early stopping)
        logger.info("Evaluating on test set...")
        test_metrics = evaluate_sequential(
            model, test_loader, n_user, n_item,
            topk_list=args.topk_list, device=device,
            compute_predictions_fn=compute_predictions_fn,
            batch_size=args.test_batch_size
        )
        
        # Log test metrics
        test_str = "Test - "
        for k in args.topk_list:
            hit = test_metrics[k]['hit']
            ndcg = test_metrics[k]['ndcg']
            test_str += f"H@{k}: {hit:.4f}, N@{k}: {ndcg:.4f}  "
        logger.info(test_str)
        
        # Early stopping (based on validation Hit@10)
        val_hit10 = val_metrics[10]['hit']
        test_hit10 = test_metrics[10]['hit']
        
        if val_hit10 > best_val_metric:
            best_val_metric = val_hit10
            best_test_metric = test_hit10
            patience_counter = 0
            best_epoch = epoch
            # Save model
            torch.save(model.state_dict(), args.save_path)
            logger.info(f"New best validation Hit@10: {best_val_metric:.4f} (Epoch {best_epoch}) - Model saved to {args.save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model for final evaluation if it was saved
    if best_epoch > 0:
        logger.info(f"Loading best model from epoch {best_epoch} for final evaluation...")
        model.load_state_dict(torch.load(args.save_path, map_location=device))
    
    # Final evaluation
    logger.info("=" * 60)
    logger.info("Training completed!")
    if best_epoch > 0:
        logger.info(f"Best Validation Hit@10: {best_val_metric:.4f} (Epoch {best_epoch})")
        logger.info(f"Best Test Hit@10 (at best val): {best_test_metric:.4f}")
        logger.info(f"Best model saved to: {args.save_path}")
    logger.info("=" * 60)
    
    # Final test evaluation
    logger.info("Final test evaluation...")
    final_test_metrics = evaluate_sequential(
        model, test_loader, n_user, n_item,
        topk_list=args.topk_list, device=device,
        compute_predictions_fn=compute_predictions_fn,
        batch_size=args.test_batch_size
    )
    
    final_test_str = "Final Test - "
    for k in args.topk_list:
        hit = final_test_metrics[k]['hit']
        ndcg = final_test_metrics[k]['ndcg']
        final_test_str += f"H@{k}: {hit:.4f}, N@{k}: {ndcg:.4f}  "
    logger.info(final_test_str)


if __name__ == "__main__":
    main()
