"""
Common utilities for baseline recommendation models (TrustSVD, DiffNet2, etc.)

This module contains shared functions and classes used across different baseline
implementations to avoid code duplication.
"""

import sys
import os
import logging

# Add src directory to path (go up 2 levels from baseline/ to project root, then into src)
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(script_dir, '../../src'))
sys.path.insert(0, src_path)

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix

# Import existing utilities
from utils import utils
from utils import indexing
from utils import evaluate as eval_utils


def setup_logging():
    """Setup logging configuration for baseline models"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)  # Also output to stdout for terminal
        ]
    )


def resolve_data_path(data_path):
    """Resolve relative data path to absolute path"""
    if not os.path.isabs(data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if data_path == '../../rec_datasets' or data_path == '../../../rec_datasets':
            # From command/baseline/, go up to project root, then to rec_datasets
            data_path = '../../../rec_datasets'
        data_path = os.path.abspath(os.path.join(script_dir, data_path))
        # If still not found, try alternative: go from project root
        if not os.path.exists(data_path) and 'idgenrec_singlegpu' in script_dir:
            # Try to find rec_datasets relative to idgenrec_singlegpu root
            project_root = script_dir
            while not os.path.exists(os.path.join(project_root, 'rec_datasets')) and project_root != '/':
                project_root = os.path.dirname(project_root)
            if os.path.exists(os.path.join(project_root, 'rec_datasets')):
                data_path = os.path.join(project_root, 'rec_datasets')
    return data_path


def parse_id(id_str):
    """Parse string ID (e.g., 'u2' -> 2, 'i52' -> 52)"""
    if id_str.startswith('u'):
        return int(id_str[1:])
    elif id_str.startswith('i'):
        return int(id_str[1:])
    else:
        return int(id_str)


def split_user_sequence(items):
    """
    Split user sequence following MultiTaskDataset_rec pattern:
    - 1 item: skip (no history)
    - 2 items: first = train, second = validation
    - >=3 items: rest = train, second to last = validation, last = test
    Returns: (train_items, val_items, test_items)
    """
    if len(items) <= 1:
        return [], [], []
    elif len(items) == 2:
        return items[:-1], items[-1:], []
    else:
        return items[:-2], items[-2:-1], items[-1:]


def load_data_using_utils(data_path, dataset_name, return_social_as_adj=False):
    """
    Load data using existing idgenrec utilities.
    Split per user: last item = test, second to last = validation, rest = training
    
    Args:
        data_path: Path to rec_datasets directory
        dataset_name: Name of the dataset
        return_social_as_adj: If True, returns social_adj and info_adj (for DiffNet2).
                            If False, returns trust_mat (for TrustSVD).
    
    Returns:
        If return_social_as_adj=True:
            train_mat, val_mat, test_mat, social_adj, info_adj, user_id_map, item_id_map, n_user, n_item
        If return_social_as_adj=False:
            train_mat, val_mat, test_mat, trust_mat, user_id_map, item_id_map, n_user, n_item
    """
    data_path = resolve_data_path(data_path)
    
    # Load user sequences using existing utility
    user_sequence_file = os.path.join(data_path, dataset_name, 'user_sequence.txt')
    user_sequence = utils.ReadLineFromFile(user_sequence_file)
    user_sequence_dict = indexing.construct_user_sequence_dict(user_sequence)
    
    # Parse and collect all users and items
    all_users = set()
    all_items = set()
    user_items_dict = {}
    
    for user_id_str, items_str in user_sequence_dict.items():
        user_id = parse_id(user_id_str)
        all_users.add(user_id)
        items = [parse_id(item_str) for item_str in items_str]
        user_items_dict[user_id] = items
        for item_id in items:
            all_items.add(item_id)
    
    # Create ID mappings (consecutive 0-indexed)
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_users))}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_items))}
    
    n_user = len(all_users)
    n_item = len(all_items)
    
    # Create train/val/test split using same logic as MultiTaskDataset_rec
    train_rows, train_cols, train_data = [], [], []
    val_rows, val_cols, val_data = [], [], []
    test_rows, test_cols, test_data = [], [], []
    
    for old_user_id, items in user_items_dict.items():
        train_items, val_items, test_items = split_user_sequence(items)
        if not train_items and not val_items and not test_items:
            continue  # Skip users with <= 1 item
        
        new_user_id = user_id_map[old_user_id]
        
        for old_item_id in train_items:
            train_rows.append(new_user_id)
            train_cols.append(item_id_map[old_item_id])
            train_data.append(1.0)
        
        for old_item_id in val_items:
            val_rows.append(new_user_id)
            val_cols.append(item_id_map[old_item_id])
            val_data.append(1.0)
        
        for old_item_id in test_items:
            test_rows.append(new_user_id)
            test_cols.append(item_id_map[old_item_id])
            test_data.append(1.0)
    
    train_mat = csr_matrix((train_data, (train_rows, train_cols)), shape=(n_user, n_item)) if train_data else csr_matrix((n_user, n_item))
    val_mat = csr_matrix((val_data, (val_rows, val_cols)), shape=(n_user, n_item)) if val_data else csr_matrix((n_user, n_item))
    test_mat = csr_matrix((test_data, (test_rows, test_cols)), shape=(n_user, n_item)) if test_data else csr_matrix((n_user, n_item))
    
    # Load social network (friend connections) using existing utility
    friend_sequence_file = os.path.join(data_path, dataset_name, 'friend_sequence.txt')
    
    if return_social_as_adj:
        # For DiffNet2: return social_adj and info_adj
        info_adj = train_mat
        
        social_adj = None
        if os.path.exists(friend_sequence_file):
            friend_sequence = utils.ReadLineFromFile(friend_sequence_file)
            friend_sequence_dict = indexing.construct_user_sequence_dict(friend_sequence)
            
            rows, cols, data = [], [], []
            for old_user_id, friends in friend_sequence_dict.items():
                if old_user_id in user_id_map:
                    new_user_id = user_id_map[old_user_id]
                    for old_friend_id in friends:
                        old_friend_id_parsed = parse_id(old_friend_id)
                        if old_friend_id_parsed in user_id_map:
                            new_friend_id = user_id_map[old_friend_id_parsed]
                            rows.append(new_user_id)
                            cols.append(new_friend_id)
                            data.append(1.0)
            
            if data:
                social_adj = csr_matrix((data, (rows, cols)), shape=(n_user, n_user))
                # Make symmetric (undirected social graph)
                social_adj = social_adj + social_adj.T
                social_adj.data = np.ones_like(social_adj.data)  # Binary
        
        return train_mat, val_mat, test_mat, social_adj, info_adj, user_id_map, item_id_map, n_user, n_item
    else:
        # For TrustSVD: return trust_mat
        trust_mat = None
        if os.path.exists(friend_sequence_file):
            friend_sequence = utils.ReadLineFromFile(friend_sequence_file)
            friend_sequence_dict = indexing.construct_user_sequence_dict(friend_sequence)
            
            rows, cols, data = [], [], []
            for old_user_id, friends in friend_sequence_dict.items():
                if old_user_id in user_id_map:
                    new_user_id = user_id_map[old_user_id]
                    for old_friend_id in friends:
                        old_friend_id_parsed = parse_id(old_friend_id)
                        if old_friend_id_parsed in user_id_map:
                            new_friend_id = user_id_map[old_friend_id_parsed]
                            rows.append(new_user_id)
                            cols.append(new_friend_id)
                            data.append(1.0)
            
            if data:
                trust_mat = csr_matrix((data, (rows, cols)), shape=(n_user, n_user))
                if trust_mat.max() > 1.0:
                    trust_mat = trust_mat / trust_mat.max()
        
        return train_mat, val_mat, test_mat, trust_mat, user_id_map, item_id_map, n_user, n_item


def load_sequences_using_utils(data_path, dataset_name, user_id_map, item_id_map):
    """Load and remap sequences using existing utilities"""
    data_path = resolve_data_path(data_path)
    user_sequence_file = os.path.join(data_path, dataset_name, 'user_sequence.txt')
    if not os.path.exists(user_sequence_file):
        return {}
    
    user_sequence = utils.ReadLineFromFile(user_sequence_file)
    user_sequence_dict = indexing.construct_user_sequence_dict(user_sequence)
    
    # Parse and remap IDs to match model's internal IDs
    sequences = {}
    for user_id_str, items_str in user_sequence_dict.items():
        old_user_id = parse_id(user_id_str)
        if old_user_id in user_id_map:
            new_user_id = user_id_map[old_user_id]
            remapped_items = [item_id_map[parse_id(item_str)] for item_str in items_str 
                            if parse_id(item_str) in item_id_map]
            if remapped_items:
                sequences[new_user_id] = remapped_items
    
    return sequences


class TestDataset(Dataset):
    """Dataset for evaluation"""
    def __init__(self, test_mat, train_mat):
        self.test_mat = test_mat.tocoo()
        self.train_mat = train_mat.tocoo()
        self.users = list(set(self.test_mat.row))
        # Precompute train items per user for efficiency
        self.train_items_per_user = {}
        for u, i in zip(self.train_mat.row, self.train_mat.col):
            if u not in self.train_items_per_user:
                self.train_items_per_user[u] = set()
            self.train_items_per_user[u].add(i)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        train_items = list(self.train_items_per_user.get(u, set()))
        # Get test items for this user
        test_mask = self.test_mat.row == u
        test_items = list(set(self.test_mat.col[test_mask]))
        return u, train_items, test_items


class SequentialTestDataset(Dataset):
    """Dataset for sequential evaluation - only uses test items (last item per user)"""
    def __init__(self, user_sequences, train_sequences, max_history=20):
        """
        Args:
            user_sequences: dict mapping user_id -> list of items (full sequence)
            train_sequences: dict mapping user_id -> list of items (training sequence)
            max_history: maximum history length to use
        """
        self.samples = []
        self.max_history = max_history
        
        for user_id in user_sequences:
            full_seq = user_sequences[user_id]
            # Only create test sample for users with >=3 items (has test item)
            if len(full_seq) <= 2:
                continue
            
            # Test sample: history is all items except last, target is last item
            history = full_seq[:-1]
            if len(history) > max_history:
                history = history[-max_history:]
            
            self.samples.append({
                'user_id': user_id,
                'history': history,
                'target': full_seq[-1]
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def test_collate_fn(batch):
    """Custom collate function for test dataset"""
    users = [item[0] for item in batch]
    train_items = [item[1] for item in batch]
    test_items = [item[2] for item in batch]
    return users, train_items, test_items


def sequential_collate_fn(batch):
    """Custom collate function for sequential dataset"""
    user_ids = [item['user_id'] for item in batch]
    histories = [item['history'] for item in batch]
    targets = [item['target'] for item in batch]
    return user_ids, histories, targets


def _evaluate_user(all_preds, u, train_items, test_items, max_topk):
    """Helper function to evaluate a single user using existing evaluation utilities"""
    # Get predictions for this user
    scores = all_preds[u].cpu().numpy()
    
    # Mask training items
    for item in train_items:
        scores[item] = -1e8
    
    # Get top items
    top_items = np.argsort(scores)[-max_topk:][::-1]
    
    # Create relevance list for each test item (1 if in top-k, 0 otherwise)
    # Format: list of 0s and 1s where 1 indicates relevant item found at that position
    relevance = []
    test_set = set(test_items)
    for item in top_items:
        relevance.append(1 if item in test_set else 0)
    
    return relevance


def evaluate(model, test_loader, n_user, n_item, topk_list=[5, 10, 20], device='cpu', 
             compute_predictions_fn=None, **compute_kwargs):
    """
    Evaluate model using Hit Rate@K and NDCG@K - reuses existing evaluation utilities
    
    Args:
        model: The recommendation model
        test_loader: DataLoader for test data
        n_user: Number of users
        n_item: Number of items
        topk_list: List of K values for evaluation
        device: Device to run evaluation on
        compute_predictions_fn: Optional function to compute all predictions.
                               If None, uses model.predict_all()
        **compute_kwargs: Additional kwargs to pass to compute_predictions_fn
    """
    model.eval()
    
    max_topk = max(topk_list)
    all_relevance = []
    
    with torch.no_grad():
        if compute_predictions_fn is not None:
            all_preds = compute_predictions_fn(model, n_user, n_item, device, **compute_kwargs)
        else:
            all_preds = model.predict_all()
        
        for batch in test_loader:
            users, train_items_list, test_items_list = batch
            u = users[0] if isinstance(users, list) else users
            u = u.item() if hasattr(u, 'item') else int(u)
            train_items = train_items_list[0] if isinstance(train_items_list, list) else train_items_list
            test_items = test_items_list[0] if isinstance(test_items_list, list) else test_items_list
            relevance = _evaluate_user(all_preds, u, train_items, test_items, max_topk)
            all_relevance.append(relevance)
    
    avg_metrics = {}
    for k in topk_list:
        hit = eval_utils.hit_at_k(all_relevance, k) / len(all_relevance)
        ndcg = eval_utils.ndcg_at_k(all_relevance, k) / len(all_relevance)
        avg_metrics[k] = {'hit': hit, 'ndcg': ndcg}
    
    return avg_metrics


def evaluate_sequential(model, test_loader, n_user, n_item, topk_list=[5, 10], device='cpu',
                        compute_predictions_fn=None, **compute_kwargs):
    """
    Evaluate model using sequential setting - reuses existing evaluation utilities
    
    Args:
        model: The recommendation model
        test_loader: DataLoader for sequential test data
        n_user: Number of users
        n_item: Number of items
        topk_list: List of K values for evaluation
        device: Device to run evaluation on
        compute_predictions_fn: Optional function to compute all predictions.
                               If None, uses model.predict_all()
        **compute_kwargs: Additional kwargs to pass to compute_predictions_fn
    """
    model.eval()
    
    max_topk = max(topk_list)
    all_relevance = []
    
    with torch.no_grad():
        if compute_predictions_fn is not None:
            all_preds = compute_predictions_fn(model, n_user, n_item, device, **compute_kwargs)
        else:
            all_preds = model.predict_all()
        
        for batch in test_loader:
            user_ids, histories, targets = batch
            
            for u, history, target in zip(user_ids, histories, targets):
                u = u if isinstance(u, int) else int(u)
                target = target if isinstance(target, int) else int(target)
                
                # Get predictions for this user
                scores = all_preds[u].cpu().numpy()
                
                # Mask items in history (can't recommend items already in history)
                for item in history:
                    scores[item] = -1e8
                
                # Get top-K items
                top_items = np.argsort(scores)[-max_topk:][::-1]
                
                # Create relevance list: 1 if target found at position, 0 otherwise
                relevance = [1 if item == target else 0 for item in top_items]
                all_relevance.append(relevance)
    
    # Use existing evaluation utilities
    avg_metrics = {}
    for k in topk_list:
        hit = eval_utils.hit_at_k(all_relevance, k) / len(all_relevance)
        ndcg = eval_utils.ndcg_at_k(all_relevance, k) / len(all_relevance)
        avg_metrics[k] = {'hit': hit, 'ndcg': ndcg}
    
    return avg_metrics


def compute_all_predictions_batched(model, n_user, n_item, device, batch_size=1000, 
                                    forward_fn=None, **forward_kwargs):
    """
    Compute predictions for all user-item pairs in batches.
    This is needed for models that don't have an efficient predict_all() method.
    
    Args:
        model: The recommendation model
        n_user: Number of users
        n_item: Number of items
        device: Device to run on
        batch_size: Batch size for computation
        forward_fn: Optional function to call model forward. If None, uses model(user_ids, item_ids, **forward_kwargs)
        **forward_kwargs: Additional kwargs to pass to forward function
    """
    model.eval()
    all_preds = torch.zeros(n_user, n_item, device=device)
    
    with torch.no_grad():
        # Process users in batches
        for u_start in range(0, n_user, batch_size):
            u_end = min(u_start + batch_size, n_user)
            user_ids = torch.arange(u_start, u_end, device=device)
            
            # For each user, predict for all items
            for i_start in range(0, n_item, batch_size):
                i_end = min(i_start + batch_size, n_item)
                item_ids = torch.arange(i_start, i_end, device=device)
                
                # Create meshgrid for all user-item pairs in this batch
                u_grid, i_grid = torch.meshgrid(user_ids, item_ids, indexing='ij')
                u_flat = u_grid.flatten()
                i_flat = i_grid.flatten()
                
                if forward_fn is not None:
                    preds = forward_fn(model, u_flat, i_flat, **forward_kwargs)
                else:
                    preds = model(u_flat, i_flat, **forward_kwargs)
                
                all_preds[u_flat, i_flat] = preds
    
    return all_preds
