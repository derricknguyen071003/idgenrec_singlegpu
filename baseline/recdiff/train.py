import sys
import os

# Add baseline directory to path to import common_utils
script_dir = os.path.dirname(os.path.abspath(__file__))
baseline_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, baseline_path)

# Import common utilities
from common_utils import (
    setup_logging, resolve_data_path, parse_id, split_user_sequence,
    load_data_using_utils as _load_data_using_utils,
    TestDataset, test_collate_fn, evaluate as _evaluate
)

# Setup logging
setup_logging()
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Import RecDiff2 models
from model import GCNModel, SDNet, DiffusionProcess


class RecDiff2Dataset(Dataset):
    """Dataset for RecDiff2 training with BPR sampling"""
    def __init__(self, train_mat):
        coo = train_mat.tocoo()
        self.rows = coo.row
        self.cols = coo.col
        self.n_items = train_mat.shape[1]
        # Precompute positive items per user for efficient negative sampling
        self.pos_items_per_user = {}
        for u, i in zip(self.rows, self.cols):
            if u not in self.pos_items_per_user:
                self.pos_items_per_user[u] = set()
            self.pos_items_per_user[u].add(i)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        u, i = self.rows[idx], self.cols[idx]
        # Sample negative item
        j = np.random.randint(self.n_items)
        # Ensure negative item is not positive
        pos_items = self.pos_items_per_user.get(u, set())
        while j in pos_items:
            j = np.random.randint(self.n_items)
        return u, i, j


# TestDataset and test_collate_fn are imported from common_utils


def sparse_to_torch(sparse_mat, device):
    """Convert scipy sparse matrix to torch sparse tensor"""
    coo = sparse_mat.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
    values = torch.from_numpy(coo.data).float()
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def make_ui_adj(train_mat, n_user, n_item, device):
    """Create normalized user-item bipartite adjacency matrix"""
    a = csr_matrix((n_user, n_user))
    b = csr_matrix((n_item, n_item))
    mat = csr_matrix(np.vstack([np.hstack([a.A, train_mat.A]), np.hstack([train_mat.A.T, b.A])]))
    mat = (mat != 0).astype(float)
    
    # Normalize
    rowsum = np.array(mat.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = csr_matrix((d_inv_sqrt, (np.arange(len(d_inv_sqrt)), np.arange(len(d_inv_sqrt)))))
    mat = d_mat_inv_sqrt.dot(mat).dot(d_mat_inv_sqrt)
    
    return sparse_to_torch(mat, device)


def make_social_adj(social_mat, device):
    """Create normalized social adjacency matrix"""
    mat = (social_mat != 0).astype(float)
    
    # Normalize
    rowsum = np.array(mat.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = csr_matrix((d_inv_sqrt, (np.arange(len(d_inv_sqrt)), np.arange(len(d_inv_sqrt)))))
    mat = d_mat_inv_sqrt.dot(mat).dot(d_mat_inv_sqrt)
    
    return sparse_to_torch(mat, device)


def evaluate(model, diff_process, test_loader, ui_adj, social_adj, n_user, n_item, topk_list=[5, 10, 20], device='cpu'):
    """Evaluate RecDiff model using Hit Rate@K and NDCG@K - reuses common utilities"""
    model.eval()
    diff_process.eval()
    
    def compute_predictions_fn(model, n_user, n_item, device, **kwargs):
        """Compute predictions for RecDiff model"""
        ui_adj = kwargs['ui_adj']
        social_adj = kwargs['social_adj']
        diff_process = kwargs['diff_process']
        
        # Get embeddings for all users and items
        ui_emb, social_emb = model(ui_adj, social_adj)
        u_emb = ui_emb[:n_user]
        i_emb = ui_emb[n_user:]
        
        # Denoise social embeddings for all users
        denoised_social = diff_process.sample(model.sdnet, social_emb, steps=0)
        # Combine user and denoised social embeddings
        combined_u_emb = u_emb + denoised_social
        
        # Compute all predictions: R = U * I^T
        all_preds = torch.mm(combined_u_emb, i_emb.t())
        return all_preds
    
    return _evaluate(model, test_loader, n_user, n_item, topk_list, device,
                    compute_predictions_fn=compute_predictions_fn,
                    ui_adj=ui_adj, social_adj=social_adj, diff_process=diff_process)


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
    
    # Create adjacency matrices
    logging.info("Creating adjacency matrices...")
    ui_adj = make_ui_adj(train_mat, n_user, n_item, device)
    social_adj = make_social_adj(trust_mat if trust_mat is not None else csr_matrix((n_user, n_user)), device)
    
    # Create models
    logging.info("Initializing models...")
    gcn_model = GCNModel(n_user, n_item, args.n_hid, args.n_layers, args.s_layers).to(device)
    sdnet = SDNet(args.n_hid, args.emb_size).to(device)
    gcn_model.sdnet = sdnet  # Attach for easy access
    diff_process = DiffusionProcess(args.steps, args.noise_scale, args.noise_min, args.noise_max, device).to(device)
    
    # Optimizers
    opt_gcn = torch.optim.Adam(gcn_model.parameters(), lr=args.lr)
    opt_diff = torch.optim.Adam(sdnet.parameters(), lr=args.difflr)
    
    # Data loaders
    train_loader = DataLoader(
        RecDiff2Dataset(train_mat),
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
    
    # Test loader (for final evaluation)
    test_loader = DataLoader(
        TestDataset(test_mat, train_mat),
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
        gcn_model.train()
        sdnet.train()
        diff_process.train()
        total_loss = 0
        loss_components = {'bpr': 0, 'diff': 0, 'reg': 0}
        
        for u, i, j in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            u, i, j = u.to(device), i.to(device), j.to(device)
            
            # Get embeddings
            ui_emb, social_emb = gcn_model(ui_adj, social_adj)
            u_emb = ui_emb[:n_user]
            i_emb = ui_emb[n_user:]
            
            # Diffusion loss
            u_social = social_emb[u]
            diff_loss, denoised = diff_process.compute_loss(sdnet, u_social, args.reweight)
            
            # Combine embeddings
            u_vec = u_emb[u] + denoised
            
            # BPR loss
            pos_scores = (u_vec * i_emb[i]).sum(dim=1)
            neg_scores = (u_vec * i_emb[j]).sum(dim=1)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            
            # Regularization
            reg_loss = args.reg * (u_vec.norm(2).pow(2) + i_emb[i].norm(2).pow(2) + i_emb[j].norm(2).pow(2))
            
            loss = bpr_loss + diff_loss + reg_loss
            total_loss += loss.item()
            loss_components['bpr'] += bpr_loss.item()
            loss_components['diff'] += diff_loss.item()
            loss_components['reg'] += reg_loss.item()
            
            opt_gcn.zero_grad()
            opt_diff.zero_grad()
            loss.backward()
            opt_gcn.step()
            opt_diff.step()
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, '
                     f'BPR={loss_components["bpr"]/len(train_loader):.4f}, '
                     f'Diff={loss_components["diff"]/len(train_loader):.4f}, '
                     f'Reg={loss_components["reg"]/len(train_loader):.4f}')
        
        # Evaluate on validation set
        if (epoch + 1) % args.eval_every == 0:
            topk_list = [5, 10, args.topk] if args.topk not in [5, 10] else [5, 10]
            metrics = evaluate(gcn_model, diff_process, val_loader, ui_adj, social_adj, n_user, n_item, topk_list, device)
            
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
                torch.save({
                    'gcn': gcn_model.state_dict(),
                    'sdnet': sdnet.state_dict()
                }, args.save_path)
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
        checkpoint = torch.load(args.save_path, map_location=device)
        gcn_model.load_state_dict(checkpoint['gcn'])
        sdnet.load_state_dict(checkpoint['sdnet'])
    
    logging.info(f'Best Validation HR@10: {best_hr_10:.4f} (Epoch {best_epoch})')
    
    # Final evaluation on test set
    logging.info(f'\n=== Final Evaluation (Test Set) ===')
    topk_list = [5, 10, args.topk] if args.topk not in [5, 10] else [5, 10]
    test_metrics = evaluate(gcn_model, diff_process, test_loader, ui_adj, social_adj, n_user, n_item, topk_list, device)
    
    logging.info("\nTest Set Results:")
    for k in sorted(topk_list):
        logging.info(f"  HR@{k}={test_metrics[k]['hit']:.4f}, NDCG@{k}={test_metrics[k]['ndcg']:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train RecDiff2 using existing codebase utilities')
    parser.add_argument('--data_path', type=str, default='../../rec_datasets', help='Path to rec_datasets directory')
    parser.add_argument('--datasets', type=str, default='lastfm', help='Dataset name')
    parser.add_argument('--cuda', type=int, default=-1, help='CUDA device ID (-1 for CPU, 0+ for GPU)')
    parser.add_argument('--n_hid', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GCN layers for user-item graph')
    parser.add_argument('--s_layers', type=int, default=2, help='Number of GCN layers for social graph')
    parser.add_argument('--emb_size', type=int, default=16, help='Time embedding size for diffusion')
    parser.add_argument('--steps', type=int, default=20, help='Number of diffusion steps')
    parser.add_argument('--noise_scale', type=float, default=0.1, help='Noise scale for diffusion')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='Minimum noise level')
    parser.add_argument('--noise_max', type=float, default=0.01, help='Maximum noise level')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for GCN')
    parser.add_argument('--difflr', type=float, default=0.001, help='Learning rate for diffusion network')
    parser.add_argument('--reg', type=float, default=0.0001, help='Regularization weight')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--n_epoch', type=int, default=150, help='Number of epochs')
    parser.add_argument('--topk', type=int, default=20, help='Top-K for evaluation')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save_path', type=str, default='recdiff2_model.pth', help='Path to save model')
    parser.add_argument('--reweight', action='store_true', help='Use reweighted diffusion loss')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (number of evaluations without improvement)')
    args = parser.parse_args()
    
    train(args)
