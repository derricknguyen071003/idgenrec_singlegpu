import numpy as np
import os
import pickle
import argparse
import inspect
import logging
import sys
import random
import torch
from pathlib import Path
import wandb
def parse_global_args(parser):
    # Basic configuration
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--model_dir", type=str, default='../model', help='The model directory')
    parser.add_argument("--log_dir", type=str, default='../log', help='The log directory')
    parser.add_argument("--item_gpu", type=int, default=0, help='gpu id for item recommender (for multi-GPU mode)')
    parser.add_argument('--logging_level', type=int, default=logging.INFO,help='Logging Level, 0, 10, ..., 50')
    
    # Training configuration
    parser.add_argument("--id_epochs", type=int, default=10, help=("train id for certain num of epochs"))
    parser.add_argument("--rec_epochs", type=int, default=10, help=("train rec for certain num of epochs"))
    parser.add_argument("--id_batch_size", type=int, default=4, help="batch size for id generator")
    parser.add_argument("--social_batch_size", type=int, default=4, help="batch size for social model")
    parser.add_argument("--rec_batch_size", type=int, default=4, help="batch size for rec model")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of data loading workers for DataLoader (8 is more stable than 16)")
    parser.add_argument("--prefetch_factor", type=int, default=32, help="prefetch factor for DataLoader")
    parser.add_argument("--train", type=int, default=1, help='Train or not (1 for train, 0 for no train)')
    
    # Model paths
    parser.add_argument("--rec_model_path", type=str, help="path to rec model")
    parser.add_argument("--social_model_path", type=str, help="path to social model")
    
    # Learning rates
    parser.add_argument("--id_lr", type=float, default=1e-3, help="learning rate for recommendation model")
    parser.add_argument("--rec_lr", type=float, default=1e-5, help="learning rate for generation model")
    
    # Optimizer arguments
    parser.add_argument("--optim", type=str, default='AdamW', help='The name of the optimizer')
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--warmup_prop", type=float, default=0.05, help="Warmup proportion for scheduler")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    
    # Model architecture
    parser.add_argument("--backbone", type=str, default='t5-small', help='Default backbone model name')
    parser.add_argument("--metrics", type=str, default='hit@1,hit@5,hit@10,hit@20,ndcg@1,ndcg@5,ndcg@10,ndcg@20', help='Metrics for evaluation')
    
    # Run configuration
    parser.add_argument('--run_id', type=str, default='default', help='Unique identifier for this run to separate index files')
    parser.add_argument("--alt_style", type=str, default="id_first", help="choose from rec_first or id_first")
    parser.add_argument("--rounds", type=int, default=3, help="number of iterations")
    parser.add_argument("--use_wandb", type=int, default=1, help="use wandb")
    parser.add_argument("--use_friend_seq", type=int, default=0, help="use friend sequence")
    parser.add_argument("--random_remove_friend", type=float, default=0.0, help="randomly remove friend connections (0.0 to 1.0, percentage of connections to remove)")
    parser.add_argument("--phase", type=int, default=0, help="phase")
    parser.add_argument("--run_type", type=str, default="original_idgenrec", help="choose from original_idgenrec, social_to_rec, social_to_id, social_to_both, 1id2rec, 2id2rec, 2id1rec")
    
    # Dataset arguments
    parser.add_argument("--data_path", type=str, default='../rec_datasets', help="data directory")
    parser.add_argument("--item_indexing", type=str, default='sequential', help="item indexing method, including random, sequential, collaborative, and generative")
    parser.add_argument("--tasks", type=str, default='sequential,direct,straightforward', help="Downstream tasks, separate by comma")
    parser.add_argument("--datasets", type=str, default='Beauty', help="Dataset names, separate by comma")
    parser.add_argument("--prompt_file", type=str, default='../template/prompt.txt', help='the path of the prompt template file')
    parser.add_argument("--social_prompt_file", type=str, default='../template/prompt_social.txt', help='the path of the social prompt template file')
    parser.add_argument("--socialtoid_mode", type=str, default='sequential', choices=['sequential', 'flat'], help='Social data mode: sequential for friend recommendation, idgen for ID generation without target')
    
    # Sequential task arguments
    parser.add_argument("--max_his", type=int, default=-1, help='the max number of items in history sequence, -1 means no limit')
    parser.add_argument("--his_sep", type=str, default=' ; ', help='The separator used for history')
    
    # Prompt sampling arguments
    parser.add_argument("--test_prompt", type=str, default='seen:0', help='the prompt for testing')
    parser.add_argument("--sample_prompt", type=int, default=0, help='sample prompt or not')
    parser.add_argument("--social_sample_prompt", type=int, default=0, help='sample prompt for social data')
    parser.add_argument("--sample_num", type=str, default='2,2,2', help='the number of sampled data for each task')
    parser.add_argument("--social_quantization_id", type=int, default=0, help='use social quantization or not')
    
    # Discrete diffusion arguments
    parser.add_argument("--use_diffusion", type=int, default=0, help="Enable discrete diffusion (1) or not (0)")
    parser.add_argument("--diffusion_timesteps", type=int, default=100, help="Number of diffusion timesteps T")
    parser.add_argument("--diffusion_beta_max", type=float, default=0.1, help="Maximum corruption probability β_max")
    parser.add_argument("--diffusion_cross_prob", type=float, default=0.5, help="Probability of using cross-view noise vs random noise")
    parser.add_argument("--lambda_mask", type=float, default=0.1, help="Weight for noise mask prediction loss")
    parser.add_argument("--lambda_kl", type=float, default=0.1, help="Weight for KL divergence loss between views")
    parser.add_argument("--noise_head_dropout", type=float, default=0.1, help="Dropout probability for noise prediction head")
    
    # Early stopping arguments
    parser.add_argument("--early_stopping_patience", type=int, default=-1, help="Number of epochs to wait before early stopping (-1 to disable)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0, help="Minimum change in validation loss to qualify as improvement")
    parser.add_argument("--save_best_model", type=int, default=1, help="Save best model checkpoint based on validation loss (1 to enable, 0 to disable)")
    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def torch_isin(elements: torch.Tensor, test_elements: torch.Tensor) -> torch.Tensor:
        if hasattr(torch, 'isin'):
            return torch.isin(elements, test_elements)
        else:
            # Broadcasted comparison and reduction
            return (elements[..., None] == test_elements).any(-1)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)        
        
def remove_friend_connections_from_lines(lines, removal_percentage, seed=None):
    if seed is not None:
        random.seed(seed)
    if not 0.0 <= removal_percentage <= 1.0:
        raise ValueError("Removal percentage must be between 0.0 and 1.0")
    if removal_percentage == 0.0:
        return lines
    modified_lines = []
    total_connections = 0
    removed_connections = 0
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            modified_lines.append(line)
            continue   
        parts = line.split()
        if len(parts) < 2:  # User with no friends
            modified_lines.append(line)
            continue
        user_id = parts[0]
        friends = parts[1:]
        num_friends = len(friends)
        num_to_remove = int(num_friends * removal_percentage)   
        if num_to_remove > 0:
            friends_to_remove = random.sample(friends, num_to_remove)
            remaining_friends = [f for f in friends if f not in friends_to_remove]
            removed_connections += num_to_remove
        else:
            remaining_friends = friends
        total_connections += num_friends
        if remaining_friends:
            modified_lines.append(f"{user_id} {' '.join(remaining_friends)}")
        else:
            modified_lines.append(f"{user_id}")
    logging.info(f"Friend removal complete: {removed_connections}/{total_connections} connections removed ({removed_connections/total_connections*100:.2f}%)")
    return modified_lines

def ReadLineFromFile(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def WriteDictToFile(path, write_dict):
    with open(path, 'w') as out:
        for user, items in write_dict.items():
            if type(items) == list:
                out.write(user + ' ' + ' '.join(items) + '\n')
            else:
                out.write(user + ' ' + str(items) + '\n')

def get_init_paras_dict(class_name, paras_dict):
    base_list = inspect.getmro(class_name)
    paras_list = []
    for base in base_list:
        paras = inspect.getfullargspec(base.__init__)
        paras_list.extend(paras.args)
    paras_list = sorted(list(set(paras_list)))
    out_dict = {}
    for para in paras_list:
        if para == 'self':
            continue
        out_dict[para] = paras_dict[para]
    return out_dict

def setup_logging(args):
    # Example logging directory: ../log/train/Beauty/default.log
    dataset_str = getattr(args, 'datasets', 'UnknownDataset')
    if args.train:
        folder = os.path.join(args.log_dir, "train", dataset_str)
    else:
        folder = os.path.join(args.log_dir, "test", dataset_str)
    if not os.path.exists(folder):
        os.makedirs(folder)
    log_file = os.path.join(folder, args.run_id + '.log')
    if os.path.exists(log_file):
        os.remove(log_file)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=args.logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))   
    logging.info(f"Logging {args.run_id} to {log_file}")
    return

def setup_model_path(args):
    # Example model dir: idgenrec_singlegpu/model/lastfm4i/2id2rec_experiment
    args.model_path = os.path.join(args.model_dir, args.datasets, args.run_id)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    return
    
def save_model(model, path):
    torch.save(model.state_dict(), path)
    return
    
def load_model(model, path, args, loc=None):
    if loc is None and hasattr(args, 'gpu'):
        gpuid = args.gpu.split(',')
        loc = f'cuda:{gpuid[0]}'
    state_dict = torch.load(path, map_location=loc)
    model.load_state_dict(state_dict, strict=False)
    return model

def setup_wandb(args):
    if args.use_wandb:
        # Get the absolute path to the project root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent  # src/utils/utils.py -> project root
        
        wandb.init(
            entity="trungnguyen0710vn-singapore-management-university",
            project="idgenrec-social-enhanced",
            name=args.run_id,
        )
        
        # Save all Python files in the codebase
        wandb.run.log_code(
            root=str(project_root),
            include_fn=lambda path: path.endswith('.py')
        )
    return

def insert_phrases_batch(prompt, positions, hist, max_input_len):
    """
    prompt: [batch_size, seq_len, emb_size] - embedding of the template sentence
    hist: [batch_size, phrase_num, 10, emb_size] - embeddings of the hist
    positions: [batch_size, seq_len] - binary tensor where "1" indicates insertion points
    max_input_len: int - the maximum length after processing
    """
    batch_size, seq_len, emb_size = prompt.shape
    
    batch_results = []
    
    # Iterate through each example in the batch
    for b in range(batch_size):
        result = []
        hist_idx = 0

        for i in range(seq_len):
            if positions[b, i] == 1:
                result.append(prompt[b, i].unsqueeze(0))  
                result.append(hist[b, hist_idx])
                hist_idx += 1
            else:
                result.append(prompt[b, i].unsqueeze(0))

        result_tensor = torch.cat(result, dim=0)
        
        # Pad the tensor to max_input_len
        pad_size = max_input_len - result_tensor.shape[0]
        pad_tensor = torch.zeros((pad_size, emb_size), device=prompt.device)
        result_tensor = torch.cat([result_tensor, pad_tensor], dim=0)
        
        batch_results.append(result_tensor)

    # Concatenate batch_results to get final tensor
    final_tensor = torch.stack(batch_results, dim=0)
    
    return final_tensor


