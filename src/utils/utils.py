import numpy as np
import os
import pickle
import argparse
import inspect
import logging
import sys
import random
import torch


def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--model_dir", type=str, default='../model', help='The model directory')
    parser.add_argument("--checkpoint_dir", type=str, default='../checkpoint', help='The checkpoint directory')
    parser.add_argument("--model_name", type=str, default='model.pt', help='The model name')
    parser.add_argument("--log_dir", type=str, default='../log', help='The log directory')
    parser.add_argument("--distributed", type=int, default=1, help='use distributed data parallel or not.')
    parser.add_argument("--gpu", type=str, default='0,1,2,3', help='gpu ids, if not distributed, only use the first one.')
    parser.add_argument("--master_addr", type=str, default='localhost', help='Setup MASTER_ADDR for os.environ')
    parser.add_argument("--master_port", type=str, default='12345', help='Setup MASTER_PORT for os.environ')
    parser.add_argument('--logging_level', type=int, default=logging.INFO,help='Logging Level, 0, 10, ..., 50')
    # parser.add_argument("--training_strategy", type=int, default=0, help=(
    #         "only used for generative ID. 0: train ID generator only, "
    #         "1: train recommendation model only. 2: start with training ID generator. "
    #         "3: start with training recommendation model"
    #     )
    # )
    parser.add_argument("--id_epochs", type=int, default=10, help=("train id for certain num of epochs"))
    parser.add_argument("--rec_epochs", type=int, default=10, help=("train rec for certain num of epochs"))
    parser.add_argument("--id_batch_size", type=int, default=4, help="batch size for id generator")
    parser.add_argument("--rec_batch_size", type=int, default=64, help="batch size for rec model")
    parser.add_argument("--rec_model_path", type=str, help="path to rec model")
    parser.add_argument("--id_model_path", type=str, help="path to id model")
    parser.add_argument("--id_lr", type=float, default=1e-3, help="learning rate for recommendation model")
    parser.add_argument("--rec_lr", type=float, default=1e-5, help="learning rate for generation model")
    parser.add_argument("--alt_style", type=str, default="id_first", help="choose from rec_first or id_first")
    parser.add_argument("--test_epoch_id", type=int, default=1)
    parser.add_argument("--test_epoch_rec", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3, help="number of iterations")
    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def ReadLineFromFile(path):
    if not os.path.exists(path):
        raise FileNotFoundError
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
    args.log_name = log_name(args)
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
    folder = os.path.join(args.log_dir, folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    log_file = os.path.join(args.log_dir, folder_name, args.log_name + '.log')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=args.logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    return

# In src/utils/utils.py
# Ensure 'os' is imported at the top of your utils.py: import os

def log_name(args):
    """
    Creates a descriptive log filename based on experiment arguments.
    Prioritizes alternating training parameters if an alt_style is specified.
    """
    # Determine base name part from dataset(s)
    # Use getattr for safety, in case args.datasets is not defined by any active parser
    # (though parse_global_args or dataset_args should define it)
    dataset_str = getattr(args, 'datasets', 'UnknownDataset') 
    if isinstance(dataset_str, str) and len(dataset_str.split(',')) > 1:
        log_name_base = 'MultiDs' # Indicator for multiple datasets
    else:
        log_name_base = str(dataset_str).replace('/', '_') # Sanitize if it's path-like

    params = [log_name_base]

    # Helper to add param to log name if it exists and is not a common default/empty value
    def add_param_if_relevant(attr_name, display_prefix="", default_to_skip=None, is_path=False):
        if hasattr(args, attr_name):
            val = getattr(args, attr_name)
            
            if val is None or (isinstance(val, str) and not val.strip()): # Skip None or empty strings
                return
            if default_to_skip is not None and val == default_to_skip: # Skip if it's a common default
                return

            prefix = display_prefix or attr_name.replace('_', '') # Use display_prefix or a shortened attr_name
            
            current_val_str = str(val)
            if is_path:
                current_val_str = os.path.splitext(os.path.basename(current_val_str))[0] # Filename without ext
            
            params.append(f"{prefix}{current_val_str}")

    # Add parameters relevant to the experiment
    current_alt_style = getattr(args, 'alt_style', 'none') # Default to 'none' if not set
    add_param_if_relevant('alt_style', 'as', default_to_skip='none') 
    
    if current_alt_style and str(current_alt_style).lower() not in ['none', '']:
        # Alternating training parameters
        add_param_if_relevant('rounds', 'r')
        add_param_if_relevant('id_epochs', 'ide')
        add_param_if_relevant('rec_epochs', 'rece')
        add_param_if_relevant('id_lr', 'idlr')
        add_param_if_relevant('rec_lr', 'reclr')
        if hasattr(args, 'id_batch_size'): params.append(f"idbs{args.id_batch_size}")
        if hasattr(args, 'rec_batch_size'): params.append(f"recbs{args.rec_batch_size}")
    else: 
        # Fallback to generic epochs, lr, batch_size if not using a recognized alternating style
        # These attributes ('epochs', 'lr', 'batch_size') must be defined in args by some parser
        # if this path is taken. The parse_global_args can provide defaults.
        add_param_if_relevant('epochs', 'e', default_to_skip=0) # Check against a common "not set" default
        add_param_if_relevant('lr', 'lr', default_to_skip=0.0)
        if not (hasattr(args, 'alt_style') and args.alt_style and str(args.alt_style).lower() not in ['none', '']):
             # Only add generic batch_size if not already covered by idbs/recbs
            if not (hasattr(args, 'id_batch_size') or hasattr(args, 'rec_batch_size')):
                add_param_if_relevant('batch_size', 'bs', default_to_skip=0)
        
    add_param_if_relevant('backbone', 'bb')
    add_param_if_relevant('item_indexing', 'idx')
    add_param_if_relevant('tasks')
    add_param_if_relevant('sample_prompt', 'sp', default_to_skip=0)
    if getattr(args, 'sample_prompt', 0) == 1 and hasattr(args, 'sample_num'): # Only add sample_num if sample_prompt is active
        add_param_if_relevant('sample_num', 'sn')
    add_param_if_relevant('max_his', 'mh', default_to_skip=-1)
    add_param_if_relevant('his_prefix', 'hp', default_to_skip=0) 
    add_param_if_relevant('skip_empty_his', 'seh', default_to_skip=1)
    add_param_if_relevant('prompt_file', 'pf', is_path=True)
    add_param_if_relevant('seed') 
    if hasattr(args, 'distributed'): add_param_if_relevant('distributed', 'dist', default_to_skip=0)


    # Join parameters, ensuring they are strings and sanitized
    final_log_name_parts = []
    for p_val_str in params:
        # Sanitize common problematic characters for filenames
        s_val = str(p_val_str).replace(':', '_').replace('[', '').replace(']', '').replace(',', '-').replace("'", "")
        s_val = s_val.replace(' ', '').replace('/', '_').replace('..', '') # Remove spaces, replace slashes
        if s_val: # Add only if it's not an empty string after processing
            final_log_name_parts.append(s_val)
            
    return '_'.join(final_log_name_parts).strip('_')

def setup_model_path(args):
    import datetime
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.model_path = os.path.join(args.model_dir, f"{folder_name}_id_{args.id_epochs}_rec_{args.rec_epochs}_{timestamp}")
    from pathlib import Path
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
