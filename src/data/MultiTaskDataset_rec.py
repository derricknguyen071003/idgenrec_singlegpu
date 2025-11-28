import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.prompt import load_prompt_template, get_info_from_prompt
from utils import utils
from utils import indexing
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb


class MultiTaskDatasetRec(Dataset):
    def __init__(self, args, dataset, mode, model_gen, tokenizer, phase=0, regenerate=True, component=None):
        super().__init__()
        self.model_gen = model_gen
        self.tokenizer = tokenizer
        self.data_path = args.data_path
        self.dataset = dataset
        self.tasks = args.tasks.split(',')
        self.item_indexing = args.item_indexing
        self.mode = mode
        self.args = args   
        self.phase = phase
        self.prompt = load_prompt_template(args.prompt_file, self.tasks)
        self.info = get_info_from_prompt(self.prompt)
        self.component = component if component is not None else None
        logging.info(f"(MultiTaskDatasetRec) Component: {self.component}")
        if 'history' in self.info:
            self.max_his = args.max_his
            self.his_sep = args.his_sep

        self.user_sequence = utils.ReadLineFromFile(os.path.join(self.data_path, self.dataset, 'user_sequence.txt'))
        self.user_sequence_dict = indexing.construct_user_sequence_dict(self.user_sequence)  
        indexing.generative_indexing_rec(self.data_path, self.dataset, self.user_sequence_dict, model_gen=self.model_gen, tokenizer=self.tokenizer, phase=phase, regenerate=regenerate, run_id=args.run_id, component=self.component, run_type=args.run_type)
        self.reindex_user_seq_dict, self.item_map = indexing.generative_indexing_rec(
            self.data_path, self.dataset, self.user_sequence_dict,
            model_gen=self.model_gen, tokenizer=self.tokenizer, phase=phase, 
            regenerate=False, run_id=args.run_id, component=self.component, run_type=args.run_type
        )
        if args.social_quantization_id:
            self.user_social_map = indexing.generate_user_social_generative_index(
                self.data_path, self.dataset, self.model_gen, self.tokenizer, phase, run_id=args.run_id, regenerate=True
            )
            run_dir = os.path.join(self.data_path, self.dataset, args.run_id)
            suffix = ''
            if args.run_type == '2id2rec':
                if self.component == 'item_rec':
                    suffix = '_item'
                elif self.component == 'friend_rec':
                    suffix = '_social'
            elif args.run_type == '2id1rec':
                if self.component == 'item_view':
                    suffix = '_item'
                elif self.component == 'social_view':
                    suffix = '_social'
            user_map = indexing.get_dict_from_lines(utils.ReadLineFromFile(os.path.join(run_dir, f'user_generative_index_phase_{phase}{suffix}.txt')))
            self.textual_to_social_map = {}
            for original_id, textual_id in user_map.items():
                if original_id in self.user_social_map:
                    self.textual_to_social_map[textual_id] = self.user_social_map[original_id]
            logging.info(f"(MultiTaskDatasetRec) Generated user social quantization map with {len(self.user_social_map)} users")
        else:
            self.user_social_map = {}
            self.textual_to_social_map = {}
        self.all_items = list(self.item_map.values())
        
        # Load cross-view token information for diffusion (if enabled and in 2id2rec mode)
        self.cross_view_dict = {}
        if getattr(args, 'use_diffusion', 0) and args.run_type in ['2id2rec', '2id2rec_socialtoid']:
            if self.component == 'item_rec':
                # For item view: need social view tokens
                try:
                    run_dir = os.path.join(self.data_path, self.dataset, args.run_id)
                    # Try to load social friend sequence with generated IDs
                    social_suffix = '_social'
                    social_user_index_file = os.path.join(run_dir, f'user_generative_index_phase_{phase}{social_suffix}.txt')
                    if os.path.exists(social_user_index_file):
                        social_user_dict = indexing.get_dict_from_lines(utils.ReadLineFromFile(social_user_index_file))
                        # Load friend sequence to get social tokens per user
                        friend_sequence_lines = utils.ReadLineFromFile(os.path.join(self.data_path, self.dataset, 'friend_sequence.txt'))
                        friend_sequence_dict = indexing.construct_user_sequence_dict(friend_sequence_lines)
                        # Map original user IDs to their social token sequences
                        for orig_uid, friends in friend_sequence_dict.items():
                            if orig_uid in social_user_dict:
                                # Get social tokens for this user's friends
                                social_tokens = []
                                for friend in friends[:-2]:  # Training friends only
                                    if friend in social_user_dict:
                                        social_tokens.append(social_user_dict[friend])
                                if social_tokens:
                                    self.cross_view_dict[orig_uid] = ' '.join(social_tokens)
                        logging.info(f"(MultiTaskDatasetRec) Loaded cross-view (social) tokens for {len(self.cross_view_dict)} users")
                except Exception as e:
                    logging.warning(f"(MultiTaskDatasetRec) Could not load cross-view tokens: {e}")
        
        self.data_samples = self.load_train()
        self.valid_data_samples = self.load_validation()
        self.get_prompt_info()
        self.construct_sentence()

    def load_train(self):
        """
        Load training data samples. Per-user splitting rules:
        - 1 item: train only
        - 2 items: one for train, one for validation
        - >=3 items: rest for train, last two reserved for val/test
        """
        data_samples = []

        for user in self.reindex_user_seq_dict:
            items = self.reindex_user_seq_dict[user]
            train_items, _, _ = self._split_user_sequence(items)
            if not train_items:
                continue
            for i in range(len(train_items)):
                one_sample = dict()
                one_sample['dataset'] = self.dataset
                one_sample['user_id'] = self._get_user_identifier(user)
                # Store original user ID for cross-view lookup
                one_sample['original_user_id'] = user
                one_sample['target'] = train_items[i]
                if 'history' in self.info:
                    history = train_items[:i]  
                    if self.max_his > 0:
                        history = history[-self.max_his:]
                    one_sample['history'] = self.his_sep.join(history)
                data_samples.append(one_sample)
        return data_samples

    
    def load_validation(self):
        """
        Load validation data samples
        """
        data_samples = []
        for user in self.reindex_user_seq_dict:
            items = self.reindex_user_seq_dict[user]
            train_items, val_item, _ = self._split_user_sequence(items)
            if val_item is None:
                continue
            one_sample = dict()
            one_sample['dataset'] = self.dataset
            one_sample['user_id'] = self._get_user_identifier(user)
            one_sample['original_user_id'] = user
            one_sample['target'] = val_item
            if 'history' in self.info:
                history = train_items
                if self.max_his > 0:
                    history = history[-self.max_his:]    
                one_sample['history'] = self.his_sep.join(history)
            data_samples.append(one_sample)
        return data_samples

    def _split_user_sequence(self, items):
        """
        Returns (train_items, val_item, test_item) per user following:
        - len==0: ([], None, None)
        - len==1: ([item0], None, None)
        - len==2: ([item0], item1, None)
        - len>=3: (items[:-2], items[-2], items[-1])
        """
        if not items:
            return [], None, None
        if len(items) == 1:
            return items[:], None, None
        if len(items) == 2:
            return items[:-1], items[-1], None
        return items[:-2], items[-2], items[-1]

    def _get_user_identifier(self, user):
        if getattr(self.args, 'social_quantization_id', None) and user in self.textual_to_social_map:
            return self.textual_to_social_map[user]
        return user
    
    def __len__(self):
        return len(self.data['input'])
    
    def construct_sentence(self):
        if self.args.sample_prompt == 0:
            self._construct_sentence_all()
        else:
            self._construct_sentence_sample()
    
    def shuffle(self, seed):
            g = torch.Generator()
            g.manual_seed(seed)    
            for task in self.task_data:
                indices = torch.randperm(len(self.task_data[task]), generator=g).tolist()
                self.task_data[task] = [self.task_data[task][i] for i in indices]


    def _construct_sentence_valid(self):
        """Construct validation sentences using valid_data_samples"""
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        self.idx_to_sample_idx = []  # Mapping from data index to sample index
        
        # Use first seen prompt for validation if valid_prompt not specified
        if hasattr(self, 'valid_prompt') and self.valid_prompt:
            setting = self.valid_prompt.split(':')
            prompt_key = (setting[0], setting[1])
        else:
            # Default to first seen prompt
            prompt_key = ('seen', '0')
        
        for task in self.tasks:
            for i in range(len(self.valid_data_samples)):
                datapoint = self.valid_data_samples[i]
                if prompt_key[0] in self.prompt[task] and prompt_key[1] in self.prompt[task][prompt_key[0]]:
                    self.data['input'].append(self.prompt[task][prompt_key[0]][prompt_key[1]]['Input'].format(**datapoint))
                    self.data['output'].append(self.prompt[task][prompt_key[0]][prompt_key[1]]['Output'].format(**datapoint))
                    self.idx_to_sample_idx.append(i)  # Map this data index to sample index i
    
    def _construct_sentence_all(self):
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        self.idx_to_sample_idx = []  # Mapping from data index to sample index
        for task in self.tasks:
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                for pid in self.prompt[task]['seen']:
                    self.data['input'].append(self.prompt[task]['seen'][pid]['Input'].format(**datapoint))
                    self.data['output'].append(self.prompt[task]['seen'][pid]['Output'].format(**datapoint))
                    self.idx_to_sample_idx.append(i)  # Map this data index to sample index i
                    
    def _construct_sentence_sample(self):
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        self.idx_to_sample_idx = []  
        for t in range(len(self.tasks)):
            task = self.tasks[t]
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                for j in range(self.task_prompt_num[t]):
                    pid = random.randint(0, len(self.prompt[task]['seen']) - 1)
                    self.data['input'].append(self.prompt[task]['seen'][str(pid)]['Input'].format(**datapoint))
                    self.data['output'].append(self.prompt[task]['seen'][str(pid)]['Output'].format(**datapoint))
                    self.idx_to_sample_idx.append(i)  # Map this data index to sample index i
            
    def get_prompt_info(self):
        """
        Calculate number of prompts and cumulative index for each task
        - task_prompt_num: save the number of prompts for each task
        - task_index: the cumulative index for each task. if task_index[i-1] <= idx < task_index[i], then the idx belongs to task[i]
            - For example, there are 100 data samples in total, there are 3 tasks, the task_prompt_num is [2,1,3], then the task_index is [200, 300, 600].
        """       
        if self.args.sample_prompt == 0:
            self.task_prompt_num = [len(self.prompt[task]['seen']) for task in self.tasks]
        else:
            sample_number = self.args.sample_num.split(',')
            self.task_prompt_num = [int(sample_number[i]) for i in range(len(self.tasks))]
        self.task_index = [self.task_prompt_num[0] * len(self.data_samples)]
        for i in range(1, len(self.task_prompt_num)):
            self.task_index.append(self.task_index[i-1] + self.task_prompt_num[i] * len(self.data_samples))
        self.task_data = dict()
        for i in range(len(self.tasks)):
            if i == 0:
                start = 0
            else:
                start = self.task_index[i-1]
            end = self.task_index[i]
            task = self.tasks[i]
            self.task_data[task] = [i for i in range(start, end)]
 
    def __getitem__(self, idx):
        sample_idx = self.idx_to_sample_idx[idx]
        result = {
            'input': self.data['input'][idx],
            'output': self.data['output'][idx],
            'user_idx': self.data_samples[sample_idx]['user_id']
        }
        
        # Add cross-view tokens if available (for diffusion)
        if hasattr(self, 'cross_view_dict') and self.cross_view_dict:
            # Get original user ID from data sample
            user_id = self.data_samples[sample_idx].get('original_user_id', self.data_samples[sample_idx]['user_id'])
            # Try to find cross-view tokens (may need to map through textual_to_social_map)
            if user_id in self.cross_view_dict:
                result['cross_view_tokens'] = self.cross_view_dict[user_id]
            else:
                result['cross_view_tokens'] = None
        else:
            result['cross_view_tokens'] = None
        
        return result


class ValidationDatasetRec(MultiTaskDatasetRec):
    """Validation dataset that uses valid_data_samples instead of data_samples"""
    def __init__(self, base_dataset):
        # Don't call super().__init__ to avoid re-loading data
        # Instead, copy all necessary attributes from base dataset
        Dataset.__init__(self)
        
        # Copy all attributes from base dataset
        for attr in dir(base_dataset):
            if not attr.startswith('_') and not callable(getattr(base_dataset, attr, None)):
                try:
                    setattr(self, attr, getattr(base_dataset, attr))
                except:
                    pass
        
        # Override with validation samples
        self.data_samples = base_dataset.valid_data_samples
        if len(self.data_samples) > 0:
            self.construct_validation_sentence()
        else:
            # Create empty data structure if no validation samples
            self.data = {'input': [], 'output': []}
            self.idx_to_sample_idx = []
    
    def construct_validation_sentence(self):
        """Construct validation sentences using valid_data_samples"""
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        self.idx_to_sample_idx = []
        
        # Use first seen prompt for validation if valid_prompt not specified
        if hasattr(self.args, 'valid_prompt') and self.args.valid_prompt:
            setting = self.args.valid_prompt.split(':')
            prompt_key = (setting[0], setting[1])
        else:
            # Default to first seen prompt
            prompt_key = ('seen', '0')
        
        for task in self.tasks:
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                if prompt_key[0] in self.prompt[task] and prompt_key[1] in self.prompt[task][prompt_key[0]]:
                    self.data['input'].append(self.prompt[task][prompt_key[0]][prompt_key[1]]['Input'].format(**datapoint))
                    self.data['output'].append(self.prompt[task][prompt_key[0]][prompt_key[1]]['Output'].format(**datapoint))
                    self.idx_to_sample_idx.append(i)