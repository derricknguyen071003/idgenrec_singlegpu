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
import logging
import pdb
import re

class TestDatasetGen(Dataset):
    def __init__(self, args, dataset, task, model_gen, tokenizer, regenerate, phase, component=None, run_type=None):
        super().__init__()
        self.args = args
        self.data_path = args.data_path
        self.dataset = dataset
        self.task = task
        self.phase = phase
        self.component = component if component is not None else None
        self.run_type = run_type if run_type is not None else None
        self.prompt = load_prompt_template(args.prompt_file, [self.task])
        self.info = get_info_from_prompt(self.prompt)
        if 'history' in self.info:
            self.max_his = args.max_his
            self.his_sep = args.his_sep
        self.user_sequence = utils.ReadLineFromFile(os.path.join(self.data_path, self.dataset, 'user_sequence.txt'))
        self.user_sequence_dict = indexing.construct_user_sequence_dict(self.user_sequence)
        self.reindex_user_seq_dict, self.item_map = indexing.generative_indexing_rec(self.data_path, self.dataset, self.user_sequence_dict, model_gen=model_gen, tokenizer=tokenizer, regenerate=False, phase=self.args.phase, run_id=self.args.run_id, component=self.component, run_type=self.run_type) 
        self.all_items = list(self.item_map.values())
        self.test_prompt = args.test_prompt
        self.data_samples = self.load_test()    
        self.construct_sentence()
        
    def load_test(self):
        """
        Load test data samples. Only include users with at least 3 items:
        - 1 item: train only (no test)
        - 2 items: train + validation (no test)
        - >=3 items: train + validation + test (use last item for test)
        """
        data_samples = []
        for user in self.reindex_user_seq_dict:
            items = self.reindex_user_seq_dict[user]
            # Only test users with at least 3 items (they have test data)
            if len(items) < 3:
                continue
            one_sample = dict()
            one_sample['dataset'] = self.dataset
            one_sample['user_id'] = user
            one_sample['target'] = items[-1]  # Last item is test target
            if 'history' in self.info:
                history = items[:-1]  # All items except the last (test) one
                if self.max_his > 0:
                    history = history[-self.max_his:]
                one_sample['history'] = self.his_sep.join(history)
            data_samples.append(one_sample)
        return data_samples
    
    def __len__(self):
        return len(self.data_samples)
    
    def construct_sentence(self):
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        info = self.test_prompt.split(':')
        prompt = self.prompt[self.task][info[0]][info[1]]
        for i in range(len(self.data_samples)):
            datapoint = self.data_samples[i]
            self.data['input'].append(prompt['Input'].format(**datapoint))
            self.data['output'].append(prompt['Output'].format(**datapoint))     
    def __getitem__(self, idx):
        return {'input': self.data['input'][idx],
               'output': self.data['output'][idx]}

    
class TestDatasetSocial(Dataset):
    def __init__(self, args, dataset, task, model_gen, tokenizer, regenerate, phase, component=None, run_type=None):
        super().__init__()
        self.args = args
        self.data_path = args.data_path
        self.dataset = dataset
        self.task = task
        self.phase = phase
        self.component = component if component is not None else None
        self.run_type = run_type if run_type is not None else None
        self.prompt = load_prompt_template(args.social_prompt_file, [self.task])
        self.info = get_info_from_prompt(self.prompt)
        #logging.info(f"Info: {self.info}")
        if 'history' in self.info:
            self.max_his = args.max_his
            self.his_sep = args.his_sep
        friend_sequence_lines = utils.ReadLineFromFile(os.path.join(self.data_path, self.dataset, f'friend_sequence.txt'))
        if hasattr(args, 'random_remove_friend') and args.random_remove_friend > 0.0:
            friend_sequence_lines = utils.remove_friend_connections_from_lines(
                friend_sequence_lines, args.random_remove_friend, args.seed
            )
        self.friend_sequence_dict = indexing.construct_user_sequence_dict(friend_sequence_lines)
        self.reindex_friend_seq_dict, self.user_map = indexing.generative_indexing_social(
            self.data_path, self.dataset, self.friend_sequence_dict, 
            model_gen=model_gen, tokenizer=tokenizer, regenerate=False, 
            phase=self.args.phase, run_id=self.args.run_id, 
            component=self.component, run_type=self.run_type
        ) 
        self.all_users = list(self.user_map.values())
        self.test_prompt = args.test_prompt
        self.data_samples = self.load_test()    
        self.construct_sentence()
        
        
    def load_test(self):
        """
        Load test data samples. Only include users with at least 3 friends:
        - 1 friend: train only (no test)
        - 2 friends: train + validation (no test)
        - >=3 friends: train + validation + test (use last friend for test)
        """
        data_samples = []
        for user in self.reindex_friend_seq_dict:
            friends = self.reindex_friend_seq_dict[user]
            # Only test users with at least 3 friends (they have test data)
            if len(friends) < 3:
                continue
            one_sample = dict()
            one_sample['dataset'] = self.dataset
            one_sample['user_id'] = user
            one_sample['target'] = friends[-1]  # Last friend is test target
            if 'history' in self.info:
                history = friends[:-1]  # All friends except the last (test) one
                if self.max_his > 0:
                    history = history[-self.max_his:]
                one_sample['history'] = self.his_sep.join(history)
            data_samples.append(one_sample)
        return data_samples
    
    def __len__(self):
        return len(self.data_samples)
    
    def construct_sentence(self):
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        info = self.test_prompt.split(':')
        prompt = self.prompt[self.task][info[0]][info[1]]
        for i in range(len(self.data_samples)):
            datapoint = self.data_samples[i]
            self.data['input'].append(prompt['Input'].format(**datapoint))
            self.data['output'].append(prompt['Output'].format(**datapoint))     
    
    def __getitem__(self, idx):
        return {'input': self.data['input'][idx],
               'output': self.data['output'][idx]}
