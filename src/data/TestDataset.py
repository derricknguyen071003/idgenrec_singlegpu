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
        #logging.info(f"Args: {args}")
        self.data_path = args.data_path
        self.dataset = dataset
        self.task = task
        self.phase = phase
        self.component = component if component is not None else None
        self.run_type = run_type if run_type is not None else None
        self.prompt = load_prompt_template(args.prompt_file, [self.task])
        self.info = get_info_from_prompt(self.prompt)
        #logging.info(f"Info: {self.info}")
        if 'history' in self.info:
            self.max_his = args.max_his
            self.his_sep = args.his_sep
        
        # load user sequence data
        self.user_sequence = utils.ReadLineFromFile(os.path.join(self.data_path, self.dataset, 'user_sequence.txt'))
        self.user_sequence_dict = indexing.construct_user_sequence_dict(self.user_sequence)
        # logging.info(f"Current data path: {self.data_path}")
        # logging.info(f"Current dataset: {self.dataset}")
        # logging.info(f"Current phase: {self.args.phase}")
        # logging.info(f"Run ID: {self.args.run_id}")
        self.reindex_user_seq_dict, self.item_map = indexing.generative_indexing_rec(self.data_path, self.dataset, self.user_sequence_dict, model_gen=model_gen, tokenizer=tokenizer, regenerate=False, phase=self.args.phase, run_id=self.args.run_id, component=self.component, run_type=self.run_type) 
        self.all_items = list(self.item_map.values())
        self.test_prompt = args.test_prompt
        self.data_samples = self.load_test()    
        self.construct_sentence()
        
    def load_test(self):
        """
        Load test data samples
        """
        data_samples = []
        for user in self.reindex_user_seq_dict:
            items = self.reindex_user_seq_dict[user]
            one_sample = dict()
            one_sample['dataset'] = self.dataset
            one_sample['user_id'] = user
            one_sample['target'] = items[-1]
            if 'history' in self.info:
                history = items[:-1]
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

    
