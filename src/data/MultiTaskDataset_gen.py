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
from sentence_transformers import SentenceTransformer

class MultiTaskDatasetGen(Dataset):
    def __init__(self, args, dataset, mode, phase=0, model_gen=None, tokenizer=None, component=None):
        super().__init__()
        self.data_path = args.data_path
        self.dataset = dataset
        self.tasks = args.tasks.split(',')
        self.item_indexing = args.item_indexing
        self.mode = mode
        self.args = args
        self.phase = phase
        self.component = component if component is not None else None
        self.prompt = load_prompt_template(args.prompt_file, self.tasks)
        self.info = get_info_from_prompt(self.prompt)
        if args.social_quantization_id:
            self.model_gen = model_gen
            self.tokenizer = tokenizer
            self.sentence_transformer = SentenceTransformer("sentence-transformers/sentence-t5-base")
        if 'history' in self.info:
            self.max_his = int(args.max_his / 2)
            self.his_sep = args.his_sep
        # load user sequence data
        self.user_sequence = utils.ReadLineFromFile(os.path.join(self.data_path, self.dataset, 'user_sequence.txt'))
        self.user_sequence_dict = indexing.construct_user_sequence_dict(self.user_sequence)  
        if args.run_type == '2id2rec' or args.run_type == '2id2rec_socialtoid':
            self.reindex_user_seq_dict, self.item_map = indexing.generative_indexing_id(self.data_path, self.dataset, self.user_sequence_dict, phase, run_id=args.run_id, component=self.component, run_type=args.run_type, social_quantization_id=self.args.social_quantization_id)
        else:
            self.reindex_user_seq_dict, self.item_map = indexing.generative_indexing_id(self.data_path, self.dataset, self.user_sequence_dict, phase, run_id=args.run_id, component=self.component, social_quantization_id=self.args.social_quantization_id)
        self.reindex_user_seq_dict_rec, self.item_map_rec = self.reindex_user_seq_dict, self.item_map
        logging.info("Reindex GEN data (item_id) with generative indexing method (single GPU)")
        if args.social_quantization_id:
            self.user_social_map = indexing.generate_user_social_generative_index(
                self.data_path, self.dataset, self.model_gen, self.tokenizer, phase, run_id=args.run_id, regenerate=False
            )
            run_dir = os.path.join(self.data_path, self.dataset, args.run_id)
            user_map = indexing.get_dict_from_lines(utils.ReadLineFromFile(os.path.join(run_dir, f'user_generative_index_phase_{phase}.txt')))
            self.textual_to_social_map = {}
            for original_id, textual_id in user_map.items():
                if original_id in self.user_social_map:
                    self.textual_to_social_map[textual_id] = self.user_social_map[original_id]
            logging.info(f"(MultiTaskDatasetGen) Generated user social quantization map with {len(self.user_social_map)} users")
        else:
            self.user_social_map = {}
            self.textual_to_social_map = {}
        
        self.all_items = list(self.item_map.values())
        self.data_samples = self.load_train_id()
        self.get_prompt_info()
        self.generate_data()

    def shuffle(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        for task in self.task_data:
            indices = torch.randperm(len(self.task_data[task]), generator=g).tolist()
            self.task_data[task] = [self.task_data[task][i] for i in indices]

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

    def load_train_id(self):
        """
        Load training data samples
        """
        data_samples = []
        for user in self.reindex_user_seq_dict:
            items = self.reindex_user_seq_dict[user][:-2]
            items_rec = self.reindex_user_seq_dict_rec[user][:-2]
            for i in range(len(items)):
                one_sample = dict()
                one_sample['dataset'] = self.dataset
                if self.args.social_quantization_id and user in self.textual_to_social_map:
                    one_sample['user_id'] = self.textual_to_social_map[user]
                else:
                    one_sample['user_id'] = user
                one_sample['target'] = items_rec[i]
                if 'history' in self.info:
                    history = items[:i]
                    if self.max_his > 0:
                        history = history[-self.max_his:]
                    one_sample['history'] = history
                data_samples.append(one_sample)
        return data_samples

    def load_validation(self):
        """
        Load validation data samples
        """
        data_samples = []
        for user in self.reindex_user_seq_dict:
            items = self.reindex_user_seq_dict[user]
            one_sample = dict()
            one_sample['dataset'] = self.dataset
            one_sample['user_id'] = user
            one_sample['target'] = items[-2]
            if 'history' in self.info:
                history = items[:-2]
                if self.max_his > 0:
                    history = history[-self.max_his:]
                one_sample['history'] = self.his_sep.join(history)
            data_samples.append(one_sample)
        return data_samples
         
    def __len__(self):
        return len(self.data['input_prompt'])

    
 
    def construct_sentence(self):
        if self.mode == 'train':
            if self.args.sample_prompt == 0:
                self._construct_sentence_all()
            else:
                self._construct_sentence_sample()
            logging.info(f"Input: {self.data['input'][100]} , Output: {self.data['output'][100]} ")
            logging.info(f"Input: {self.data['input'][101]} , Output: {self.data['output'][101]} ")
        elif self.mode == 'validation':
            self._construct_sentence_valid()
            logging.info(f"Input: {self.data['input'][101]} , Output: {self.data['output'][101]} ")
    
    def _construct_sentence_valid(self):
        self.data = {}
        setting = self.valid_prompt.split(':')
        for task in self.tasks:
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                self.data['input'].append(self.prompt[task][setting[0]][setting[1]]['Input'].format(**datapoint))
                self.data['output'].append(self.prompt[task][setting[0]][setting[1]]['Output'].format(**datapoint))
    
    def _construct_sentence_all(self):
        self.data = {}
        for task in self.tasks:
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                for pid in self.prompt[task]['seen']:
                    self.data['input'].append(self.prompt[task]['seen'][pid]['Input'].format(**datapoint))
                    self.data['output'].append(self.prompt[task]['seen'][pid]['Output'].format(**datapoint))
                    
    def _construct_sentence_sample(self):
        self.data = {}
        for t in range(len(self.tasks)):
            task = self.tasks[t]
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                for j in range(self.task_prompt_num[t]):
                    pid = random.randint(0, len(self.prompt[task]['seen']) - 1)
                    self.data['input'].append(self.prompt[task]['seen'][str(pid)]['Input'].format(**datapoint))
                    self.data['output'].append(self.prompt[task]['seen'][str(pid)]['Output'].format(**datapoint))
        
    def generate_data(self):
        """
        Applies prompt templates to raw user-item interaction data.
        Generates natural language input-output pairs for generative training.
        Supports multi-task settings by looping through multiple tasks and multiple prompt styles."""
        self.data = {}
        self.data['history'] = []
        self.data['target'] = []
        self.data['input_prompt'] = []
        self.data['output_prompt'] = []

        for t in range(len(self.tasks)):
            task = self.tasks[t]
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                for j in range(self.task_prompt_num[t]):
                    pid = random.randint(0, len(self.prompt[task]['seen']) - 1)
                    dataset = datapoint['dataset']
                    user_id = datapoint['user_id']
                    target = datapoint['target']
                    self.data['history'].append(datapoint['history'])
                    self.data['target'].append(target)

                    input_prompt = self.prompt[task]['seen'][str(pid)]['Input'].format(**{'dataset': dataset, 'user_id': user_id, 'history': '{history}'})
                    output_prompt = self.prompt[task]['seen'][str(pid)]['Output'].format(**{'dataset': dataset, 'target': target})

                    self.data['input_prompt'].append(input_prompt)
                    self.data['output_prompt'].append(output_prompt)
        return True

    def __getitem__(self, idx):
        return {'history': self.data['history'][idx],
               'target': self.data['target'][idx],
               'input_prompt': self.data['input_prompt'][idx],
               'output_prompt': self.data['output_prompt'][idx]}