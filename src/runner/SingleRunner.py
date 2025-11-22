import torch
torch.cuda.empty_cache()
from transformers import get_linear_schedule_with_warmup, T5Config, T5ForConditionalGeneration, AutoTokenizer, GenerationConfig
from torch.optim import AdamW
import logging
from tqdm import tqdm
import wandb
from utils import utils, evaluate 
from data.TestDataset import TestDatasetGen
from data.TestDataset_social import TestDatasetSocial
from torch.utils.data import DataLoader
from processor.Collator import Collator, TestCollator 
import numpy as np
import os
import string 
import datetime
from utils.generation_trie import Trie, prefix_allowed_tokens_fn
import utils.generation_trie as gt
from utils import indexing
from utils.dataset_utils import get_dataset_generative, get_loader
from transformers import T5Config, T5ForConditionalGeneration
from utils.discrete_diffusion import (
    add_timestep_tokens_to_tokenizer, 
    prepend_timestep_token,
    create_noise_head,
    compute_kl_divergence
)


class SingleRunner:

    def __init__(self, model_rec=None, model_gen=None, model_social=None, tokenizer=None, 
                 train_loader_id=None, train_loader_rec=None, train_loader_rec_social=None, train_loader_social=None,
                 valid_loader=None, device=None, args=None, component=None):
        self.model_rec = model_rec.to(device) if model_rec else None
        self.model_gen = model_gen.to(device) if model_gen else None
        self.model_social = model_social.to(device) if model_social else None
        self.tokenizer = tokenizer
        self.train_loader_id = train_loader_id
        self.train_loader_rec = train_loader_rec if train_loader_rec else None
        self.train_loader_rec_social = train_loader_rec_social if train_loader_rec_social else None
        self.train_loader_social = train_loader_social if train_loader_social else None
        self.valid_loader = valid_loader 
        self.device = device
        self.args = args
        self.component = component if component is not None else None
        self.rounds = args.rounds
        self.global_epoch_tracker = 0
        self.total_id_epoch = 0
        self.social_optimizer = None
        self.social_scheduler = None
        self.id_optimizer, self.id_scheduler, self.rec_optimizer, self.rec_scheduler = self.create_optimizer_and_scheduler()
        self.metrics = args.metrics.split(',')
        self.generate_num = max([int(m.split('@')[1]) for m in self.metrics if '@' in m and len(m.split('@')) > 1] or [10])
        punctuation_tokens = [self.tokenizer.encode(p, add_special_tokens=False)[0] for p in string.punctuation]
        self.punctuation_tokens_tensor = torch.tensor(punctuation_tokens, device=self.device)        
        self.use_diffusion = getattr(args, 'use_diffusion', 0)

        if self.use_diffusion:
            num_timesteps = getattr(args, 'diffusion_timesteps', 100)
            self.timestep_token_ids = add_timestep_tokens_to_tokenizer(self.tokenizer, num_timesteps)
            if self.model_rec:
                self.model_rec.resize_token_embeddings(len(self.tokenizer))
            if self.model_gen:
                self.model_gen.resize_token_embeddings(len(self.tokenizer))
            if self.model_social is not None:
                self.model_social.resize_token_embeddings(len(self.tokenizer))
            logging.info(f"Added timestep tokens: T={num_timesteps}, vocab_size={len(self.tokenizer)}")
            
            # Create noise prediction heads only during training (not needed for inference)
            if self.args.train:
                noise_head_dropout = self.args.noise_head_dropout
                if self.model_rec:
                    self.noise_head_rec = create_noise_head(self.model_rec, dropout=noise_head_dropout).to(self.device)
                else:
                    self.noise_head_rec = None
                if self.model_social is not None:
                    self.noise_head_social = create_noise_head(self.model_social, dropout=noise_head_dropout).to(self.device)
                else:
                    self.noise_head_social = None
                logging.info("Created noise prediction heads for recommender models (training mode)")
            else:
                self.noise_head_rec = None
                self.noise_head_social = None
                logging.info("Skipping noise prediction heads (inference mode - not needed)")
        else:
            self.timestep_token_ids = None
            self.noise_head_rec = None
            self.noise_head_social = None
    


    def create_optimizer_and_scheduler(self):
        batch_per_epoch_id = len(self.train_loader_id) if self.train_loader_id else 0
        batch_per_epoch_rec = len(self.train_loader_rec) if self.train_loader_rec else 0
        id_total_steps = batch_per_epoch_id // self.args.gradient_accumulation_steps * self.args.id_epochs * self.rounds
        id_warmup_steps = int(id_total_steps * self.args.warmup_prop)
        rec_total_steps = batch_per_epoch_rec // self.args.gradient_accumulation_steps * self.args.rec_epochs * self.rounds
        rec_warmup_steps = int(rec_total_steps * self.args.warmup_prop)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_id = None
        scheduler_id = None
        if self.model_gen:
            optimizer_grouped_parameters_id = [
                {
                    "params": [
                        p
                        for n, p in self.model_gen.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model_gen.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_grouped_parameters_id = []
        optimizer_rec = None
        scheduler_rec = None
        if self.model_rec:
            optimizer_grouped_parameters_rec = [
                {
                    "params": [
                        p
                        for n, p in self.model_rec.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model_rec.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_grouped_parameters_rec = []

        if self.args.optim.lower() == 'adamw':
            if optimizer_grouped_parameters_id:
                optimizer_id = AdamW(optimizer_grouped_parameters_id, lr=self.args.id_lr, eps=self.args.adam_eps)
            if optimizer_grouped_parameters_rec:
                optimizer_rec = AdamW(optimizer_grouped_parameters_rec, lr=self.args.rec_lr, eps=self.args.adam_eps)
        else:
            raise NotImplementedError
            
        if optimizer_id and id_total_steps > 0:
            scheduler_id = get_linear_schedule_with_warmup(optimizer_id, id_warmup_steps, id_total_steps)
        if optimizer_rec and rec_total_steps > 0:
            scheduler_rec = get_linear_schedule_with_warmup(optimizer_rec, rec_warmup_steps, rec_total_steps)
        return optimizer_id, scheduler_id, optimizer_rec, scheduler_rec
    
    def _train_id_generator_phase(self, current_round_num):
        logging.info(f"--- Round {current_round_num + 1}: Training ID Generator ---")
        for param in self.model_gen.parameters(): param.requires_grad = True
        for param in self.model_rec.parameters(): param.requires_grad = False
        self.model_gen.train()
        self.model_rec.train() 
        for id_epoch in range(self.args.id_epochs):
            logging.info(f"ID Gen - Round {current_round_num + 1}, Epoch {id_epoch + 1}/{self.args.id_epochs}")
            self.train_loader_id.sampler.set_epoch(self.global_epoch_tracker)
            epoch_losses = []
            for batch in tqdm(self.train_loader_id):
                input_prompt_ids = batch[0].to(self.device, non_blocking=True) 
                input_prompt_positions = batch[1].to(self.device, non_blocking=True)
                hist_ids = batch[2].to(self.device, non_blocking=True)
                hist_att = batch[3].to(self.device, non_blocking=True)
                output_ids = batch[4].to(self.device, non_blocking=True)
                batch_size = hist_ids.shape[0]
                hist_size = hist_ids.shape[1]
                input_tensor = hist_ids.view(-1, hist_ids.shape[-1])
                output = self.model_gen.generate_with_grad(
                            input_tensor,
                            attention_mask=hist_att.view(-1, hist_att.shape[-1]), 
                            max_length=10,
                            min_length=1,
                            num_beams=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                            output_hidden_states=False,
                            renormalize_logits=True,
                        )
                probabilities = torch.cat([score.unsqueeze(1) for score in output['scores']], dim=1)
                train_id_token_size = probabilities.shape[1] 
                token_embeddings = self.model_rec.shared.weight  
                hist_embeddings = torch.einsum('bsv,ve->bse', probabilities, token_embeddings)
                hist_embeddings = hist_embeddings.view(batch_size, hist_size, train_id_token_size, -1) 
                temp_ids = output['sequences'][:, 1:]
                punctuation_mask = utils.torch_isin(temp_ids, self.punctuation_tokens_tensor)
                batch_size_, hist_size_, seq_length_minus_one_, embedding_dim_ = hist_embeddings.shape
                punctuation_mask = punctuation_mask.view(batch_size_, hist_size_, seq_length_minus_one_)
                hist_embeddings[punctuation_mask.unsqueeze(-1).expand_as(hist_embeddings)] = 0
                input_prompt_embeddings = token_embeddings[input_prompt_ids]
                max_prompt_size = input_prompt_embeddings.shape[1]
                max_hist_num = hist_ids.shape[1] 
                max_input_len = max_prompt_size + max_hist_num * train_id_token_size 
                final_input = utils.insert_phrases_batch(input_prompt_embeddings, 
                                                    input_prompt_positions, 
                                                    hist_embeddings, 
                                                    max_input_len)
                norms = torch.norm(final_input, dim=-1)
                attention_mask = (norms > 1e-6).long()
                output = self.model_rec(
                    inputs_embeds=final_input, 
                    attention_mask=attention_mask,
                    labels=output_ids,
                    return_dict=True,
                )
                loss = output["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_gen.parameters(), self.args.clip)
                self.id_optimizer.step()
                self.id_scheduler.step()
                self.model_gen.zero_grad()
                self.model_rec.zero_grad()
                epoch_losses.append(loss.item())
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('nan')
            logging.info(f"ID Gen - Avg Loss Round {current_round_num+1} Epoch {id_epoch+1}: {avg_epoch_loss:.4f}")
            wandb.log({
                "id_gen/loss": avg_epoch_loss
            })

            self.global_epoch_tracker += 1
            self.total_id_epoch +=1

    def _train_recommender_phase(self, current_round_num):
        logging.info(f"--- Round {current_round_num + 1}: Training Recommender ---")
        for param in self.model_rec.parameters(): param.requires_grad = True
        for param in self.model_gen.parameters(): param.requires_grad = False
        self.model_rec.train()
        self.model_gen.train()
        current_phase_for_rec_dataset = current_round_num
        logging.info(f"Recommender training (Round {current_round_num+1}): Refreshing dataset/loader for phase {current_phase_for_rec_dataset}")
        _, refreshed_TrainSetRec = get_dataset_generative(self.args, self.model_gen, self.tokenizer, phase=current_phase_for_rec_dataset)
        _, self.train_loader_rec = get_loader(self.args, self.tokenizer, None, refreshed_TrainSetRec) 
        for rec_epoch in range(self.args.rec_epochs):
            logging.info(f"Recommender - Round {current_round_num + 1}, Epoch {rec_epoch + 1}/{self.args.rec_epochs}")
            self.train_loader_rec.sampler.set_epoch(self.global_epoch_tracker)
            epoch_losses = []
            for batch in tqdm(self.train_loader_rec):
                input_ids = batch[0].to(self.device, non_blocking=True)
                attn_mask = batch[1].to(self.device, non_blocking=True)
                output_ids = batch[3].to(self.device, non_blocking=True)
                user_ids = batch[5].to(self.device, non_blocking=True) if len(batch) > 5 else None
                token_embeddings = self.model_rec.shared.weight
                input_embeds = token_embeddings[input_ids]
                output = self.model_rec(
                    inputs_embeds=input_embeds,
                    attention_mask=attn_mask,
                    labels=output_ids,
                    return_dict=True,
                )
                loss = output.loss 
                loss.backward()                
                torch.nn.utils.clip_grad_norm_(self.model_rec.parameters(), self.args.clip)
                self.rec_optimizer.step()
                self.rec_scheduler.step()
                self.model_rec.zero_grad()
                self.model_gen.zero_grad()

                epoch_losses.append(loss.item())

            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('nan')
            logging.info(f"Rec - Avg Loss Round {current_round_num+1} Epoch {rec_epoch+1}: {avg_epoch_loss:.4f}")            
            wandb.log({
                "rec/loss": avg_epoch_loss
            })

            self.global_epoch_tracker += 1

    def train(self):
        for round_num in range(self.rounds):
            logging.info(f"========== Starting Alternation Round {round_num + 1}/{self.rounds} ==========")
            current_alt_style = self.args.alt_style
            if current_alt_style == 'id_first':
                logging.info(f"Training ID Gen first in round {round_num + 1} with style: {current_alt_style}")
                self._train_id_generator_phase(round_num)     
                logging.info(f"Training Recommender in round {round_num + 1} with style: {current_alt_style}")
                self._train_recommender_phase(round_num)
            elif current_alt_style == 'rec_first':
                logging.info(f"Training Recommender first in round {round_num + 1} with style: {current_alt_style}")
                self._train_recommender_phase(round_num)
                logging.info(f"Training ID Gen in round {round_num + 1} with style: {current_alt_style}")   
                self._train_id_generator_phase(round_num)
            logging.info(f"========== Finished Alternation Round {round_num + 1}/{self.rounds} ==========")
            if self.args.model_path: 
                os.makedirs(self.args.model_path, exist_ok=True)
                logging.info(f"Model directory ensured: {self.args.model_path}")
                gen_path = os.path.join(self.args.model_path, f"model_gen_round{round_num+1}_final.pt")
                torch.save(self.model_gen.state_dict(), gen_path)
                rec_path = os.path.join(self.args.model_path, f"model_rec_round{round_num+1}_final.pt")
                torch.save(self.model_rec.state_dict(), rec_path)
            self.global_epoch_tracker += 1
        logging.info("--- Alternating Training Finished ---")
        self._test_recommender()

    def get_testloader_friend(self):
        self.testloaders_social = []
        datasets_to_test = self.args.datasets.split(',')
        tasks_to_test = self.args.tasks.split(',')
        collator_social_test = TestCollator(self.tokenizer)
        for dataset_name in datasets_to_test:
            for task_name in tasks_to_test:                                
                logging.info("Using TestDatasetSocial for testing")
                test_data_social = TestDatasetSocial(
                    args=self.args,
                    dataset=dataset_name,
                    task=task_name,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=False,
                    phase=self.args.rounds,
                    component = "friend_rec",
                    run_type = self.args.run_type
                )
            test_loader_social = DataLoader(
                dataset=test_data_social,
                batch_size=self.args.eval_batch_size,
                collate_fn=collator_social_test,
                shuffle=False,
            )
            
            self.testloaders_social.append(test_loader_social)
            num_samples = len(test_data_social)
            num_batches = len(test_loader_social)
            logging.info(f"Created social testloader for {dataset_name}, with {num_samples} samples ({num_batches} batches, batch_size={self.args.eval_batch_size})")
                                          
    def get_testloader_item(self):
        self.testloaders_rec = []
        datasets_to_test = self.args.datasets.split(',')
        tasks_to_test = self.args.tasks.split(',')
        collator_rec_test = TestCollator(self.tokenizer)

        for dataset_name in datasets_to_test:
            for task_name in tasks_to_test:                
                logging.info("Using TestDatasetGen for testing")
                test_data_rec = TestDatasetGen(
                    args=self.args,
                    dataset=dataset_name,
                    task=task_name,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=False,  
                    phase = self.args.rounds,
                    component = "item_rec",
                    run_type = self.args.run_type
                )
            test_loader_rec = DataLoader(
                dataset=test_data_rec,
                batch_size=self.args.eval_batch_size,
                collate_fn=collator_rec_test,
                shuffle=False,
            )

            self.testloaders_rec.append(test_loader_rec)
            num_samples = len(test_data_rec)
            num_batches = len(test_loader_rec)
            logging.info(f"Created recommender testloader for {dataset_name}, with {num_samples} samples ({num_batches} batches, batch_size={self.args.eval_batch_size})")

                   
    def _test_recommender(self):
        self.get_testloader_item()
        self.model_rec.eval()
        logging.info(f"--- Testing Item Recommender Performance ---")
        for loader in self.testloaders_rec:
            self.test_dataset_task(loader, test_type="item")
        
        self.get_testloader_friend()
        logging.info(f"--- Testing Friend Recommender Performance ---")
        for loader in self.testloaders_social:
            self.test_dataset_task(loader, test_type="friend")
        logging.info("--- Testing Finished ---")

    def test_dataset_task(self, testloader, test_type):        
        test_total = 0
        with torch.no_grad():
            if test_type == "item":
                state_dict = torch.load(self.args.rec_model_path, map_location=self.device)
                config = T5Config.from_pretrained(self.args.backbone)
                rec_model = T5ForConditionalGeneration.from_pretrained(self.args.backbone, config=config)
                if 'shared.weight' in state_dict:
                    vocab_size = state_dict['shared.weight'].shape[0]
                    rec_model.resize_token_embeddings(vocab_size)
                rec_model.load_state_dict(state_dict)
                rec_model.to(self.device)
                rec_model.eval()
                model_to_use = rec_model
                candidates = testloader.dataset.all_items
            elif test_type == "friend":
                state_dict = torch.load(self.args.social_model_path, map_location=self.device)
                config = T5Config.from_pretrained(self.args.backbone)
                social_model = T5ForConditionalGeneration.from_pretrained(self.args.backbone, config=config)
                if 'shared.weight' in state_dict:
                    vocab_size = state_dict['shared.weight'].shape[0]
                    social_model.resize_token_embeddings(vocab_size)
                social_model.load_state_dict(state_dict)
                social_model.to(self.device)
                social_model.eval()
                model_to_use = social_model
                candidates = testloader.dataset.all_users
            candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(candidate)
                    for candidate in candidates
                ]
            )
            logging.info(f"Number of {test_type} candidates: {len(candidates)}")
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
            metrics_res = np.array([0.0] * len(self.metrics))
            use_timestep_in_eval = (getattr(self.args, 'use_diffusion', 0) and 
                                   hasattr(self, 'timestep_token_ids') and 
                                   self.timestep_token_ids is not None)
            if use_timestep_in_eval:
                logging.info("Using timestep t=0 token during evaluation (diffusion enabled)")
            else: 
                logging.info("Not using timestep token during evaluation (diffusion disabled)")
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device, non_blocking=True)
                attn = batch[1].to(self.device, non_blocking=True)
                output_ids = batch[3].to(self.device, non_blocking=True)
                if use_timestep_in_eval:
                    batch_size = input_ids.shape[0]
                    timestep_0 = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                    input_ids, attn = prepend_timestep_token(
                        input_ids, timestep_0, self.timestep_token_ids, attn
                    )   
                prediction = model_to_use.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_length=30,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num,
                        num_return_sequences=self.generate_num,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                rel_results = evaluate.rel_results(generated_sents, gold_sents, prediction_scores, self.generate_num)
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            
            metrics_res /= test_total
            
            for i in range(len(self.metrics)):
                logging.info(f'{self.metrics[i]}: {metrics_res[i]:.3f}')
        
   