# In src/runner/SingleRunner.py
import torch
torch.cuda.empty_cache()
from transformers import AdamW, get_linear_schedule_with_warmup, T5Config, T5ForConditionalGeneration, AutoTokenizer, GenerationConfig
import logging
from tqdm import tqdm
from utils import utils, evaluate # Make sure evaluate methods are compatible
# import utils.generation_trie as gt # If using P5-style constrained decoding for testing
from data.TestDataset import TestDataset, TestDatasetGen  # Ensure both are imported
from torch.utils.data import DataLoader
from processor.Collator import Collator, TestCollator 
import time
import numpy as np
import os
import string # For punctuation removal, as in DistributedRunner_gen.py
from utils.generation_trie import Trie, prefix_allowed_tokens_fn
import utils.generation_trie as gt # For P5-style constrained decoding, if used
from utils import indexing # For generative indexing, if used

# If datasets are reloaded/refreshed in the training loop
from utils.dataset_utils import get_dataset_generative, get_loader
# from types import MethodType # Not needed in runner if done in main


class SingleRunner:
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument("--optim", type=str, default='AdamW', help='The name of the optimizer')
        parser.add_argument("--lr", type=float, default=1e-4, help="Default learning rate (not used if id_lr/rec_lr set)")
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value")
        parser.add_argument("--logging_step", type=int, default=100, help="Log every N steps")
        parser.add_argument("--warmup_prop", type=float, default=0.05, help="Warmup proportion for scheduler")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--adam_eps", type=float, default=1e-6)
        parser.add_argument("--train", type=int, default=1, help='Train or not (1 for train, 0 for no train)')
        parser.add_argument("--backbone", type=str, default='t5-small', help='Default backbone model name')
        parser.add_argument("--id_model_backbone", type=str, default=None, help="Backbone for ID model, defaults to --backbone if None")
        parser.add_argument("--rec_model_backbone", type=str, default=None, help="Backbone for Rec model, defaults to --backbone if None")
        parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer, defaults to --backbone if None")
        parser.add_argument("--metrics", type=str, default='hit@5,hit@10,ndcg@5,ndcg@10', help='Metrics for evaluation')
        parser.add_argument("--load", type=int, default=0, help='Load model from model path (not fully implemented for alternating runner yet)')
        parser.add_argument("--valid_select", type=int, default=0, help='Use validation loss to select models')
        parser.add_argument("--test_before_train", type=int, default=0, help='Whether to test before training starts')
        parser.add_argument("--max_output_len", type=int, default=20, help="Max length for generated sequences during testing")
        parser.add_argument("--num_workers", type=int, default=0, help="Num workers for dataloader")
        parser.add_argument("--test_filtered", type=int, default=0, help='whether filter out the items in the training data.')
        parser.add_argument("--test_filtered_batch", type=int, default=1, help='whether testing with filtered data in batch.')


        return parser

    def __init__(self, model_rec, model_gen, tokenizer, 
                 train_loader_id, train_loader_rec, 
                 valid_loader, # This will be None with --valid_select 0
                 device, args):
        
        self.model_rec = model_rec.to(device) if model_rec else None
        self.model_gen = model_gen.to(device) if model_gen else None
        self.tokenizer = tokenizer
        self.train_loader_id = train_loader_id
        self.train_loader_rec = train_loader_rec
        self.valid_loader = valid_loader 
        self.device = device
        self.args = args
        
        self.num_alternations = args.rounds
        self.global_epoch_tracker = 0
        self.total_id_epoch = 0
        self.test_filtered = args.test_filtered
        self.test_filtered_batch = args.test_filtered_batch

        if args.train: # Only create if training
            self.id_optimizer, self.id_scheduler, \
            self.rec_optimizer, self.rec_scheduler = self.create_optimizer_and_scheduler_2()
        self.get_testloader() # Prepare test loaders once

        self.metrics = args.metrics.split(',')
        self.generate_num = max([int(m.split('@')[1]) for m in self.metrics if '@' in m and len(m.split('@')) > 1] or [10])
        print("SingleRunner using device:", self.device)

    def _create_optimizers_for_alternating(self):
        # This function creates separate optimizers and learning rate schedulers for both
        # the ID generator (model_gen) and the recommender (model_rec) models.
        # It groups parameters for weight decay, calculates total and warmup steps
        # based on dataloader length, epochs, and alternations, and returns optimizers
        # and schedulers for alternating training phases.
        logging.info("Building Optimizers and Schedulers for Alternating Training (Single GPU)")
        optimizer_id, scheduler_id = None, None
        optimizer_rec, scheduler_rec = None, None
        no_decay = ["bias", "LayerNorm.weight"]

        if self.model_gen and self.train_loader_id:
            if not list(self.model_gen.parameters()):
                logging.warning("ID Generator model has no parameters!")
            else:
                batch_per_epoch_id = len(self.train_loader_id)
                id_total_steps = batch_per_epoch_id // self.args.gradient_accumulation_steps * self.args.id_epochs * self.num_alternations
                id_warmup_steps = int(id_total_steps * self.args.warmup_prop)
                logging.info(f'ID Gen Config: Batches/epoch: {batch_per_epoch_id}, ID Epochs/round: {self.args.id_epochs}, Rounds: {self.num_alternations}')
                logging.info(f'ID Gen Config: Total steps: {id_total_steps}, Warmup steps: {id_warmup_steps}, LR: {self.args.id_lr}')
                
                optimizer_grouped_parameters_id = [
                    {"params": [p for n, p in self.model_gen.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": self.args.weight_decay},
                    {"params": [p for n, p in self.model_gen.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
                ]
                if sum(len(group["params"]) for group in optimizer_grouped_parameters_id) > 0:
                    optimizer_id = AdamW(optimizer_grouped_parameters_id, lr=self.args.id_lr, eps=self.args.adam_eps)
                    scheduler_id = get_linear_schedule_with_warmup(optimizer_id, id_warmup_steps, id_total_steps)
                else:
                    logging.warning("ID Generator has no parameters requiring gradients for the optimizer.")
        else:
            logging.info("ID Generator or its train_loader not available, skipping optimizer setup for ID Gen.")

        if self.model_rec and self.train_loader_rec:
            if not list(self.model_rec.parameters()):
                 logging.warning("Recommender model has no parameters!")
            else:
                batch_per_epoch_rec = len(self.train_loader_rec)
                rec_total_steps = batch_per_epoch_rec // self.args.gradient_accumulation_steps * self.args.rec_epochs * self.num_alternations
                rec_warmup_steps = int(rec_total_steps * self.args.warmup_prop)
                logging.info(f'Recommender Config: Batches/epoch: {batch_per_epoch_rec}, Rec Epochs/round: {self.args.rec_epochs}, Rounds: {self.num_alternations}')
                logging.info(f'Recommender Config: Total steps: {rec_total_steps}, Warmup steps: {rec_warmup_steps}, LR: {self.args.rec_lr}')

                optimizer_grouped_parameters_rec = [
                    {"params": [p for n, p in self.model_rec.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": self.args.weight_decay},
                    {"params": [p for n, p in self.model_rec.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
                ]
                if sum(len(group["params"]) for group in optimizer_grouped_parameters_rec) > 0:
                    optimizer_rec = AdamW(optimizer_grouped_parameters_rec, lr=self.args.rec_lr, eps=self.args.adam_eps)
                    scheduler_rec = get_linear_schedule_with_warmup(optimizer_rec, rec_warmup_steps, rec_total_steps)
                else:
                    logging.warning("Recommender has no parameters requiring gradients for the optimizer.")
        else:
            logging.info("Recommender or its train_loader not available, skipping optimizer setup for Rec.")
            
        return optimizer_id, scheduler_id, optimizer_rec, scheduler_rec

    def create_optimizer_and_scheduler_2(self):
            if self.args.rank == 0:
                logging.info("Building Optimizer and Scheduler")
            batch_per_epoch_id = len(self.train_loader_id)
            batch_per_epoch_rec = len(self.train_loader_rec)
            id_total_steps = batch_per_epoch_id // self.args.gradient_accumulation_steps * self.args.id_epochs * self.num_alternations
            id_warmup_steps = int(id_total_steps * self.args.warmup_prop)
            
            rec_total_steps = batch_per_epoch_rec // self.args.gradient_accumulation_steps * self.args.rec_epochs * self.num_alternations
            rec_warmup_steps = int(rec_total_steps * self.args.warmup_prop)
            if self.args.rank == 0:
                logging.info(f'Batch per epoch id: {batch_per_epoch_id}')
                logging.info(f'Warmup proportion: {self.args.warmup_prop}')
                logging.info(f'Total id generator steps: {id_total_steps}')
                logging.info(f'Warm up id generator steps: {id_warmup_steps}')
                logging.info(f'Batch per epoch rec: {batch_per_epoch_rec}')
                logging.info(f'Warmup proportion: {self.args.warmup_prop}')
                logging.info(f'Total rec generator steps: {rec_total_steps}')
                logging.info(f'Warm up rec generator steps: {rec_warmup_steps}')

            no_decay = ["bias", "LayerNorm.weight"]
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

            if self.args.rank == 0:
                logging.info(f"Building Optimizer {self.args.optim}")

            if self.args.optim.lower() == 'adamw':
                optimizer_id = AdamW(optimizer_grouped_parameters_id, lr=self.args.id_lr, eps=self.args.adam_eps)
                optimizer_rec = AdamW(optimizer_grouped_parameters_rec, lr=self.args.rec_lr, eps=self.args.adam_eps)
            else:
                raise NotImplementError
            scheduler_id = get_linear_schedule_with_warmup(optimizer_id, id_warmup_steps, id_total_steps)
            scheduler_rec = get_linear_schedule_with_warmup(optimizer_rec, rec_warmup_steps, rec_total_steps)

            return optimizer_id, scheduler_id, optimizer_rec, scheduler_rec
    
    def torch_isin(elements: torch.Tensor, test_elements: torch.Tensor) -> torch.Tensor:
        """
        Compatibility wrapper for torch.isin().
        Returns a boolean tensor indicating whether each element in `elements` is in `test_elements`.

        Works in PyTorch < 1.10 where torch.isin is not available.
        """
        if hasattr(torch, 'isin'):
            return torch.isin(elements, test_elements)
        else:
            # Broadcasted comparison and reduction
            return (elements[..., None] == test_elements).any(-1)
    def _train_id_generator_phase(self, current_round_num):
        """
        Trains the ID Generator (self.model_gen) for self.args.id_epochs.
        Adapted from the ID Gen training loop in DistributedRunner_gen.py.
        This version implements the end-to-end training: model_gen creates ID embeddings,
        model_rec processes them, and model_rec's loss updates model_gen.
        """
        if not (self.model_gen and self.train_loader_id and self.id_optimizer and self.model_rec):
            logging.warning("Skipping ID Generator training phase: model_gen, model_rec, train_loader_id, or id_optimizer missing.")
            return

        logging.info(f"--- Round {current_round_num + 1}: Training ID Generator ---")
        for param in self.model_gen.parameters(): param.requires_grad = True
        for param in self.model_rec.parameters(): param.requires_grad = False # Freeze recommender's weights for this phase
        
        self.model_gen.train()
        self.model_rec.train() 

        for id_epoch in range(self.args.id_epochs):
            logging.info(f"ID Gen - Round {current_round_num + 1}, Epoch {id_epoch + 1}/{self.args.id_epochs} (Overall step context: {self.global_epoch_tracker})")
            self.train_loader_id.sampler.set_epoch(self.global_epoch_tracker)
            
            
            epoch_losses = []
            for batch_idx, batch in enumerate(self.train_loader_id):
                input_prompt_ids = batch[0].to(self.device) #partially filled in, tokenized prompt (replace {history} with semicolons)
                input_prompt_positions = batch[1].to(self.device) # Binary mask where history items are inserted
                hist_ids = batch[2].to(self.device) # Raw item IDs for model_gen to convert to textual IDs
                hist_att = batch[3].to(self.device) # Attention mask where history items are inserted
                output_ids = batch[4].to(self.device) # Target sequence for model_rec (e.g., next item's textual ID)

                batch_size = hist_ids.shape[0]
                hist_size = hist_ids.shape[1] # Number of history items per user

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
                            early_stopping=True,
                        )

                probabilities = torch.cat([score.unsqueeze(1) for score in output['scores']], dim=1)
                
                train_id_token_size = probabilities.shape[1]
        
                token_embeddings = self.model_rec.shared.weight  # use rec models' embedding as input
                hist_embeddings = torch.einsum('bsv,ve->bse', probabilities, token_embeddings)
                hist_embeddings = hist_embeddings.view(batch_size, hist_size, train_id_token_size, -1)  # [bs, hist_size, id_token_size, xxx]

                # Remove punctuation embeddings
                temp_ids = output['sequences'][:, 1:]

                punctuation_tokens = [self.tokenizer.encode(p, add_special_tokens=False)[0] for p in string.punctuation]

                punctuation_tokens_tensor = torch.tensor(punctuation_tokens).to(self.device)
                punctuation_mask = SingleRunner.torch_isin(temp_ids, punctuation_tokens_tensor)
                # reshape
                batch_size_, hist_size_, seq_length_minus_one_, embedding_dim_ = hist_embeddings.shape
                punctuation_mask = punctuation_mask.view(batch_size_, hist_size_, seq_length_minus_one_)

                hist_embeddings[punctuation_mask.unsqueeze(-1).expand_as(hist_embeddings)] = 0

                input_prompt_embeddings = token_embeddings[input_prompt_ids]

                # calculate the max sequence size
                max_prompt_size = input_prompt_embeddings.shape[1]
                max_hist_num = hist_ids.shape[1]
                max_input_len = max_prompt_size + max_hist_num * train_id_token_size
                final_input = self.insert_phrases_batch(input_prompt_embeddings, 
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

                # compute loss masking padded tokens
                loss = output["loss"]

                # update
                loss.backward()
                loss = loss / self.args.gradient_accumulation_steps

                
                torch.nn.utils.clip_grad_norm_(self.model_gen.parameters(), self.args.clip)
                #update the model_gen parameters using the optimizer
                self.id_optimizer.step()
                # update the learning rate scheduler for the ID optimizer if it exits
                self.id_scheduler.step()
                #clear gradients for the next step (clear after each optimization step)
                self.model_gen.zero_grad()
                self.model_rec.zero_grad()

                #store the loss of each batch
                epoch_losses.append(loss.item() * self.args.gradient_accumulation_steps)
                if (batch_idx + 1) % self.args.logging_step == 0:
                    # log the current learning rate and loss
                     current_lr = self.id_optimizer.param_groups[0]['lr'] if self.id_optimizer else 0
                     logging.info(f"ID Gen - Round {current_round_num+1} Epoch {id_epoch+1}, Batch {batch_idx+1}/{len(self.train_loader_id)}: Loss {epoch_losses[-1]:.4f} LR: {current_lr:.2e}")
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('nan')
            logging.info(f"ID Gen - Avg Loss Round {current_round_num+1} Epoch {id_epoch+1}: {avg_epoch_loss:.4f}")

            self.global_epoch_tracker += 1
            self.total_id_epoch +=1

    def _train_recommender_phase(self, current_round_num):
        """
        Trains the Recommender (self.model_rec) for self.args.rec_epochs.
        This is a more standard sequence-to-sequence training loop using textual IDs.
        """
        if not (self.model_rec and self.train_loader_rec and self.rec_optimizer):
            logging.warning("Skipping Recommender training phase: model, loader, or optimizer missing.")
            return

        logging.info(f"--- Round {current_round_num + 1}: Training Recommender ---")
        if self.model_gen:
            for param in self.model_gen.parameters(): param.requires_grad = False
            self.model_gen.eval()
        for param in self.model_rec.parameters(): param.requires_grad = True
        self.model_rec.train()
        self.model_gen.train()  
        current_phase_for_rec_dataset = self.total_id_epoch
        logging.info(f"Recommender training (Round {current_round_num+1}): Refreshing dataset/loader for phase {current_phase_for_rec_dataset}")

        # Refresh the dataset using the newly generated ID files
        refreshed_TrainSetID, refreshed_TrainSetRec, refreshed_ValidSet = get_dataset_generative(
            self.args, self.model_gen, self.tokenizer, 
            #regenerate=False, 
            phase=current_phase_for_rec_dataset
        )

        # with open("refreshed_TrainSetRec.txt", "w", encoding="utf-8") as f:
        #     for i, item in enumerate(refreshed_TrainSetRec):
        #         f.write(f"Item {i}: {str(item)}\n") # this looks right
        if refreshed_TrainSetRec:
            _, self.train_loader_rec, _ = get_loader(self.args, self.tokenizer, refreshed_TrainSetID, refreshed_TrainSetRec, None)
            
            logging.info(f"Recommender train_loader_rec refreshed. New length: {len(self.train_loader_rec) if self.train_loader_rec else 0}")
        else:
            logging.warning("Failed to refresh TrainSetRec for recommender phase. Using existing or aborting if None.")
            if not self.train_loader_rec:
                return
      
        tokenizer = self.tokenizer  

        for rec_epoch in range(self.args.rec_epochs):
            logging.info(f"Recommender - Round {current_round_num + 1}, Epoch {rec_epoch + 1}/{self.args.rec_epochs} (Overall step context: {self.global_epoch_tracker})")
            self.train_loader_rec.sampler.set_epoch(self.global_epoch_tracker)
            # with open("rec_train_debug_batch.txt", "w", encoding="utf-8") as f:
            #     f.write("")
            epoch_losses = []
            for batch_idx, batch in enumerate(self.train_loader_rec):
                input_ids = batch[0].to(self.device)
                attn_mask = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)  
                output_ids = batch[3].to(self.device)

                # Decode inputs and targets
                # decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                # decoded_targets = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                # # Write to file in append mode
                # with open("rec_train_debug_batch.txt", "a", encoding="utf-8") as f:
                #     for i in range(len(decoded_inputs)):
                #         f.write(f"[Batch {batch_idx + 1} - Sample {i + 1}]\n")
                #         f.write(f"Input Prompt:  {decoded_inputs[i]}\n")
                #         f.write(f"Target Output: {decoded_targets[i]}\n")
                #         f.write("=" * 80 + "\n")

                # Compute loss
                output = self.model_rec(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=output_ids, 
                    return_dict=True,
                )
                loss = output.loss
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                
                torch.nn.utils.clip_grad_norm_(self.model_rec.parameters(), self.args.clip)
                self.rec_optimizer.step()
                self.rec_scheduler.step()
                self.model_rec.zero_grad()
                self.model_gen.zero_grad()  

                epoch_losses.append(loss.item() * self.args.gradient_accumulation_steps)
                if (batch_idx + 1) % self.args.logging_step == 0:
                    current_lr = self.rec_optimizer.param_groups[0]['lr'] if self.rec_optimizer else 0
                    logging.info(f"Rec - Epoch {rec_epoch+1}, Batch {batch_idx+1}/{len(self.train_loader_rec)}: Loss {epoch_losses[-1]:.4f} LR: {current_lr:.2e}")

            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('nan')
            logging.info(f"Rec - Avg Loss Round {current_round_num+1} Epoch {rec_epoch+1}: {avg_epoch_loss:.4f}")

            self.global_epoch_tracker += 1
            
            # if self.args.test_epoch_rec > 0 and (rec_epoch + 1) % self.args.test_epoch_rec == 0:
            #     self._test_recommender(current_round_num, rec_epoch)

    def train(self):
        if not self.args.train:
            logging.info("Training disabled by args.train=0.")
            if self.args.test_before_train > 0: self._test_both_models()
            return

        logging.info("Starting training process...")
        if self.args.test_before_train > 0:
            self._test_both_models()
        
        for round_num in range(self.num_alternations):
            logging.info(f"========== Starting Alternation Round {round_num + 1}/{self.num_alternations} ==========")
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
            logging.info(f"========== Finished Alternation Round {round_num + 1}/{self.num_alternations} ==========")
            if self.args.model_path: 
                if self.model_gen:
                    gen_path = os.path.join(self.args.model_path, f"model_gen_round{round_num+1}_final.pt")
                    torch.save(self.model_gen.state_dict(), gen_path)
                    logging.info(f"Saved ID Gen model to {gen_path}")
                if self.model_rec:
                    rec_path = os.path.join(self.args.model_path, f"model_rec_round{round_num+1}_final.pt")
                    torch.save(self.model_rec.state_dict(), rec_path)
                    logging.info(f"Saved Rec model to {rec_path}")
            self.global_epoch_tracker += 1
        logging.info("--- Alternating Training Finished ---")
        self._test_both_models()

    def get_testloader(self):
        self.testloaders_rec = []
        datasets_to_test = self.args.datasets.split(',')
        tasks_to_test = self.args.tasks.split(',')
        if self.test_filtered > 0:
            collator_rec_test = TestCollator(self.tokenizer)
        else:
            collator_rec_test = Collator(self.tokenizer)

        for dataset_name in datasets_to_test:
            for task_name in tasks_to_test:
                if task_name == 'sequential':
                    try:
                        if self.args.item_indexing == 'generative':
                            test_data_rec = TestDatasetGen(
                                args=self.args,
                                dataset=dataset_name,
                                task=task_name,
                                model_gen=self.model_gen,
                                tokenizer=self.tokenizer,
                                regenerate=True,  
                                phase = 0,                             
                            )
                        else:
                            test_data_rec = TestDataset(
                                args=self.args,
                                dataset=dataset_name,
                                task=task_name
                            )

                        test_loader_rec = DataLoader(
                            dataset=test_data_rec,
                            batch_size=self.args.eval_batch_size,
                            collate_fn=collator_rec_test,
                            shuffle=False,
                        )

                        self.testloaders_rec.append(test_loader_rec)
                        logging.info(f"Created recommender testloader for {dataset_name}, task {task_name} with {len(test_loader_rec)} test sets.")
                        # with open("testloader_rec_batches.txt", "w") as f:
                        #     for i, batch in enumerate(test_loader_rec):
                        #         input_str = self.tokenizer.decode(batch[0][0], skip_special_tokens=True)
                        #         output_str = self.tokenizer.decode(batch[3][0], skip_special_tokens=True)
                        #         f.write(f"Batch {i}:\n")
                        #         f.write(f"Decoded Input Prompt: {input_str}\n")
                        #         f.write(f"Decoded Target Item: {output_str}\n\n")

                    except Exception as e:
                        import traceback
                        logging.error(f"Error creating recommender testloader for {dataset_name}, task {task_name}: {e}")
                        traceback.print_exc()

    def _test_recommender(self, current_round_num=-1, current_epoch_num=-1):
        if not (self.model_rec and self.testloaders_rec):
            logging.info("Recommender model or test loaders not available for testing.")
            return

        self.model_rec.eval()
        logging.info(f"--- Testing Recommender Performance (Round {current_round_num+1}, Epoch {current_epoch_num+1}) ---")
        
        for loader in self.testloaders_rec:
            #if self.test_filtered > 0 :
            #    logging.info(f"Testing with filtered data in batch mode for {loader.dataset.dataset} dataset, {loader.dataset.task} task")
            #self._perform_rec_test_on_loader(loader, self.model_rec)
            self.test_dataset_task(loader)
    def _test_both_models(self):
        logging.info("--- Final Test Phase (or Intermittent Test) ---")
        self.get_testloader()
        self._test_recommender()
    #SAME
    def test_dataset_task(self, testloader):
        
        logging.info(f'testing {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        test_total = 0
        with torch.no_grad():
            candidates = testloader.dataset.all_items
            candidate_trie = gt.Trie(
                [
                    #[0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    [0] + self.tokenizer.encode(candidate)
                    for candidate in candidates
                ]
            )
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
            
            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                
                prediction = self.model_rec.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        #whole_word_ids=whole_input_ids,
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
                #logging.info(f"Gold sentences: {gold_sents}")
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                #logging.info(f"Generated sentences: {generated_sents}")
                # print(generated_sents)
                # exit()
                rel_results = evaluate.rel_results(generated_sents, gold_sents, prediction_scores, self.generate_num)
                #logging.info(f"Relevant results: {rel_results}")
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            
            metrics_res /= test_total
            
            for i in range(len(self.metrics)):
                logging.info(f'{self.metrics[i]}: {metrics_res[i]}')
    #SAME
    def insert_phrases_batch(self, prompt, positions, hist, max_input_len):
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
            pad_tensor = torch.zeros((pad_size, emb_size)).to(self.device)
            result_tensor = torch.cat([result_tensor, pad_tensor], dim=0)
            
            batch_results.append(result_tensor)

        # Concatenate batch_results to get final tensor
        final_tensor = torch.stack(batch_results, dim=0)
        
        return final_tensor


