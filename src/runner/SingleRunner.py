import torch
import torch.nn.functional as F
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
    compute_kl_divergence,
    compute_kl_divergence_from_probs
)


class SingleRunner:

    def __init__(self, model_rec=None, model_gen=None, model_social=None, tokenizer=None, 
                 train_loader_id=None, train_loader_rec=None, train_loader_rec_social=None, train_loader_social=None,
                 valid_loader=None, device=None, args=None, component=None, other_view_model=None):
        self.model_rec = model_rec.to(device) if model_rec else None
        self.model_gen = model_gen.to(device) if model_gen else None
        self.model_social = model_social.to(device) if model_social else None
        self.other_view_model = other_view_model.to(device) if other_view_model else None
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
        self.metrics = args.metrics.split(',')
        self.generate_num = max([int(m.split('@')[1]) for m in self.metrics if '@' in m and len(m.split('@')) > 1] or [10])
        punctuation_tokens = [self.tokenizer.encode(p, add_special_tokens=False)[0] for p in string.punctuation]
        self.punctuation_tokens_tensor = torch.tensor(punctuation_tokens, device=self.device)        
        self.use_diffusion = getattr(args, 'use_diffusion', 0)
        self.noise_head_rec = None
        self.noise_head_social = None
        self.id_optimizer, self.id_scheduler, self.rec_optimizer, self.rec_scheduler = self.create_optimizer_and_scheduler()
        if self.use_diffusion:
            self.timestep_token_ids = add_timestep_tokens_to_tokenizer(self.tokenizer, self.args.diffusion_timesteps)
            self.model_rec.resize_token_embeddings(len(self.tokenizer))
            self.model_gen.resize_token_embeddings(len(self.tokenizer))
            self.model_social.resize_token_embeddings(len(self.tokenizer)) if self.model_social is not None else None
            logging.info(f"Added timestep tokens: T={self.args.diffusion_timesteps}, vocab_size={len(self.tokenizer)}")
            if self.args.train:
                noise_head_dropout = self.args.noise_head_dropout
                self.noise_head_rec = create_noise_head(self.model_rec, dropout=noise_head_dropout).to(self.device)
                self.noise_head_social = create_noise_head(self.model_social, dropout=noise_head_dropout).to(self.device) if self.model_social is not None else None
                logging.info("Created noise prediction heads for recommender models")
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
            # Add noise head parameters if diffusion is enabled
            if self.use_diffusion and self.noise_head_rec is not None:
                optimizer_grouped_parameters_rec[0]["params"].extend([
                    p for n, p in self.noise_head_rec.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ])
                optimizer_grouped_parameters_rec[1]["params"].extend([
                    p for n, p in self.noise_head_rec.named_parameters()
                    if any(nd in n for nd in no_decay)
                ])
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
            for batch in self.train_loader_id:
                input_prompt_ids = batch[0].to(self.device, non_blocking=True) 
                input_prompt_positions = batch[1].to(self.device, non_blocking=True)
                hist_ids = batch[2].to(self.device, non_blocking=True)
                hist_att = batch[3].to(self.device, non_blocking=True)
                output_ids = batch[4].to(self.device, non_blocking=True)
                batch_size = hist_ids.shape[0]
                hist_size = hist_ids.shape[1]
                input_tensor = hist_ids.view(-1, hist_ids.shape[-1])
                hist_att_flat = hist_att.view(-1, hist_att.shape[-1])
                
                # Generate with gradient tracking
                output = self.model_gen.generate_with_grad(
                            input_tensor,
                            attention_mask=hist_att_flat, 
                            max_length=10,
                            min_length=1,
                            num_beams=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                            output_hidden_states=False,
                            renormalize_logits=True,
                        )
                # Memory-efficient processing: compute embeddings timestep by timestep
                # instead of concatenating all scores first
                scores = output['scores']
                train_id_token_size = len(scores)
                token_embeddings = self.model_rec.shared.weight
                embedding_dim = token_embeddings.shape[1]
                
                # Process each timestep separately to reduce peak memory
                hist_embeddings_list = []
                temp_ids = output['sequences'][:, 1:]  # [batch*hist_size, seq_len-1]
                
                for t in range(train_id_token_size):
                    # Get logits for this timestep: [batch*hist_size, vocab_size]
                    logits_t = scores[t]
                    
                    # Compute embeddings using matmul (more memory efficient than einsum)
                    # matmul is equivalent to einsum('bsv,ve->bse') for single timestep
                    embeds_t = torch.matmul(logits_t, token_embeddings)  # [batch*hist_size, embedding_dim]
                    hist_embeddings_list.append(embeds_t)
                    #del logits_t  # Clear logits, but keep embeds_t in list for stacking
                
                # Stack along sequence dimension: [batch*hist_size, seq_len, embedding_dim]
                hist_embeddings_flat = torch.stack(hist_embeddings_list, dim=1)
                #del hist_embeddings_list
                
                # Reshape to [batch_size, hist_size, seq_len, embedding_dim]
                hist_embeddings = hist_embeddings_flat.view(batch_size, hist_size, train_id_token_size, embedding_dim)
                #del hist_embeddings_flat
                
                # Apply punctuation mask more efficiently (avoid expand_as)
                punctuation_mask = utils.torch_isin(temp_ids, self.punctuation_tokens_tensor)
                punctuation_mask = punctuation_mask.view(batch_size, hist_size, train_id_token_size)
                # Use broadcasting instead of expand_as to avoid large temporary tensor
                # Set to 0 where punctuation_mask is True (punctuation tokens)
                hist_embeddings = hist_embeddings * (~punctuation_mask).unsqueeze(-1).float()
                #del punctuation_mask, temp_ids
                
                # Get prompt embeddings
                input_prompt_embeddings = token_embeddings[input_prompt_ids]
                max_prompt_size = input_prompt_embeddings.shape[1]
                max_hist_num = hist_ids.shape[1] 
                max_input_len = max_prompt_size + max_hist_num * train_id_token_size 
                
                # Insert phrases (memory-intensive, but necessary)
                final_input = utils.insert_phrases_batch(input_prompt_embeddings, input_prompt_positions, hist_embeddings, max_input_len)
                
                # Compute attention mask more efficiently
                # Use sum of squares instead of norm to avoid sqrt computation
                norms_sq = (final_input ** 2).sum(dim=-1)  # [batch_size, max_input_len]
                attention_mask = (norms_sq > 1e-12).long()  # Equivalent to norm > 1e-6
                #del norms_sq
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
                
                # Store loss before cleanup
                loss_value = loss.item()
                epoch_losses.append(loss_value)
                
                # Incremental memory cleanup to smooth out memory fluctuations
                # Delete tensors in stages to avoid sudden memory drops
                # Stage 1: Clear generation outputs (largest tensors first)
                # del output, loss
                # del scores, hist_embeddings  # scores from generation, hist_embeddings from processing
                
                # # Stage 2: Clear intermediate computations
                # # Note: temp_ids and punctuation_mask already deleted earlier (line 226)
                # del input_tensor, hist_att_flat
                # del input_prompt_embeddings, final_input, attention_mask
                
                # # Stage 3: Clear batch reference (DataLoader will handle prefetched batches)
                # del batch
                
                # Only clear cache when memory pressure is high (less frequent, smoother)
                # This reduces sudden memory drops while still preventing OOM
                # if len(epoch_losses) % 10 == 0:
                #     torch.cuda.empty_cache()
            
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
        if self.noise_head_rec is not None:
            for param in self.noise_head_rec.parameters(): param.requires_grad = True if self.use_diffusion else False
            self.noise_head_rec.train()
        self.model_rec.train()
        self.model_gen.train()

        current_phase_for_rec_dataset = current_round_num + 1
        logging.info(f"Recommender training (Round {current_round_num+1}): Refreshing dataset/loader for phase {current_phase_for_rec_dataset}")
        _, refreshed_TrainSetRec = get_dataset_generative(self.args, self.model_gen, self.tokenizer, phase=current_phase_for_rec_dataset, component=self.component)
        _, self.train_loader_rec = get_loader(self.args, self.tokenizer, None, refreshed_TrainSetRec) 
        for rec_epoch in range(self.args.rec_epochs):
            logging.info(f"Recommender - Round {current_round_num + 1}, Epoch {rec_epoch + 1}/{self.args.rec_epochs}")
            self.train_loader_rec.sampler.set_epoch(self.global_epoch_tracker)
            epoch_losses_ce = []
            epoch_losses_bce = []
            epoch_losses_kl = []
            corruption_rates = []
            avg_timesteps = []
            
            for batch in self.train_loader_rec:
                input_ids = batch[0].to(self.device, non_blocking=True)
                attn_mask = batch[1].to(self.device, non_blocking=True)
                output_ids = batch[3].to(self.device, non_blocking=True)
                if self.use_diffusion and len(batch) > 5:
                    timesteps = batch[5].to(self.device, non_blocking=True)  # [batch_size]
                    noise_masks = batch[6].to(self.device, non_blocking=True)  # [batch_size, seq_len]
                    input_ids, attn_mask = prepend_timestep_token(
                        input_ids, timesteps, self.timestep_token_ids, attn_mask
                    )
                token_embeddings = self.model_rec.shared.weight
                input_embeds = token_embeddings[input_ids]               
                # Forward for noise prediction head
                encoder_outputs = self.model_rec.encoder(
                    inputs_embeds=input_embeds,
                    attention_mask=attn_mask,
                    return_dict=True
                )
                encoder_hidden_states = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
                
                # Standard CE loss for next-ID prediction
                output = self.model_rec(
                    inputs_embeds=input_embeds,
                    attention_mask=attn_mask,
                    labels=output_ids,
                    return_dict=True,
                )
                loss_ce = output.loss
                
                # Noise mask prediction (BCE loss) if diffusion is enabled
                loss_bce = torch.tensor(0.0, device=self.device)
                if self.use_diffusion and self.noise_head_rec is not None and noise_masks is not None:
                    # Get noise logits from noise head
                    noise_logits = self.noise_head_rec(encoder_hidden_states)  # [batch_size, seq_len]
                    
                    # Align noise_masks with noise_logits (account for prepended timestep token)
                    if timesteps is not None:
                        # noise_masks doesn't include timestep token, so we need to align
                        # noise_logits has timestep token at position 0, noise_masks doesn't
                        noise_logits_aligned = noise_logits[:, 1:]  # Remove timestep token position
                        # Ensure same length
                        min_len = min(noise_logits_aligned.shape[1], noise_masks.shape[1])
                        noise_logits_aligned = noise_logits_aligned[:, :min_len]
                        noise_masks_aligned = noise_masks[:, :min_len]
                    else:
                        min_len = min(noise_logits.shape[1], noise_masks.shape[1])
                        noise_logits_aligned = noise_logits[:, :min_len]
                        noise_masks_aligned = noise_masks[:, :min_len]
                    
                    # Compute BCE loss
                    loss_bce = F.binary_cross_entropy_with_logits(
                        noise_logits_aligned,
                        noise_masks_aligned.float(),
                        reduction='mean'
                    )
                
                # KL divergence loss: KL(p_current || p_other)
                loss_kl = torch.tensor(0.0, device=self.device)
                # Check if we have access to the other view model (either model_social in same runner or other_view_model from different runner)
                other_model = self.model_social if self.model_social is not None else self.other_view_model
                
                if self.use_diffusion and other_model is not None:
                    # Get logits from current view (already computed above)
                    current_logits = output.logits  # [batch_size, output_seq_len, vocab_size]
                    
                    # Get other view predictions for the same input
                    # Note: other_model needs to be in training mode if we want gradients
                    other_output = other_model(
                        inputs_embeds=input_embeds,
                        attention_mask=attn_mask,
                        labels=output_ids,
                        return_dict=True,
                    )
                    other_logits = other_output.logits  # [batch_size, output_seq_len, vocab_size]
                    
                    # Align sequence lengths if needed
                    min_seq_len = min(current_logits.shape[1], other_logits.shape[1])
                    current_logits_aligned = current_logits[:, :min_seq_len, :]  # [batch_size, min_seq_len, vocab_size]
                    other_logits_aligned = other_logits[:, :min_seq_len, :]  # [batch_size, min_seq_len, vocab_size]
                    
                    # Compute KL divergence: KL(p_current || p_other)
                    # For item view: KL(p_item || p_social)
                    # For social view: KL(p_social || p_item)
                    # Using compute_kl_divergence which handles logits directly
                    loss_kl = compute_kl_divergence(
                        logits_p=current_logits_aligned,  # Current view (P)
                        logits_q=other_logits_aligned,  # Other view (Q)
                        temperature=1.0,
                        epsilon=1e-8,
                        reduction='mean'
                    )
                
                # Combine losses
                lambda_mask = getattr(self.args, 'lambda_mask', 0.1)
                lambda_kl = getattr(self.args, 'lambda_kl', 0.1)
                loss = loss_ce + lambda_mask * loss_bce + lambda_kl * loss_kl
                
                loss.backward()
                
                # Gradient clipping (include noise head if present)
                params_to_clip = list(self.model_rec.parameters())
                if self.use_diffusion and self.noise_head_rec is not None:
                    params_to_clip.extend(self.noise_head_rec.parameters())
                torch.nn.utils.clip_grad_norm_(params_to_clip, self.args.clip)
                
                self.rec_optimizer.step()
                self.rec_scheduler.step()
                self.model_rec.zero_grad()
                self.model_gen.zero_grad()
                if self.use_diffusion and self.noise_head_rec is not None:
                    self.noise_head_rec.zero_grad()

                epoch_losses_ce.append(loss_ce.item())
                if self.use_diffusion:
                    epoch_losses_bce.append(loss_bce.item())
                    epoch_losses_kl.append(loss_kl.item())
                    if noise_masks is not None:
                        corruption_rates.append(noise_masks.float().mean().item())
                    if timesteps is not None:
                        avg_timesteps.append(timesteps.float().mean().item())

            avg_epoch_loss_ce = sum(epoch_losses_ce) / len(epoch_losses_ce) if epoch_losses_ce else float('nan')
            logging.info(f"Rec - Avg Loss Round {current_round_num+1} Epoch {rec_epoch+1}: CE={avg_epoch_loss_ce:.4f}")
            
            log_dict = {
                "rec/loss": avg_epoch_loss_ce
            }
            
            if self.use_diffusion:
                if epoch_losses_bce:
                    avg_epoch_loss_bce = sum(epoch_losses_bce) / len(epoch_losses_bce)
                    log_dict["diffusion/loss_bce"] = avg_epoch_loss_bce
                    logging.info(f"Rec - BCE Loss: {avg_epoch_loss_bce:.4f}")
                if epoch_losses_kl:
                    avg_epoch_loss_kl = sum(epoch_losses_kl) / len(epoch_losses_kl)
                    log_dict["diffusion/loss_kl"] = avg_epoch_loss_kl
                    logging.info(f"Rec - KL Loss: {avg_epoch_loss_kl:.4f}")
                if corruption_rates:
                    log_dict["diffusion/corruption_rate"] = sum(corruption_rates) / len(corruption_rates)
                if avg_timesteps:
                    log_dict["diffusion/avg_timestep"] = sum(avg_timesteps) / len(avg_timesteps)
            
            wandb.log(log_dict)
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
                # Save social model if it exists
                if self.model_social is not None:
                    social_path = os.path.join(self.args.model_path, f"model_social_round{round_num+1}_final.pt")
                    torch.save(self.model_social.state_dict(), social_path)
                    logging.info(f"Saved social model: {social_path}")
                # Save noise heads if diffusion is enabled (optional - can be recreated)
                if self.use_diffusion:
                    if self.noise_head_rec is not None:
                        noise_head_rec_path = os.path.join(self.args.model_path, f"noise_head_rec_round{round_num+1}_final.pt")
                        torch.save(self.noise_head_rec.state_dict(), noise_head_rec_path)
                        logging.info(f"Saved noise head (rec): {noise_head_rec_path}")
                    if self.noise_head_social is not None:
                        noise_head_social_path = os.path.join(self.args.model_path, f"noise_head_social_round{round_num+1}_final.pt")
                        torch.save(self.noise_head_social.state_dict(), noise_head_social_path)
                        logging.info(f"Saved noise head (social): {noise_head_social_path}")
            self.global_epoch_tracker += 1
        logging.info("--- Alternating Training Finished ---")

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
                    current_vocab_size = len(self.tokenizer)
                    # Handle vocabulary size mismatch (e.g., timestep tokens added)
                    if vocab_size > current_vocab_size:
                        # Check if difference matches expected timestep tokens
                        num_timesteps = getattr(self.args, 'diffusion_timesteps', 100)
                        expected_diff = num_timesteps
                        if vocab_size - current_vocab_size == expected_diff:
                            logging.info(f"Detected timestep tokens in checkpoint. Adding {expected_diff} timestep tokens to tokenizer.")
                            self.timestep_token_ids = add_timestep_tokens_to_tokenizer(self.tokenizer, num_timesteps)
                            current_vocab_size = len(self.tokenizer)
                    rec_model.resize_token_embeddings(vocab_size)
                rec_model.load_state_dict(state_dict)
                rec_model.to(self.device)
                rec_model.eval()
                model_to_use = rec_model
                # Ensure tokenizer vocab size matches model vocab size
                # If timestep tokens were added, resize model to match tokenizer
                model_vocab_size = model_to_use.config.vocab_size
                tokenizer_vocab_size = len(self.tokenizer)
                if tokenizer_vocab_size > model_vocab_size:
                    logging.info(f"Resizing model vocab from {model_vocab_size} to {tokenizer_vocab_size} to match tokenizer (timestep tokens added).")
                    model_to_use.resize_token_embeddings(tokenizer_vocab_size)
                    model_vocab_size = model_to_use.config.vocab_size
                elif tokenizer_vocab_size != model_vocab_size:
                    logging.error(f"Tokenizer vocab size ({tokenizer_vocab_size}) != Model vocab size ({model_vocab_size}). "
                                 f"This will cause CUDA errors. Please ensure they match.")
                    raise ValueError(f"Vocabulary size mismatch: tokenizer={tokenizer_vocab_size}, model={model_vocab_size}")
                candidates = testloader.dataset.all_items
            elif test_type == "friend":
                state_dict = torch.load(self.args.social_model_path, map_location=self.device)
                config = T5Config.from_pretrained(self.args.backbone)
                social_model = T5ForConditionalGeneration.from_pretrained(self.args.backbone, config=config)
                if 'shared.weight' in state_dict:
                    vocab_size = state_dict['shared.weight'].shape[0]
                    current_vocab_size = len(self.tokenizer)
                    # Handle vocabulary size mismatch (e.g., timestep tokens added)
                    if vocab_size > current_vocab_size:
                        # Check if difference matches expected timestep tokens
                        num_timesteps = getattr(self.args, 'diffusion_timesteps', 100)
                        expected_diff = num_timesteps
                        if vocab_size - current_vocab_size == expected_diff:
                            logging.info(f"Detected timestep tokens in checkpoint. Adding {expected_diff} timestep tokens to tokenizer.")
                            self.timestep_token_ids = add_timestep_tokens_to_tokenizer(self.tokenizer, num_timesteps)
                            current_vocab_size = len(self.tokenizer)
                    social_model.resize_token_embeddings(vocab_size)
                social_model.load_state_dict(state_dict)
                social_model.to(self.device)
                social_model.eval()
                model_to_use = social_model
                # Ensure tokenizer vocab size matches model vocab size
                # If timestep tokens were added, resize model to match tokenizer
                model_vocab_size = model_to_use.config.vocab_size
                tokenizer_vocab_size = len(self.tokenizer)
                if tokenizer_vocab_size > model_vocab_size:
                    logging.info(f"Resizing model vocab from {model_vocab_size} to {tokenizer_vocab_size} to match tokenizer (timestep tokens added).")
                    model_to_use.resize_token_embeddings(tokenizer_vocab_size)
                    model_vocab_size = model_to_use.config.vocab_size
                elif tokenizer_vocab_size != model_vocab_size:
                    logging.error(f"Tokenizer vocab size ({tokenizer_vocab_size}) != Model vocab size ({model_vocab_size}). "
                                 f"This will cause CUDA errors. Please ensure they match.")
                    raise ValueError(f"Vocabulary size mismatch: tokenizer={tokenizer_vocab_size}, model={model_vocab_size}")
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
                
                # Validate token IDs are within model vocab size BEFORE prepending timestep tokens
                model_vocab_size = model_to_use.config.vocab_size
                max_token_id_before = input_ids.max().item()
                min_token_id_before = input_ids.min().item()
                if max_token_id_before >= model_vocab_size or min_token_id_before < 0:
                    logging.error(f"Invalid token IDs in input: min={min_token_id_before}, max={max_token_id_before}, vocab_size={model_vocab_size}. "
                                f"Clamping token IDs to valid range.")
                    input_ids = torch.clamp(input_ids, min=0, max=model_vocab_size - 1)
                
                if use_timestep_in_eval:
                    batch_size = input_ids.shape[0]
                    timestep_0 = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                    # Validate timestep token IDs are valid
                    if self.timestep_token_ids is not None:
                        timestep_token_id = self.timestep_token_ids[0]  # timestep 0
                        if timestep_token_id >= model_vocab_size:
                            logging.error(f"Timestep token ID {timestep_token_id} >= model vocab_size {model_vocab_size}. "
                                        f"Model vocab size needs to be increased to accommodate timestep tokens.")
                            raise ValueError(f"Timestep token ID {timestep_token_id} exceeds model vocab size {model_vocab_size}")
                    input_ids, attn = prepend_timestep_token(
                        input_ids, timestep_0, self.timestep_token_ids, attn
                    )
                    # Validate token IDs AFTER prepending timestep tokens
                    max_token_id_after = input_ids.max().item()
                    min_token_id_after = input_ids.min().item()
                    if max_token_id_after >= model_vocab_size or min_token_id_after < 0:
                        logging.error(f"Invalid token IDs after prepending timestep: min={min_token_id_after}, max={max_token_id_after}, vocab_size={model_vocab_size}. "
                                    f"Clamping token IDs to valid range.")
                        input_ids = torch.clamp(input_ids, min=0, max=model_vocab_size - 1)
                
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
                logging.info(f'{metrics_res[i]:.3f}')
        
   