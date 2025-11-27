import torch
import numpy as np
import logging
from utils.discrete_diffusion import DiscreteDiffusionScheduler
from utils.discrete_diffusion import corrupt_sequence, get_cross_view_tokens
import random
class Collator:
    def __init__(self, tokenizer, args=None):
        self.tokenizer = tokenizer
        self.args = args
        if self.args.use_diffusion:
            self.scheduler = DiscreteDiffusionScheduler(
                num_timesteps=self.args.diffusion_timesteps,
                beta_max=self.args.diffusion_beta_max
            )
            self.use_diffusion = True
            self.vocab_size = len(tokenizer)
            self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            logging.info(f"Collator: Diffusion enabled with T={self.scheduler.num_timesteps}, β_max={self.args.diffusion_beta_max}")
        else:
            logging.info("Collator: Diffusion disabled")
            self.use_diffusion = False
            self.scheduler = None
        self._logged_first_sample = False

    def __call__(self, batch):
        input_texts = [input_text['input'] for input_text in batch]
        output_texts = [input_text['output'] for input_text in batch]
        cross_view_texts = [input_text.get('cross_view_tokens', None) for input_text in batch]
        inputs = self.tokenizer.batch_encode_plus(
            input_texts, padding="longest", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"]
        cross_view_token_ids_list = []
        if any(cv is not None for cv in cross_view_texts):
            for cv_text in cross_view_texts:
                if cv_text is not None:
                    cv_tokens = self.tokenizer.encode(cv_text, add_special_tokens=False)
                    cross_view_token_ids_list.append(torch.tensor(cv_tokens, dtype=torch.long))
                else:
                    cross_view_token_ids_list.append(None)
        
        # Apply diffusion corruption if enabled
        noise_masks = None
        timesteps = None
        if self.use_diffusion:
            corrupted_input_ids = []
            noise_masks_list = []
            timesteps_list = []
            
            for i, input_id_seq in enumerate(input_ids):
                input_tensor = torch.tensor(input_id_seq, dtype=torch.long)
                t = random.randint(0, self.scheduler.num_timesteps - 1)
                timesteps_list.append(t)
                cross_view_tokens = None
                if i < len(cross_view_token_ids_list) and cross_view_token_ids_list[i] is not None:
                    cross_view_tokens = get_cross_view_tokens(
                        item_sequence=input_tensor,
                        social_sequence=cross_view_token_ids_list[i],
                        pad_token_id=self.pad_token_id
                    )
                corrupted_seq, noise_mask = corrupt_sequence(
                    clean_sequence=input_tensor,
                    timestep=t,
                    scheduler=self.scheduler,
                    vocab_size=self.vocab_size,
                    cross_view_tokens=cross_view_tokens,
                    cross_view_prob=self.args.diffusion_cross_prob,
                    pad_token_id=self.pad_token_id,
                    seed=None
                )
                
                corrupted_input_ids.append(corrupted_seq.tolist())
                noise_masks_list.append(noise_mask.tolist())
            input_ids = corrupted_input_ids
            noise_masks = noise_masks_list
            timesteps = timesteps_list
        
        whole_word_ids = []
        for input_id in input_ids:
            tokenized_text = self.tokenizer.convert_ids_to_tokens(input_id)
            whole_word_id = calculate_whole_word_ids(tokenized_text, input_id)
            whole_word_ids.append(whole_word_id)
        input_attention = inputs["attention_mask"]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts, padding="longest", truncation=True, max_length=512
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]

        result = (
            torch.tensor(input_ids),
            torch.tensor(input_attention),
            torch.tensor(whole_word_ids),
            torch.tensor(output_ids),
            torch.tensor(output_attention),
        )
        
        # Add timesteps and noise masks if diffusion is enabled
        if self.use_diffusion and timesteps is not None and noise_masks is not None:
            result = result + (
                torch.tensor(timesteps),
                torch.tensor(noise_masks),
            )
        
        return result

    
class TestCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_texts = [input_text['input'] for input_text in batch]
        output_texts = [input_text['output'] for input_text in batch]

        inputs = self.tokenizer.batch_encode_plus(
            input_texts, padding="longest", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"]
        whole_word_ids = []
        for input_id in input_ids:
            tokenized_text = self.tokenizer.convert_ids_to_tokens(input_id)
            whole_word_id = calculate_whole_word_ids(tokenized_text, input_id)
            whole_word_ids.append(whole_word_id)
        input_attention = inputs["attention_mask"]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts, padding="longest", truncation=True, max_length=512
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]

        return (
            torch.tensor(input_ids),
            torch.tensor(input_attention),
            torch.tensor(whole_word_ids),
            torch.tensor(output_ids),
            torch.tensor(output_attention),     
        )

    
class CollatorGen:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output_texts = [input_text['output_prompt'] for input_text in batch]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts, padding="longest", truncation=True, max_length=512
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]
        histories = [input_text['history'] for input_text in batch]
        input_prompt = [input_text['input_prompt'] for input_text in batch]
        hist_lengths = [len(hist) for hist in histories]
        input_prompt_ph = []  
        input_insert_positions = []
        tokenized_prompts = []  
        for i, p in enumerate(input_prompt):
            length = hist_lengths[i]
            p_s = p.replace("{history}", " ; " * (length))
            tokens = self.tokenizer.tokenize(p_s)
            insert_p = [1 if token == ";" else 0 for token in tokens]
            tokenized_prompts.append(tokens)
            input_prompt_ph.append(p_s)
            input_insert_positions.append(insert_p)
        input_prompt_inputs = self.tokenizer.batch_encode_plus(
            tokenized_prompts, is_split_into_words=True, padding="longest", truncation=True, max_length=512
        )
        # Pad each insert_p to match its corresponding encoded sequence length
        for i, insert_p in enumerate(input_insert_positions):
            target_len = len(input_prompt_inputs['input_ids'][i])
            while len(insert_p) < target_len:
                insert_p.append(0)
            # Truncate if needed (in case insert_p is longer)
            if len(insert_p) > target_len:
                input_insert_positions[i] = insert_p[:target_len]
        flattened_histories = [plain_text for hist in histories for plain_text in hist]
        hist_lengths = [len(hist) for hist in histories] 
        max_hist_length = max(hist_lengths) if hist_lengths else 0
    
        # Handle case when all histories are empty
        if not flattened_histories:
            # Create empty tensors with proper shape
            history_input_ids = torch.zeros((len(histories), max_hist_length, 1), dtype=torch.long)
            history_input_attention = torch.zeros((len(histories), max_hist_length, 1))
        else:
            # process input history, need two level of paddings, history level and plain text level
            history_inputs = self.tokenizer.batch_encode_plus(
                flattened_histories, padding="longest", truncation=True, max_length=256)
            max_hist_token = len(history_inputs['input_ids'][0]) if history_inputs['input_ids'] else 1

            # Apply padding at the history level
            padded_histories = []
            padded_attention_mask_histories = []
            current_index = 0

            for length in hist_lengths:
                padded_hist = torch.zeros((max_hist_length, max_hist_token), dtype=torch.long)
                padded_attention_mask = torch.zeros((max_hist_length, max_hist_token))
                
                if length > 0:  # Only assign if there are actual history items
                    padded_hist[:length] = torch.tensor(history_inputs['input_ids'][current_index:current_index+length], dtype=torch.long)
                    padded_attention_mask[:length] = torch.tensor(history_inputs['attention_mask'][current_index:current_index+length])
                
                padded_histories.append(padded_hist)
                padded_attention_mask_histories.append(padded_attention_mask)
                current_index += length
            
            history_input_ids = torch.stack(padded_histories)
            history_input_attention = torch.stack(padded_attention_mask_histories)

        return (
            torch.tensor(input_prompt_inputs['input_ids']),
            torch.tensor(input_insert_positions),
            history_input_ids,
            history_input_attention,
            torch.tensor(output_ids),
            torch.tensor(output_attention),
        )


def calculate_whole_word_ids(tokenized_text, input_ids):
    whole_word_ids = []
    curr = 0
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == "<pad>":
            curr = 0
        if tokenized_text[i].startswith("▁"):
            curr += 1
            whole_word_ids.append(curr)
        else:
            whole_word_ids.append(curr)
    return whole_word_ids[: len(input_ids) - 1] + [0]  # [0] for </s>
