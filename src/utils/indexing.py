from utils import utils
import os   
import tqdm
import re
import tqdm
import shutil
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import torch
        
def generative_indexing_id(data_path, dataset, user_sequence_dict, phase=0, run_id=None, component=None, run_type=None, social_quantization_id=0):
    run_dir = os.path.join(data_path, dataset, run_id)
    os.makedirs(run_dir, exist_ok=True)
    suffix = ''
    if run_type == '2id2rec' or run_type == '2id2rec_socialtoid':
        suffix = '_item' if component == 'item_rec' else '_social'    
    user_index_file = os.path.join(run_dir, f'user_generative_index_phase_{phase}{suffix}.txt')
    item_index_file = os.path.join(run_dir, f'item_generative_indexing_phase_{phase}{suffix}.txt') 
    
    if (run_type == '2id2rec' or run_type == '2id2rec_socialtoid') and component == 'item_rec' and phase > 0 and social_quantization_id:
        # Look for cross-social enhanced user index file using same phase
        cross_social_files = []
        for round_num in range(phase + 1):  # Check all possible round numbers
            cross_social_file = os.path.join(run_dir, f'cross_social_index_social_phase_{phase}_item_phase_{phase}_round_{round_num}.txt')
            if os.path.exists(cross_social_file):
                cross_social_files.append(cross_social_file)
        
        if cross_social_files:
            # Use the most recent cross-social enhanced user index
            latest_cross_social_file = max(cross_social_files, key=os.path.getmtime)
            print(f"Using cross-social enhanced user index: {latest_cross_social_file}")
            user_map = get_dict_from_lines(utils.ReadLineFromFile(latest_cross_social_file))
        else:
            # Fallback to regular user index file
            if phase == 0:
                original_user_index_file = os.path.join(data_path, dataset, f'user_generative_index_phase_0.txt')
                if os.path.exists(original_user_index_file) and not os.path.exists(user_index_file):
                    shutil.copy2(original_user_index_file, user_index_file)
                    print(f"Copied existing user index file to run directory (with suffix): {user_index_file}")
            if not os.path.exists(user_index_file):
                raise FileNotFoundError(
                    f"User index file not found: {user_index_file}\n"
                    f"Expected file path: {user_index_file}\n"
                    f"Run directory: {run_dir}\n"
                    f"Phase: {phase}, Suffix: '{suffix}'\n"
                    f"Run type: {run_type}, Component: {component}\n"
                    f"For phase 0, the original file should exist at: {os.path.join(data_path, dataset, 'user_generative_index_phase_0.txt')}"
                )
            user_map = get_dict_from_lines(utils.ReadLineFromFile(user_index_file))
    else:
        if phase == 0:
            original_user_index_file = os.path.join(data_path, dataset, f'user_generative_index_phase_0.txt')
            original_item_index_file = os.path.join(data_path, dataset, f'item_generative_indexing_phase_0.txt')
            if os.path.exists(original_user_index_file) and not os.path.exists(user_index_file):
                shutil.copy2(original_user_index_file, user_index_file)
                print(f"Copied existing user index file to run directory: {user_index_file}")
            if os.path.exists(original_item_index_file) and not os.path.exists(item_index_file):
                shutil.copy2(original_item_index_file, item_index_file)
                print(f"Copied existing item index file to run directory: {item_index_file}")
        if not os.path.exists(user_index_file):
            raise FileNotFoundError(
                f"User index file not found: {user_index_file}\n"
                f"Expected file path: {user_index_file}\n"
                f"Run directory: {run_dir}\n"
                f"Phase: {phase}, Suffix: '{suffix}'\n"
                f"Run type: {run_type}, Component: {component}\n"
                f"For phase 0, the original file should exist at: {os.path.join(data_path, dataset, 'user_generative_index_phase_0.txt')}"
            )
        user_map = get_dict_from_lines(utils.ReadLineFromFile(user_index_file))
    
    item_map = get_dict_from_lines(utils.ReadLineFromFile(item_index_file))
    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    return reindex_user_sequence_dict, item_map

def generative_indexing_rec(data_path, dataset, user_sequence_dict, model_gen, tokenizer, regenerate=True, phase=0, run_id=None, component=None, run_type=None):
    item_text_file = os.path.join(data_path, dataset, 'item_plain_text.txt')
    user_sequence_file = os.path.join(data_path, dataset, 'user_sequence.txt')
    run_dir = os.path.join(data_path, dataset, run_id)
    os.makedirs(run_dir, exist_ok=True)
    if run_type == '2id2rec' or run_type == '2id2rec_socialtoid':
        suffix = '_item' if component == 'item_rec' else '_social'    
    item_index_file = os.path.join(run_dir, f'item_generative_indexing_phase_{phase}{suffix}.txt')
    user_index_file = os.path.join(run_dir, f'user_generative_index_phase_{phase}{suffix}.txt') 
    if phase == 0:
        original_item_index_file = os.path.join(data_path, dataset, f'item_generative_indexing_phase_0.txt')
        original_user_index_file = os.path.join(data_path, dataset, f'user_generative_index_phase_0.txt')
        if os.path.exists(original_item_index_file) and not os.path.exists(item_index_file):            
            shutil.copy2(original_item_index_file, item_index_file)
            print(f"Copied existing item index file to run directory: {item_index_file}")     
        if os.path.exists(original_user_index_file) and not os.path.exists(user_index_file):
            shutil.copy2(original_user_index_file, user_index_file)
            print(f"Copied existing user index file to run directory: {user_index_file}")
    if (phase == 0 and not os.path.exists(item_index_file)) or (phase != 0 and regenerate):
        print(f"(re)generate textual id with id generator phase {phase} for component {component}!")
        generate_item_id_from_text(item_text_file, item_index_file, model_gen, tokenizer)
        logging.info(f"Regenerated item ID phase {phase} for component {component} at {item_index_file}")
    item_info = utils.ReadLineFromFile(item_index_file)
    item_map = get_dict_from_lines(item_info)
    if (phase == 0 and not os.path.exists(user_index_file)) or (phase != 0 and regenerate):
        print(f"(re)generate user id with id generator phase {phase} for component {component}!")
        generate_user_id_from_text(item_map, user_index_file, user_sequence_file, model_gen, tokenizer)
        logging.info(f"Regenerated user ID phase {phase} for component {component} at {user_index_file}")
    user_info = utils.ReadLineFromFile(user_index_file)
    user_map = get_dict_from_lines(user_info)
    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    return reindex_user_sequence_dict, item_map 

def _filter_users_with_training_data(sequence_dict, is_social=False):
    """
    Filter users to only include those that will have training data.
    
    For social (friend) sequences:
    - Skip users with len <= 2 (they won't have training data after split)
    - Include users with len >= 3 (at least 2 training friends after split)
    
    For item sequences:
    - Include all users with at least 1 item (all have training data)
    
    Args:
        sequence_dict: Dictionary mapping user_id to list of items/friends
        is_social: If True, apply social filtering logic (skip users with <= 2 friends)
    
    Returns:
        Filtered sequence dictionary
    """
    filtered_dict = {}
    skipped_count = 0
    
    for user_id, sequence in sequence_dict.items():
        if is_social:
            # For social: skip users with <= 2 friends (they won't have training data)
            # len==1: 1 training friend -> skipped in load_train (needs >= 2 training friends)
            # len==2: 1 training friend -> skipped in load_train (needs >= 2 training friends)
            # len>=3: at least 2 training friends -> included
            if len(sequence) <= 2:
                skipped_count += 1
                continue
        else:
            # For items: include all users with at least 1 item
            if len(sequence) == 0:
                skipped_count += 1
                continue
        
        filtered_dict[user_id] = sequence
    
    if skipped_count > 0:
        logging.info(f"Filtered out {skipped_count} users without training data (is_social={is_social}, remaining={len(filtered_dict)})")
    
    return filtered_dict

def generative_indexing_social(data_path, dataset, friend_sequence_dict, phase=0, run_id=None, component=None, model_gen=None, tokenizer=None, regenerate=True, run_type=None):
    item_text_file = os.path.join(data_path, dataset, 'item_plain_text.txt')
    user_sequence_file = os.path.join(data_path, dataset, 'user_sequence.txt')
    run_dir = os.path.join(data_path, dataset, run_id)
    os.makedirs(run_dir, exist_ok=True)
    suffix = ''
    if run_type == '2id2rec' or run_type == '2id2rec_socialtoid':
        if component == 'item_rec':
            suffix = '_item'
        elif component == 'friend_rec':
            suffix = '_social'
    elif run_type == '2id1rec':
        if component == 'item_view':
            suffix = '_item'
        elif component == 'social_view':
            suffix = '_social'
    
    # Filter users to only include those that will have training data
    # Users with <= 2 friends won't have training data (they get skipped in load_train)
    friend_sequence_dict = _filter_users_with_training_data(friend_sequence_dict, is_social=True)
    
    item_index_file = os.path.join(run_dir, f'item_generative_indexing_phase_{phase}{suffix}.txt')
    logging.info(f"Item index file: {item_index_file}")
    user_index_file = os.path.join(run_dir, f'user_generative_index_phase_{phase}{suffix}.txt')
    
    if phase == 0:
        original_item_index_file = os.path.join(data_path, dataset, f'item_generative_indexing_phase_0.txt')
        original_user_index_file = os.path.join(data_path, dataset, f'user_generative_index_phase_0.txt')
        
        if os.path.exists(original_item_index_file) and not os.path.exists(item_index_file):
            shutil.copy2(original_item_index_file, item_index_file)
            print(f"Copied existing item index file to run directory (with suffix): {item_index_file}")
            
        if os.path.exists(original_user_index_file) and not os.path.exists(user_index_file):
            shutil.copy2(original_user_index_file, user_index_file)
            print(f"Copied existing user index file to run directory (with suffix): {user_index_file}")
    if (phase == 0 and not os.path.exists(item_index_file)) or (phase != 0 and regenerate):
        if model_gen and tokenizer:
            print(f"(re)generate item textual id with id generator phase {phase} for component {component}!")
            generate_item_id_from_text(item_text_file, item_index_file, model_gen, tokenizer)
    item_info = utils.ReadLineFromFile(item_index_file)
    item_map = get_dict_from_lines(item_info)
    if (phase == 0 and not os.path.exists(user_index_file)) or (phase != 0 and regenerate):
        if model_gen and tokenizer:
            print(f"(re)generate user id with id generator phase {phase} for component {component}!")
            # Only generate IDs for users that will have training data
            user_filter = set(friend_sequence_dict.keys()) if friend_sequence_dict else None
            generate_user_id_from_text(item_map, user_index_file, user_sequence_file, model_gen, tokenizer, user_filter=user_filter)
    user_info = utils.ReadLineFromFile(user_index_file)
    user_map = get_dict_from_lines(user_info)
    
    reindex_friend_sequence_dict = reindex_social(friend_sequence_dict, user_map)
    return reindex_friend_sequence_dict, user_map
 
def get_dict_from_lines(lines): 
    index_map = dict()
    for line in lines:
        info = line.split(" ", 1)
        index_map[info[0]] = info[1]
    return index_map
              
def reindex(user_sequence_dict, user_map, item_map):
    reindex_user_sequence_dict = dict()
    for user in user_sequence_dict:
        # Use textual ID if available, otherwise fallback to original ID
        # (users without textual IDs were filtered out because they have no training data,
        #  but we still need to handle them if they appear in other users' sequences)
        if user in user_map:
            uid = user_map[user]  # Use textual ID
        else:
            uid = user  # Fallback to original ID
            logging.debug(f"User {user} has no textual ID, using original ID as fallback")
        
        items = user_sequence_dict[user]
        # Only include items that exist in item_map, fallback to original item ID if not mapped
        mapped_items = []
        for i in items:
            if i in item_map:
                mapped_items.append(item_map[i])  # Use textual item ID
            else:
                mapped_items.append(i)  # Fallback to original item ID
                logging.debug(f"Item {i} has no textual ID, using original ID as fallback")
        
        if mapped_items:  # Only add if there are items
            reindex_user_sequence_dict[uid] = mapped_items
        
    return reindex_user_sequence_dict
    
def reindex_social(friend_sequence_dict, user_map):
    reindex_friend_sequence_dict = dict()
    for user in friend_sequence_dict:
        # Use textual ID if available, otherwise fallback to original ID
        if user in user_map:
            uid = user_map[user]  # Use textual ID
        else:
            uid = user  # Fallback to original ID
            logging.debug(f"User {user} has no textual ID, using original ID as fallback")
        
        items = friend_sequence_dict[user]
        mapped_friends = []
        for i in items:
            if i in user_map:
                mapped_friends.append(user_map[i])  # Use textual friend ID
            else:
                mapped_friends.append(i)  # Fallback to original friend ID
                logging.debug(f"Friend {i} has no textual ID, using original ID as fallback")
        
        if mapped_friends:
            reindex_friend_sequence_dict[uid] = mapped_friends
        
    return reindex_friend_sequence_dict
    
def construct_user_sequence_dict(user_sequence):
    user_seq_dict = dict()
    for line in user_sequence:
        user_seq = line.split(" ")
        user_seq_dict[user_seq[0]] = user_seq[1:]
    return user_seq_dict

def generate_item_id_from_text(item_text_file_dir, item_id_file_dir, model_gen, tokenizer):
    device = next(model_gen.parameters()).device
    logging.info(f"Generating item id using device: {device}")
    model_gen.eval()
    for param in model_gen.parameters():
        param.requires_grad = False
    torch.cuda.synchronize()

    item_text_dict = {}
    with open(item_text_file_dir, 'r') as file:
        for line in file:
            id_, text = line.strip().split(' ', 1)
            item_text_dict[id_] = text.strip()

    id_set = set()
    item_id_dict = {}
    max_dp = 0

    for iid, text in tqdm.tqdm(item_text_dict.items(), desc="Generating Item IDs"):
        found = False
        dp = 1.
        min_l = 1
        #logging.info(f"Generating item id for {iid} with text: {text}")
        while not found:
            inputs = tokenizer([text], max_length=256, truncation=True, return_tensors="pt").to(device)
            generate_fn = model_gen.module.generate if hasattr(model_gen, "module") else model_gen.generate
            with torch.no_grad():
                outputs = generate_fn(
                    **inputs,
                    num_beams=10, num_beam_groups=10, do_sample=False,
                    min_length=min_l, max_length=min_l + 10,
                    diversity_penalty=dp, num_return_sequences=10
                )

            decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for output in decoded_output:
                tags = re.findall(r'\b\w+\b', output)
                gen_id = ' '.join(tags)
                if gen_id not in id_set:
                    id_set.add(gen_id)
                    item_id_dict[iid] = gen_id
                    max_dp = max(max_dp, dp)
                    found = True
                    break

            dp += 1
            if dp >= 10:
                min_l += 10
                dp = 1.
        #logging.info(f'New id: {gen_id} for item {iid}')
    with open(item_id_file_dir, "w") as f:
        for k, v in item_id_dict.items():
            f.write(f"{k} {v}\n")

    return True

def generate_text_id_in_memory(text, model_gen, tokenizer, existing_ids=None):
    device = next(model_gen.parameters()).device
    model_gen.eval()
    for param in model_gen.parameters():
        param.requires_grad = False

    found = False
    dp = 1.
    min_l = 1
    max_attempts = 50
    
    for attempt in range(max_attempts):
        try:
            inputs = tokenizer([text], max_length=256, truncation=True, return_tensors="pt").to(device)

            generate_fn = model_gen.module.generate if hasattr(model_gen, "module") else model_gen.generate
            with torch.no_grad():
                outputs = generate_fn(
                    **inputs,
                    num_beams=10, num_beam_groups=10, do_sample=False,
                    min_length=min_l, max_length=min_l + 10,
                    diversity_penalty=dp, num_return_sequences=10
                )

            decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for output in decoded_output:
                tags = re.findall(r'\b\w+\b', output)
                gen_id = ' '.join(tags)
                if existing_ids is None or gen_id not in existing_ids:
                    return gen_id

            dp += 1
            if dp >= 10:
                min_l += 10
                dp = 1.
                
        except Exception as e:
            print(f"Error generating ID: {e}")
            break
    
    return None

def generate_user_id_from_text(item_map, user_index_file, user_sequence_file, model_gen, tokenizer, user_filter=None):
    """
    Generate user textual IDs from their item sequences.
    
    Args:
        item_map: Dictionary mapping item IDs to textual item IDs
        user_index_file: Output file path for user index
        user_sequence_file: Input file with user sequences
        model_gen: Model for generating IDs
        tokenizer: Tokenizer
        user_filter: Optional set of user IDs to include (if None, include all users)
    """
    device = next(model_gen.parameters()).device
    model_gen.eval()
    for param in model_gen.parameters():
        param.requires_grad = False
    torch.cuda.synchronize()
    logging.info(f"Generating user id using device: {device}")
    if user_filter is not None:
        logging.info(f"Filtering users: only generating IDs for {len(user_filter)} users with training data")
    user_seq_dict = {}
    with open(user_sequence_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts:
                uid = parts[0]
                # Skip users not in filter if filter is provided
                if user_filter is not None and uid not in user_filter:
                    continue
                item_ids = parts[1:]
                mapped = [item_map[item] for item in item_ids if item in item_map]
                if mapped:
                    user_seq_dict[uid] = mapped

    id_set = set()
    user_id_dict = {}
    max_dp = 0

    for uid, mapped_items in tqdm.tqdm(user_seq_dict.items(), desc="Generating User IDs"):
        text = ' '.join(mapped_items[:-1])
        found = False
        dp = 1.
        min_l = 1
        logging.info(f"Generating user id for {uid} with text: {text}")
        if text is None or text == "":
            logging.warning(f"Text is None or empty for user {uid}")
            continue
        while not found:
            inputs = tokenizer([text], max_length=256, truncation=True, return_tensors="pt").to(device)

            generate_fn = model_gen.module.generate if hasattr(model_gen, "module") else model_gen.generate
            with torch.no_grad():
                outputs = generate_fn(
                    **inputs,
                    num_beams=10, num_beam_groups=10, do_sample=False,
                    min_length=min_l, max_length=min_l + 10,
                    diversity_penalty=dp, num_return_sequences=10
                )

            decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for output in decoded_output:
                tags = re.findall(r'\b\w+\b', output)
                gen_id = ' '.join(tags)
                if gen_id not in id_set:
                    id_set.add(gen_id)
                    user_id_dict[uid] = gen_id
                    max_dp = max(max_dp, dp)
                    found = True
                    break

            dp += 1
            if dp >= 10:
                min_l += 10
                dp = 1.
        logging.info(f'New id: {gen_id} for user {uid}')
    with open(user_index_file, "w") as f:
        for uid, gen_id in user_id_dict.items():
            f.write(f"{uid} {gen_id}\n")

    return True

def generate_user_social_generative_index(data_path, dataset, model_gen, tokenizer, phase=0, run_id=None, regenerate=True):
    run_dir = os.path.join(data_path, dataset, run_id)
    os.makedirs(run_dir, exist_ok=True)
    user_social_index_file = os.path.join(run_dir, f'user_social_generative_index_phase_{phase}.txt')
    
    if regenerate:
        sentence_transformer = SentenceTransformer("sentence-transformers/sentence-t5-base")
        print(f"(re)generate user social generative index with model phase {phase}!")
        user_index_file = os.path.join(run_dir, f'user_generative_index_phase_{phase}.txt')
        if not os.path.exists(user_index_file):
            print(f"Error: User generative index file not found: {user_index_file}")
            return {}
        user_generative_index_dict = get_dict_from_lines(utils.ReadLineFromFile(user_index_file))
        friend_sequence = utils.ReadLineFromFile(os.path.join(data_path, dataset, "friend_sequence.txt"))
        friend_lookup = {}
        for line in friend_sequence:
            parts = line.lower().split()
            if len(parts) > 1:
                if len(parts) >= 4:  
                    training_friends = parts[1:-2]
                    if training_friends:  
                        friend_lookup[parts[0]] = training_friends
        
        textual_to_original = {v: k for k, v in user_generative_index_dict.items()}
        user_social_id_dict = {}
        for uid, textual_user_id in tqdm.tqdm(user_generative_index_dict.items(), desc="Generating User Social IDs"):
            try:
                social_quantization_id = _social_quantization_user_optimized(
                    textual_user_id=textual_user_id,
                    textual_to_original=textual_to_original,
                    friend_lookup=friend_lookup,
                    user_generative_index_dict=user_generative_index_dict,
                    model_gen=model_gen,
                    tokenizer=tokenizer,
                    sentence_transformer=sentence_transformer,
                )
                user_social_id_dict[uid] = social_quantization_id
                
            except Exception as e:
                logging.error(f"Error generating social quantization for user {uid}: {e}")
                user_social_id_dict[uid] = textual_user_id
        with open(user_social_index_file, "w") as f:
            for uid, social_id in user_social_id_dict.items():
                f.write(f"{uid} {social_id}\n")
        
        logging.info(f"Generated user social generative index with {len(user_social_id_dict)} users")
    user_social_map = get_dict_from_lines(utils.ReadLineFromFile(user_social_index_file))
    return user_social_map

def generate_cross_social_index(data_path, dataset, model_gen, tokenizer, social_phase=0, item_phase=0, run_id=None, regenerate=True, round_num=None):
    run_dir = os.path.join(data_path, dataset, run_id)
    os.makedirs(run_dir, exist_ok=True)
    # Include round number in filename to ensure each round gets its own cross-social index
    if round_num is not None:
        cross_social_index_file = os.path.join(run_dir, f'cross_social_index_social_phase_{social_phase}_item_phase_{item_phase}_round_{round_num}.txt')
    else:
        cross_social_index_file = os.path.join(run_dir, f'cross_social_index_social_phase_{social_phase}_item_phase_{item_phase}.txt')
    if regenerate:
        print(f"(re)generate cross social index with social phase {social_phase} and item phase {item_phase}!")
        
        # Load item and social user indices using correct phases
        item_user_index_file = os.path.join(run_dir, f'user_generative_index_phase_{item_phase}_item.txt')
        social_user_index_file = os.path.join(run_dir, f'user_generative_index_phase_{social_phase}_social.txt')
        
        if not os.path.exists(item_user_index_file) or not os.path.exists(social_user_index_file):
            print(f"Error: Required index files not found")
            return {}
        
        item_user_dict = get_dict_from_lines(utils.ReadLineFromFile(item_user_index_file))
        social_user_dict = get_dict_from_lines(utils.ReadLineFromFile(social_user_index_file))
        
        # Load friend sequence for social neighbors
        # IMPORTANT: Only use training friends (exclude last 2: validation and test)
        # Convention: friends[:-2] = train, friends[-2] = validation, friends[-1] = test
        friend_sequence = utils.ReadLineFromFile(os.path.join(data_path, dataset, "friend_sequence.txt"))
        friend_lookup = {}
        for line in friend_sequence:
            parts = line.lower().split()
            if len(parts) > 1:
                if len(parts) >= 4:  
                    training_friends = parts[1:-2]
                    if training_friends:  
                        friend_lookup[parts[0]] = training_friends
        
        cross_social_dict = {}
        for uid, item_textual_id in item_user_dict.items():
            if uid in social_user_dict and uid in friend_lookup:
                social_neighbors = friend_lookup[uid]
                neighbor_textual_ids = [social_user_dict.get(neighbor, "") for neighbor in social_neighbors if neighbor in social_user_dict]
                neighbor_text = " ".join(neighbor_textual_ids)
                
                enhanced_id = item_textual_id + " " + neighbor_text
                cross_social_dict[uid] = enhanced_id
            else:
                cross_social_dict[uid] = item_textual_id
        
        with open(cross_social_index_file, "w") as f:
            for uid, enhanced_id in cross_social_dict.items():
                f.write(f"{uid} {enhanced_id}\n")
        
        print(f"Generated cross-social index with {len(cross_social_dict)} users")
    
    return get_dict_from_lines(utils.ReadLineFromFile(cross_social_index_file))

def _social_quantization_user_optimized(textual_user_id, textual_to_original, friend_lookup, user_generative_index_dict, model_gen, tokenizer, sentence_transformer):
    if textual_user_id not in textual_to_original:
        return textual_user_id
    
    original_user_id = textual_to_original[textual_user_id]
    if original_user_id not in friend_lookup:
        return textual_user_id
    
    # Get 1-hop friends
    friend_1 = friend_lookup[original_user_id]
    if len(friend_1) < 1:  # Skip users with no friends
        return textual_user_id
    
    # Get 2-hop friends using lookup dictionary
    friend_2_all = []
    friend_1_set = set(friend_1)
    for friend in friend_1:
        if friend in friend_lookup:
            friend_2 = [f for f in friend_lookup[friend] if f not in friend_1_set]
            friend_2_all.extend(friend_2)
    
    # Get 3-hop friends
    friend_3_all = []
    friend_2_set = set(friend_2_all)
    for friend in friend_2_all:
        if friend in friend_lookup:
            friend_3 = [f for f in friend_lookup[friend] if f not in friend_2_set]
            friend_3_all.extend(friend_3)
    
    # Skip if no friends at any level
    if not friend_1 and not friend_2_all and not friend_3_all:
        logging.warning(f"No friends at any level for user {textual_user_id}. Fallback to original ID.")
        return textual_user_id
    
    # Get friend IDs text
    try:
        friend_ids_text = ' '.join([user_generative_index_dict[friend] for friend in friend_1 if friend in user_generative_index_dict])
        friend_ids_text_2 = ' '.join([user_generative_index_dict[friend] for friend in friend_2_all if friend in user_generative_index_dict])
        friend_ids_text_3 = ' '.join([user_generative_index_dict[friend] for friend in friend_3_all if friend in user_generative_index_dict])
    except KeyError:
        logging.warning(f"No friends at any level for user {textual_user_id}. Fallback to original ID.")
        return textual_user_id
    
    # Generate tags for each hop level using in-memory processing
    generated_id = generate_text_id_in_memory(friend_ids_text, model_gen, tokenizer)
    generated_id_2 = generate_text_id_in_memory(friend_ids_text_2, model_gen, tokenizer)
    generated_id_3 = generate_text_id_in_memory(friend_ids_text_3, model_gen, tokenizer)
    
    # Skip if generation failed
    if not generated_id or not generated_id_2 or not generated_id_3:
        return textual_user_id
        
    # Process user and compute similarities
    user_id_text = user_generative_index_dict.get(original_user_id, "")
    if not user_id_text:
        return textual_user_id
        
    user_id_text = ' '.join(list(set(user_id_text.split())))
    if user_id_text is not None:
        user_id_text_embedding = sentence_transformer.encode(user_id_text, show_progress_bar=False)
    else:
        user_id_text_embedding = np.zeros(768)
    
    # Extract and encode tags
    sep_tags = generated_id.split()
    if sep_tags and len(sep_tags) > 0:
        sep_tags_embedding = sentence_transformer.encode(sep_tags, show_progress_bar=False)
    else:
        sep_tags_embedding = np.zeros(768)
    sep_tags_2 = generated_id_2.split()
    if sep_tags_2 and len(sep_tags_2) > 0:
        sep_tags_2_embedding = sentence_transformer.encode(sep_tags_2, show_progress_bar=False)
    else:
        sep_tags_2_embedding = np.zeros(768)
    sep_tags_3 = generated_id_3.split()
    if sep_tags_3 and len(sep_tags_3) > 0:
        sep_tags_3_embedding = sentence_transformer.encode(sep_tags_3, show_progress_bar=False)
    else:
        sep_tags_3_embedding = np.zeros(768)
    
    # Compute similarities and select best tags
    cosine_scores = np.dot(sep_tags_embedding, user_id_text_embedding)
    v1_indices = np.argsort(cosine_scores)[-3:]
    v1_indices = v1_indices[v1_indices < len(sep_tags)]
    
    residual_1hop = user_id_text_embedding - np.mean(sep_tags_embedding[v1_indices], axis=0)
    cosine_scores_2 = np.dot(sep_tags_2_embedding, residual_1hop)
    v2_indices = np.argsort(cosine_scores_2)[-2:]
    v2_indices = v2_indices[v2_indices < len(sep_tags_2)]
    
    residual_2hop = residual_1hop - np.mean(sep_tags_2_embedding[v2_indices], axis=0)
    cosine_scores_3 = np.dot(sep_tags_3_embedding, residual_2hop)
    v3_indices = np.argsort(cosine_scores_3)[-1:]
    v3_indices = v3_indices[v3_indices < len(sep_tags_3)]
    
    social_quantization = (
        " " + " ".join([sep_tags[i] for i in v1_indices]) #+
    #    " " + " ".join([sep_tags_2[i] for i in v2_indices]) +
    #    " " + " ".join([sep_tags_3[i] for i in v3_indices])
    )

    user_with_social_quantization = user_id_text + " " + social_quantization

    return user_with_social_quantization
