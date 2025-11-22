#!/usr/bin/env python3

import sys
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config
from sentence_transformers import SentenceTransformer

def tokenize_line(line):
    """Convert a line to lowercase and extract tokens (words)."""
    import re
    tokens = re.findall(r'\b\w+\b', line.lower())
    return tokens

def extract_original_id(line):
    """Extract the original ID from a line (e.g., 'i13585' from 'i13585 musica catalana')."""
    import re
    match = re.match(r'^(\w+)', line.strip())
    return match.group(1) if match else None

def lines_are_similar(line1, line2):
    """Check if two lines have similar word sets (order doesn't matter)."""
    words1 = set(tokenize_line(line1))
    words2 = set(tokenize_line(line2))
    return words1 == words2

def analyze_file_differences(file1, file2):
    """Analyze differences between two files and return percentage statistics."""
    
    # Check if files exist
    if not os.path.exists(file1):
        print(f"Error: File '{file1}' does not exist.")
        return None
    
    if not os.path.exists(file2):
        print(f"Error: File '{file2}' does not exist.")
        return None
    
    # Read both files
    try:
        with open(file1, 'r', encoding='utf-8') as f:
            lines1 = f.readlines()
        with open(file2, 'r', encoding='utf-8') as f:
            lines2 = f.readlines()
    except Exception as e:
        print(f"Error reading files: {e}")
        return None
    
    # Strip whitespace and get line counts
    lines1 = [line.strip() for line in lines1]
    lines2 = [line.strip() for line in lines2]
    file1_lines = len(lines1)
    file2_lines = len(lines2)

    # Compare lines using word-based similarity
    max_lines = max(file1_lines, file2_lines)
    min_lines = min(file1_lines, file2_lines)
    
    similar_lines = 0
    different_lines = 0
    different_original_ids = []  # Track original IDs with different textual content
    
    # Compare lines up to the minimum length
    for i in range(min_lines):
        if lines_are_similar(lines1[i], lines2[i]):
            similar_lines += 1
        else:
            different_lines += 1
            # Extract original ID from the line with differences
            original_id = extract_original_id(lines1[i])
            if original_id:
                different_original_ids.append(original_id)
    
    # Count extra lines in the longer file as differences
    extra_lines = max_lines - min_lines
    different_lines += extra_lines
    
    # For extra lines, also track their original IDs
    if extra_lines > 0:
        longer_file_lines = lines1 if file1_lines > file2_lines else lines2
        for i in range(min_lines, max_lines):
            original_id = extract_original_id(longer_file_lines[i])
            if original_id:
                different_original_ids.append(original_id)

    # Calculate percentages
    if max_lines > 0:
        diff_percentage = (different_lines / max_lines) * 100
        similar_percentage = (similar_lines / max_lines) * 100
    
    return {
        'diff_percentage': diff_percentage if max_lines > 0 else 0,
        'different_original_ids': different_original_ids
    }

def compare_files(file1, file2, dataset_base):
    """Compare two files and print the differences with formatted output."""
    if not os.path.isabs(file1):
        # Try current directory first
        if not os.path.exists(file1):
            # Try in the copy directory
            copy_path = f"{dataset_base}/copy/" + file1
            if os.path.exists(copy_path):
                file1 = copy_path

    if not os.path.isabs(file2):
        # Try current directory first
        if not os.path.exists(file2):
            # Try in the copy directory
            copy_path = f"{dataset_base}/copy/" + file2
            if os.path.exists(copy_path):
                file2 = copy_path

    # Run the analysis
    result = analyze_file_differences(file1, file2)

    # Print percentage with 2 decimal precision
    print(f"Difference percentage: {result['diff_percentage']:.2f}% with {len(result['different_original_ids'])} differences")
    return result['different_original_ids']

def find_one_hop_neighbors(uid, dataset_base):
    """Find one-hop neighbors of a user in the social graph."""
    friend_sequence = f'{dataset_base}/friend_sequence.txt'
    with open(friend_sequence, 'r') as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == uid:
                return parts[1:]
    return []

def get_textual_ids(user_ids, indexing_file):
    """Get textual representations for a list of user IDs from an indexing file."""
    textual_ids = {}
    with open(indexing_file, 'r') as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] in user_ids:
                textual_ids[parts[0]] = " ".join(parts[1:])
    return textual_ids[user_ids[0]]

def get_user_embedding(user_text, model_path, device='cpu'):
    """Get user embedding using T5 encoder."""
    if model_path.endswith('.pt'):
        base_model = "t5-small"
        config = T5Config.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = T5ForConditionalGeneration.from_pretrained(base_model, config=config)
        checkpoint = torch.load(model_path, map_location=device)
        vocab_size = checkpoint['shared.weight'].shape[0]
        model.resize_token_embeddings(vocab_size)
        model.load_state_dict(checkpoint)
    else:
        config = T5Config.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path, config=config)
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(user_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        encoder_outputs = model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
        last_hidden_states = encoder_outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand_as(last_hidden_states)
        masked_hidden_states = last_hidden_states * attention_mask
        sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
        sum_attention_mask = torch.clamp(torch.sum(attention_mask, dim=1), min=1e-9)
        return (sum_hidden_states / sum_attention_mask).cpu().numpy()

def load_model(model_path, device='cpu'):
    """Load the T5 model once and return model and tokenizer."""
    if model_path.endswith('.pt'):
        base_model = "t5-small"
        config = T5Config.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = T5ForConditionalGeneration.from_pretrained(base_model, config=config)
        checkpoint = torch.load(model_path, map_location=device)
        vocab_size = checkpoint['shared.weight'].shape[0]
        model.resize_token_embeddings(vocab_size)
        model.load_state_dict(checkpoint)
    else:
        config = T5Config.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path, config=config)
    
    model.to(device)
    model.eval()
    return model, tokenizer

@torch.no_grad()
def get_user_embedding_fast(texts, model, tokenizer, device='cpu', max_length=512):
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    encoder_out = model.encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        return_dict=True
    ).last_hidden_state  # (batch, seq, hidden)

    # mean-pool only over valid tokens
    mask = inputs.attention_mask.unsqueeze(-1)  # (batch, seq, 1)
    masked = encoder_out * mask
    pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    return pooled.detach().cpu().numpy()

def get_user_embedding_sentence_transformer(texts, model, device='cpu'):
    """Get user embedding using Sentence Transformer sentence-t5-base model."""
    if isinstance(texts, str):
        texts = [texts]
    
    # Move model to device if needed
    if hasattr(model, 'to'):
        model = model.to(device)
    
    # Get embeddings
    embeddings = model.encode(texts, convert_to_tensor=True, device=device)
    
    # Convert to numpy array
    if hasattr(embeddings, 'cpu'):
        return embeddings.cpu().numpy()
    else:
        return embeddings

def load_sentence_transformer_model(model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
    """Load the Sentence Transformer model."""
    model = SentenceTransformer(model_name)
    if device != 'cpu' and torch.cuda.is_available():
        model = model.to(device)
    return model


def main():
    DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    DATASET = 'lastfm_full_10_10'
    DATASET_BASE = f'rec_datasets/{DATASET}'
    MODEL_PATH = '/home/derrick/idgenrec_singlegpu/model/lastfm_train_20251010_100144/model_rec_round2_final.pt' #original
    file1 = os.path.join(DATASET_BASE, "original_noleak/user_generative_index_phase_2.txt")
    file2 = os.path.join(DATASET_BASE, "lastfm_socialtoid_1prompt/user_generative_index_phase_2.txt")
    

    # DATASET = 'yelp_30_2_5'
    # DATASET_BASE = f'rec_datasets/{DATASET}'
    # MODEL_PATH = '/home/derrick/idgenrec_singlegpu/model/yelp_30_2_5_train_20251015_103424/model_rec_round2_final.pt' #original
    # file1 = os.path.join(DATASET_BASE, "yelp_original/user_generative_index_phase_2.txt")
    # file2 = os.path.join(DATASET_BASE, "yelp_socialtoid/user_generative_index_phase_2.txt")
    
    print(f"File1: {file1}")
    print(f"File2: {file2}")
    #sentence_transformer_model = load_sentence_transformer_model(device=DEVICE)
    model, tokenizer = load_model(MODEL_PATH, DEVICE)
    print("Model loaded successfully!")
    different_users = compare_files(file1, file2, DATASET_BASE)
    original = 0
    socialtoid = 0
    no_common = 0
    original_emb = []
    socialtoid_emb = []
    
    print(f"\nRunning comparison on {len(different_users)} users...")
    
    for u in different_users:
        fr_embeddings = []
        for fr in find_one_hop_neighbors(u, DATASET_BASE):
            if fr != u:
                fr_embeddings.append(get_user_embedding_fast(get_textual_ids([fr], file1), model, tokenizer, DEVICE))
        
        if fr_embeddings:
            avg_fr_embedding = np.mean(fr_embeddings, axis=0)
            
            # Get user embeddings from both approaches
            user_embedding_original = get_user_embedding_fast(get_textual_ids([u], file1), model, tokenizer, DEVICE)
            user_embedding_socialtoid = get_user_embedding_fast(get_textual_ids([u], file2), model, tokenizer, DEVICE)
            
            # Calculate similarities
            original_similarity = abs(cosine_similarity(user_embedding_original, avg_fr_embedding)[0][0])
            socialtoid_similarity = abs(cosine_similarity(user_embedding_socialtoid, avg_fr_embedding)[0][0])
            print(f"Original similarity: {original_similarity:.4f}, Social-to-ID similarity: {socialtoid_similarity:.4f}")
            if original_similarity > socialtoid_similarity:
                original += 1
                print(f'Original wins by {original_similarity / socialtoid_similarity*100 - 100:.2f}%')
                original_emb.append([original_similarity,original_similarity / socialtoid_similarity*100 - 100])
            else:
                socialtoid += 1
                print(f'Social-to-ID wins by {socialtoid_similarity / original_similarity*100 - 100:.2f}%')
                socialtoid_emb.append([socialtoid_similarity,socialtoid_similarity / original_similarity*100 - 100])
        else:
            no_common += 1
    
    print(f"\nResults:")
    print(f'Dataset: {DATASET}')
    print(f"Original approach wins: {original}")
    print(f"Social-to-ID approach wins: {socialtoid}")
    print(f"Users with no friends: {no_common}")
    print(f"Total users analyzed: {original + socialtoid + no_common}")
    print(f'Standard deviation of original similarities: {np.std([x[0] for x in original_emb]):.4f} and mean percentage improvement {np.mean([x[1] for x in original_emb]):.2f}%')
    print(f'Standard deviation of social-to-ID similarities: {np.std([x[0] for x in socialtoid_emb]):.4f} and mean percentage improvement {np.mean([x[1] for x in socialtoid_emb]):.2f}%')
    # Calculate percentages
    total = original + socialtoid
    if total > 0:
        print(f"\nPercentage breakdown:")
        print(f"Original approach: {original/total*100:.2f}%")
        print(f"Social-to-ID approach: {socialtoid/total*100:.2f}%")

if __name__ == "__main__":
    main()
