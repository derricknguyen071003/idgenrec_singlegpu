import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple, Union

def add_timestep_tokens_to_tokenizer(tokenizer, num_timesteps: int):
    timestep_tokens = [f"[TIMESTEP_{i}]" for i in range(num_timesteps)]
    tokenizer.add_tokens(timestep_tokens, special_tokens=True)
    timestep_token_ids = {
        i: tokenizer.convert_tokens_to_ids(f"[TIMESTEP_{i}]")
        for i in range(num_timesteps)
    }
    logging.info(f"Added {num_timesteps} timestep tokens to tokenizer. New vocab size: {len(tokenizer)}")
    return timestep_token_ids


def prepend_timestep_token(
    input_ids: torch.Tensor,
    timesteps: torch.Tensor,
    timestep_token_ids: dict,
    attention_mask: torch.Tensor
) -> tuple:
    batch_size = input_ids.shape[0]
    device = input_ids.device
    timestep_token_ids_tensor = torch.tensor(
        [timestep_token_ids[t.item()] for t in timesteps],
        device=device,
        dtype=torch.long
    ).unsqueeze(1)
    new_input_ids = torch.cat([timestep_token_ids_tensor, input_ids], dim=1)
    timestep_mask = torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
    new_attention_mask = torch.cat([timestep_mask, attention_mask], dim=1)
    return new_input_ids, new_attention_mask


class DiscreteDiffusionScheduler:
    def __init__(self, num_timesteps: int = 100, beta_max: float = 0.1):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(0, beta_max, num_timesteps)  # β_t
        self.alpha_bars = self._compute_alpha_bars()             # \bar α_t

    def _compute_alpha_bars(self) -> torch.Tensor:
        alphas = 1.0 - self.betas
        return torch.cumprod(alphas, dim=0)

    def get_alpha_bar(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        t = torch.as_tensor(t).clamp(0, self.num_timesteps - 1).long()
        return self.alpha_bars[t]


def corrupt_sequence(
    clean_sequence: torch.Tensor,          # shape: [seq_len]
    timestep: int,
    scheduler: DiscreteDiffusionScheduler,
    vocab_size: int,
    cross_view_tokens: Optional[torch.Tensor] = None,  # shape: [num_tokens]
    cross_view_prob: float = 0.5,
    pad_token_id: int = 0,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)

    device = clean_sequence.device
    seq_len = clean_sequence.size(0)

    # \bar α_t from scheduler (survival probability)
    alpha_bar = scheduler.get_alpha_bar(int(timestep))
    if not isinstance(alpha_bar, torch.Tensor):
        alpha_bar = torch.tensor(alpha_bar, device=device, dtype=torch.float32)
    else:
        alpha_bar = alpha_bar.to(device).float()

    corrupt_prob = 1.0 - alpha_bar  # P(corrupt)

    # Sample which positions are corrupted
    noise_mask = (torch.rand(seq_len, device=device) < corrupt_prob).long()
    noise_mask *= (clean_sequence != pad_token_id).long()   # never corrupt PAD

    corrupted = clean_sequence.clone()
    corrupted_pos = noise_mask.bool()

    # Prepare cross-view tokens (if any)
    if cross_view_tokens is not None:
        cross_view_tokens = cross_view_tokens.to(device)
        cross_view_tokens = cross_view_tokens[cross_view_tokens != pad_token_id]

    if corrupted_pos.any():
        # Decide cross-view vs random among corrupted positions
        use_cross = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if cross_view_tokens is not None and cross_view_tokens.numel() > 0:
            use_cross = (torch.rand(seq_len, device=device) < cross_view_prob) & corrupted_pos

            # Fill cross-view positions
            n_cv = use_cross.sum().item()
            if n_cv > 0:
                idx = use_cross.nonzero(as_tuple=True)[0]
                sampled_cv = cross_view_tokens[
                    torch.randint(0, cross_view_tokens.numel(), (n_cv,), device=device)
                ]
                corrupted[idx] = sampled_cv

        # Remaining corrupted positions → random vocab tokens
        remaining = corrupted_pos & ~use_cross
        n_rand = remaining.sum().item()
        if n_rand > 0:
            idx = remaining.nonzero(as_tuple=True)[0]
            sampled_rand = torch.randint(1, vocab_size, (n_rand,), device=device)
            corrupted[idx] = sampled_rand

    return corrupted, noise_mask


def get_cross_view_tokens(
    item_sequence: torch.Tensor,
    social_sequence: torch.Tensor,
    pad_token_id: int = 0
) -> torch.Tensor:
    item_tokens = item_sequence[item_sequence != pad_token_id].unique()
    social_tokens = social_sequence[social_sequence != pad_token_id].unique()
    return social_tokens[~torch.isin(social_tokens, item_tokens)]


def compute_kl_divergence(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temperature: float = 1.0,
    epsilon: float = 1e-8,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute KL divergence KL(P || Q) between two probability distributions.
    
    Args:
        logits_p: Logits for distribution P (item view)
            Shape: [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
        logits_q: Logits for distribution Q (social view)
            Shape: [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
        temperature: Temperature for softmax (default: 1.0)
        epsilon: Small value for numerical stability (default: 1e-8)
        reduction: Reduction method ('mean', 'sum', or 'none')
    
    Returns:
        kl_div: KL divergence KL(P || Q)
            Shape: [batch_size] if reduction='none', scalar otherwise
    """
    # Apply temperature scaling
    logits_p = logits_p / temperature
    logits_q = logits_q / temperature
    
    # Convert to probabilities with numerical stability
    p = torch.softmax(logits_p, dim=-1)
    q = torch.softmax(logits_q, dim=-1)
    
    # Add epsilon to avoid log(0)
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    
    # Compute KL divergence: KL(P || Q) = sum(P * log(P / Q))
    # = sum(P * log(P)) - sum(P * log(Q))
    log_p = torch.log(p)
    log_q = torch.log(q)
    
    kl_div = p * (log_p - log_q)
    
    # Sum over vocabulary dimension
    if kl_div.dim() == 2:  # [batch_size, vocab_size]
        kl_div = kl_div.sum(dim=-1)
    elif kl_div.dim() == 3:  # [batch_size, seq_len, vocab_size]
        kl_div = kl_div.sum(dim=-1)  # [batch_size, seq_len]
        # Optionally average over sequence length
        # kl_div = kl_div.mean(dim=-1)  # [batch_size]
    
    # Apply reduction
    if reduction == 'mean':
        return kl_div.mean()
    elif reduction == 'sum':
        return kl_div.sum()
    elif reduction == 'none':
        return kl_div
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class NoisePredictionHead(nn.Module):
    """
    Binary classification head for predicting noise mask.
    
    Takes T5 encoder hidden states and outputs per-position binary predictions
    indicating whether each token is corrupted (1) or clean (0).
    
    Args:
        hidden_dim: Hidden dimension of encoder outputs (default: 512 for T5-small)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, hidden_dim: int = 512, dropout: float = 0.1):
        super(NoisePredictionHead, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Simple MLP: hidden_dim -> hidden_dim -> 1
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            encoder_hidden_states: Encoder hidden states from T5
                Shape: [batch_size, seq_len, hidden_dim]
        
        Returns:
            noise_logits: Binary logits for each position
                Shape: [batch_size, seq_len]
        """
        # Apply MLP
        x = self.linear1(encoder_hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        # Squeeze last dimension: [batch_size, seq_len, 1] -> [batch_size, seq_len]
        noise_logits = x.squeeze(-1)
        
        return noise_logits


def create_noise_head(model, hidden_dim: int = None, dropout: float = 0.1) -> NoisePredictionHead:
    """
    Create a noise prediction head with appropriate hidden dimension.
    
    Args:
        model: T5 model to extract hidden dimension from
        hidden_dim: Explicit hidden dimension (if None, inferred from model)
        dropout: Dropout probability
    
    Returns:
        NoisePredictionHead instance
    """
    if hidden_dim is None:
        # Infer hidden dimension from model config
        if hasattr(model, 'config'):
            hidden_dim = model.config.d_model
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'config'):
            hidden_dim = model.encoder.config.d_model
        else:
            # Default for T5-small
            hidden_dim = 512
            logging.warning(f"Could not infer hidden_dim from model, using default: {hidden_dim}")
    
    logging.info(f"Creating NoisePredictionHead with hidden_dim={hidden_dim}, dropout={dropout}")
    return NoisePredictionHead(hidden_dim=hidden_dim, dropout=dropout)

