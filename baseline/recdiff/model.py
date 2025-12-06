import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sparse_mm(sparse, dense):
    """Sparse matrix multiplication"""
    return torch.sparse.mm(sparse, dense)


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        nn.init.xavier_uniform_(self.weight)
        self.activation = activation

    def forward(self, adj, u_f, v_f):
        node_f = torch.cat([u_f, v_f], dim=0)
        node_f = torch.mm(node_f, self.weight)
        node_f = sparse_mm(adj, node_f)
        if self.activation:
            node_f = self.activation(node_f)
        return node_f


class SocialGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        nn.init.xavier_uniform_(self.weight)
        self.activation = activation

    def forward(self, adj, u_f):
        u_f = torch.mm(u_f, self.weight)
        u_f = sparse_mm(adj, u_f)
        if self.activation:
            u_f = self.activation(u_f)
        return u_f


class GCNModel(nn.Module):
    def __init__(self, n_user, n_item, n_hid=64, n_layers=2, s_layers=2):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_hid = n_hid
        
        self.user_emb = nn.Parameter(torch.randn(n_user, n_hid))
        self.item_emb = nn.Parameter(torch.randn(n_item, n_hid))
        nn.init.xavier_uniform_(self.user_emb)
        nn.init.xavier_uniform_(self.item_emb)
        
        act = nn.LeakyReLU(0.5)
        self.ui_layers = nn.ModuleList([GCNLayer(n_hid, n_hid, act) for _ in range(n_layers)])
        self.social_layers = nn.ModuleList([SocialGCNLayer(n_hid, n_hid, act) for _ in range(s_layers)])

    def forward(self, ui_adj, social_adj):
        # User-item embeddings
        u_emb, i_emb = self.user_emb, self.item_emb
        ui_embs = [torch.cat([u_emb, i_emb], dim=0)]
        
        for layer in self.ui_layers:
            emb = layer(ui_adj, u_emb, i_emb)
            u_emb, i_emb = emb[:self.n_user], emb[self.n_user:]
            ui_embs.append(F.normalize(emb, p=2, dim=1))
        ui_emb = sum(ui_embs)
        
        # Social embeddings
        u_emb = self.user_emb
        social_embs = [u_emb]
        for layer in self.social_layers:
            u_emb = layer(social_adj, u_emb)
            social_embs.append(F.normalize(u_emb, p=2, dim=1))
        social_emb = sum(social_embs)
        
        return ui_emb, social_emb


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device).float() / half)
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class SDNet(nn.Module):
    def __init__(self, dim=64, emb_size=16):
        super().__init__()
        self.time_emb = nn.Linear(emb_size, emb_size)
        self.layers = nn.Sequential(
            nn.Linear(dim + emb_size, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, timesteps):
        t_emb = timestep_embedding(timesteps, self.time_emb.in_features)
        t_emb = self.time_emb(t_emb)
        h = torch.cat([x, t_emb], dim=-1)
        return self.layers(h)


class DiffusionProcess(nn.Module):
    def __init__(self, steps=20, noise_scale=0.1, noise_min=0.0001, noise_max=0.01, device='cuda'):
        super().__init__()
        self.steps = steps
        self.device = device
        
        # Beta schedule
        st_bound = noise_scale * noise_min
        e_bound = noise_scale * noise_max
        betas = torch.linspace(st_bound, e_bound, steps, device=device)
        
        alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def forward_process(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def compute_loss(self, model, x_0, reweight=True):
        batch_size = x_0.size(0)
        t = torch.randint(0, self.steps, (batch_size,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.forward_process(x_0, t, noise)
        pred_x_0 = model(x_t, t)
        
        loss = (pred_x_0 - x_0).pow(2).mean(dim=-1)
        if reweight:
            weight = (self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])).clamp(min=0.01)
            loss = loss * weight
        return loss.mean(), pred_x_0

    def sample(self, model, x_0, steps=0):
        if steps == 0:
            return x_0
        t = torch.full((x_0.size(0),), steps - 1, device=x_0.device)
        x_t = self.forward_process(x_0, t)
        
        for i in range(steps - 1, -1, -1):
            t = torch.full((x_t.size(0),), i, device=x_t.device)
            pred_x_0 = model(x_t, t)
            if i > 0:
                alpha_t = self.alphas_cumprod[i]
                alpha_prev = self.alphas_cumprod[i-1]
                beta_t = 1 - alpha_t / alpha_prev
                x_t = torch.sqrt(alpha_prev / alpha_t) * (x_t - torch.sqrt(beta_t) * (pred_x_0 - x_0)) + torch.sqrt(beta_t) * torch.randn_like(x_t)
            else:
                x_t = pred_x_0
        return x_t
