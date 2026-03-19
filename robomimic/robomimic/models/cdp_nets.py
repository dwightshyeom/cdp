import math
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
import einops

logger = logging.getLogger(__name__)

def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"Input has {x.ndim} dims, but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]

# Transformer building blocks
class SelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, attn_pdrop: float, resid_pdrop: float, block_size: int):
        super().__init__()
        assert n_embd % n_heads == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )
        self.n_head = n_heads

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y
    
class Block(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, attn_pdrop: float, resid_pdrop: float, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# Diffusion transformer model
class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        goal_conditioned: bool,
        embed_dim: int,
        embed_pdrop: float,
        attn_pdrop: float,
        resid_pdrop: float,
        n_layers: int,
        n_heads: int,
        goal_seq_len: int,
        obs_seq_len: int,
        goal_drop: float = 0.1,
        linear_output: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
            
        block_size = goal_seq_len + 2 * obs_seq_len + 1
        seq_size = goal_seq_len + obs_seq_len + 1
        
        self.tok_emb = nn.Linear(state_dim, embed_dim)
        if self.goal_conditioned:
            self.goal_emb = nn.Linear(goal_dim, embed_dim)
            
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrop)
        
        self.cond_mask_prob = goal_drop
        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim
        
        self.blocks = nn.Sequential(
            *[Block(embed_dim, n_heads, attn_pdrop, resid_pdrop, block_size) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        
        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        
        self.sigma_emb = nn.Linear(1, embed_dim)
        self.action_emb = nn.Linear(action_dim, embed_dim)
        
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100), 
                nn.SiLU(),  
                nn.Linear(100, self.action_dim)
            )
            
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, DiffusionTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def forward(self, states, actions, goals, sigma, uncond: Optional[bool] = False, keep_last_actions: Optional[bool] = False):  
        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, 'b -> b 1')
        emb_t = self.sigma_emb(sigmas.to(torch.float32))
        
        if len(states.shape) == 3 and len(emb_t.shape) == 2:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
              
        if self.goal_conditioned:
            second_half_idx = self.goal_seq_len + 1 
        else:
            second_half_idx = 1
            
        if self.training:
            goals = self.mask_cond(goals)
        if uncond:
            goals = torch.zeros_like(goals).to(self.device)  

        state_embed = self.tok_emb(states)
        action_embed = self.action_emb(actions)
        
        if self.goal_conditioned:
            goal_embed = self.goal_emb(goals)
            position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len), :]
            goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        else:
            position_embeddings = self.pos_emb[:, :t, :]
            
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:, :])
        action_x = self.drop(action_embed + position_embeddings[:, self.goal_seq_len:, :])
        
        sa_seq = torch.stack([state_x, action_x], dim=1).permute(0, 2, 1, 3).reshape(b, 2*t, self.embed_dim)
        
        if self.goal_conditioned:
            input_seq = torch.cat([emb_t, goal_x, sa_seq], dim=1)
        else:
            input_seq = torch.cat([emb_t, sa_seq], dim=1)
        
        x = self.blocks(input_seq)
        x = self.ln_f(x)
        x = x[:, second_half_idx:, :]
        x = x.reshape(b, t, 2, self.embed_dim).permute(0, 2, 1, 3)
            
        action_outputs = x[:, 1]
        pred_actions = self.action_pred(action_outputs)
        
        if keep_last_actions:
            pred_actions = torch.cat([actions[:, :-1, :], pred_actions[:, -1, :].reshape(1, 1, -1)], dim=1)

        return pred_actions
    
    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob) 
            return cond * (1. - mask)
        else:
            return cond

# Karras Preconditioner Wrapper
class KDenoiser(nn.Module):
    """
    A Karras et al. preconditioner for denoising diffusion models.
    """
    def __init__(self, inner_model: nn.Module, sigma_data=0.5):
        super().__init__()
        # Takes an already instantiated PyTorch module instead of a Hydra config
        self.inner_model = inner_model
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, state, action, goal, noise, sigma, **kwargs):
        pred_last = False
        if 'pred_last_action_only' in kwargs.keys():
            if kwargs['pred_last_action_only']:
                pred_last = True
                noise[:, :-1, :] = 0
            kwargs.pop('pred_last_action_only')

        noised_input = action + noise * append_dims(sigma, action.ndim)
            
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        
        model_output = self.inner_model(state, noised_input * c_in, goal, sigma, **kwargs)
        target = (action - c_skip * noised_input) / c_out
        
        if pred_last:
            return (model_output[:, -1, :] - target[:, -1, :]).pow(2).mean()
        else:
            return (model_output - target).pow(2).flatten(1).mean()

    def forward(self, state, action, goal, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(state, action * c_in, goal, sigma, **kwargs) * c_out + action * c_skip