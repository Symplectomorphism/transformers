"""
This is mostly a copy of https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

GPT model:
- Starts with token and positional embedding
- Transformer blocks
  - Sequential combination of a 1-hidden-layer MLP (KAN?) and a self-attention block
  - All blocks feed into a central residual pathway similar to ResNets
- The final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
# import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

# logger = logging.getLogger(__name__)

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). 
    Reference: Gaussian Error Linear Units (Hendrycks et al, https://arxiv.org/abs/1606.08415)
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here, but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * ( 1.0 / math.sqrt(k.size(-1)) )
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNarm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # ML forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model, with a context size of block_size """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        # Decision Transformer additional positional embedding
        C.max_timestep = 20
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':  dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large' : dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':  dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':    dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':   dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':    dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

            self.transformer = nn.ModuleDict(dict(
                te = nn.Embedding(config.vocab_size, config.n_embd),   # token embedding
                pe = nn.Embedding(config.block_size, config.n_embd),   # position embedding
                gpe = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd)),
                drop = nn.Dropout(config.embd_pdrop),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd),
            ))
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

            # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

            # Put in state, action and reward embeddings here.
            #

            # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight'):
                    nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

            # report number of parameters (note we don't count the decoder parameters in lm_head)
            n_params = sum(p.numel() for p in self.transformer.parameters())
            print("number of parameters: %.2fM" % (n_params/1e6,))

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

        def get_block_size(self):
            return self.block_size