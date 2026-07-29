from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 1
    n_head: int = 1
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.n_embd, self.dropout = cfg.n_head, cfg.n_embd, cfg.dropout
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        hs = c // self.n_head
        q = self.q_proj(x).view(b, t, self.n_head, hs).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_head, hs).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.out_proj(y))

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init)

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError("sequence exceeds block_size")
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
