"""
TabTransformer with model surgery.

Architecture:
  - Layers 1..FREEZE_DEPTH are frozen after pretraining
  - Original MHA heads are REMOVED and replaced with RaceContextAttention
  - A custom FormDecayFFN replaces the standard FFN at the top block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CategoricalEmbedding(nn.Module):
    """Embeds each categorical feature into a d_model-dim vector."""
    def __init__(self, cat_cardinalities: list, d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, d_model)
            for cardinality in cat_cardinalities
        ])
        self.col_tokens = nn.Parameter(torch.randn(len(cat_cardinalities), d_model) * 0.02)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        out = torch.stack(embedded, dim=1)
        out = out + self.col_tokens.unsqueeze(0)
        return out


class StandardTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, recency_weights=None):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class RaceContextAttention(nn.Module):
    """
    MODEL SURGERY: Replaces generic MHA with 4 domain-specific heads.
      head 0 - recent form trajectory
      head 1 - going preference alignment
      head 2 - distance curve fit
      head 3 - jockey-horse partnership synergy
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads  = 4
        self.head_dim = d_model // self.n_heads
        self.d_model  = d_model
        assert d_model % self.n_heads == 0, "d_model must be divisible by 4"

        self.q_projs = nn.ModuleList([nn.Linear(d_model, self.head_dim, bias=False) for _ in range(self.n_heads)])
        self.k_projs = nn.ModuleList([nn.Linear(d_model, self.head_dim, bias=False) for _ in range(self.n_heads)])
        self.v_projs = nn.ModuleList([nn.Linear(d_model, self.head_dim, bias=False) for _ in range(self.n_heads)])

        # Learnable race domain bias per head
        self.head_bias = nn.Parameter(torch.zeros(self.n_heads, 1, 1))
        self.out_proj  = nn.Linear(d_model, d_model)
        self.drop      = nn.Dropout(dropout)
        self.scale     = math.sqrt(self.head_dim)

        self.head_names = ["form_trajectory", "going_preference", "distance_curve", "jockey_synergy"]

    def forward(self, x: torch.Tensor):
        head_outputs = []
        for i in range(self.n_heads):
            q = self.q_projs[i](x)
            k = self.k_projs[i](x)
            v = self.v_projs[i](x)
            scores  = torch.bmm(q, k.transpose(1, 2)) / self.scale
            scores  = scores + self.head_bias[i]
            weights = self.drop(F.softmax(scores, dim=-1))
            head_outputs.append(torch.bmm(weights, v))
        return self.out_proj(torch.cat(head_outputs, dim=-1))


class FormDecayFFN(nn.Module):
    """
    MODEL SURGERY: Replaces standard FFN with form-decay aware network.
    Applies learnable exponential decay so recent runs matter more.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1       = nn.Linear(d_model, d_model * 4)
        self.fc2       = nn.Linear(d_model * 4, d_model)
        self.drop      = nn.Dropout(dropout)
        self.log_decay = nn.Parameter(torch.tensor(-0.1))

    def forward(self, x: torch.Tensor, recency_weights: Optional[torch.Tensor] = None):
        h = self.drop(F.gelu(self.fc1(x)))
        h = self.fc2(h)
        if recency_weights is not None:
            decay = torch.exp(self.log_decay) * recency_weights
            h = h * decay
        return h


class SurgicalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, surgical: bool = False, freeze: bool = False, dropout: float = 0.1):
        super().__init__()
        self.surgical = surgical

        if surgical:
            self.attn = RaceContextAttention(d_model, dropout)
            self.ff   = FormDecayFFN(d_model, dropout)
        else:
            self.attn = nn.MultiheadAttention(d_model, 8, dropout=dropout, batch_first=True)
            self.ff   = nn.Sequential(
                nn.Linear(d_model, d_model * 4), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
            )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x, recency_weights=None):
        if self.surgical:
            attn_out = self.attn(x)
        else:
            attn_out, _ = self.attn(x, x, x)

        x = self.norm1(x + self.drop(attn_out))

        if self.surgical and isinstance(self.ff, FormDecayFFN):
            ff_out = self.ff(x, recency_weights)
        else:
            ff_out = self.ff(x)

        x = self.norm2(x + self.drop(ff_out))
        return x


class SurgicalTabTransformer(nn.Module):
    """
    TabTransformer with model surgery:
      Layers 0..FREEZE_DEPTH-1  = frozen standard blocks
      Layers FREEZE_DEPTH..end  = surgical blocks (custom heads + FormDecayFFN)
    Outputs 128-dim race embedding for the fusion layer.
    """
    FREEZE_DEPTH = 6

    def __init__(self, cat_cardinalities: list, num_continuous: int,
                 d_model: int = 64, n_layers: int = 8,
                 dropout: float = 0.1, out_dim: int = 128):
        super().__init__()
        self.cat_embedding   = CategoricalEmbedding(cat_cardinalities, d_model)
        self.cont_projection = nn.Linear(num_continuous, d_model)
        self.cont_norm       = nn.LayerNorm(d_model)

        layers = []
        for i in range(n_layers):
            surgical = (i >= self.FREEZE_DEPTH)
            block = SurgicalTransformerBlock(d_model, surgical=surgical, freeze=not surgical, dropout=dropout)
            layers.append(block)
        self.layers = nn.ModuleList(layers)

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
            nn.GELU(),
        )

    def freeze_pretrained_layers(self):
        self.cat_embedding.requires_grad_(False)
        self.cont_projection.requires_grad_(False)
        for i in range(self.FREEZE_DEPTH):
            self.layers[i].requires_grad_(False)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor,
                recency_weights: Optional[torch.Tensor] = None):
        cat_emb  = self.cat_embedding(x_cat)
        cont_emb = self.cont_norm(self.cont_projection(x_cont)).unsqueeze(1)
        x = torch.cat([cat_emb, cont_emb], dim=1)

        for i, layer in enumerate(self.layers):
            rw = recency_weights if i >= self.FREEZE_DEPTH else None
            x = layer(x, rw)

        x = x.mean(dim=1)
        return self.output_head(x)
