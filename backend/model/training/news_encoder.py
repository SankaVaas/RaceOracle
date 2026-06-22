"""
Sentence-BERT with model surgery.

Surgery:
  - Layers 0..7 (first 8 of 12) are FROZEN
  - CLS pooling is REMOVED
  - Replaced with 3 sport-domain attention heads:
      head 0 - injury signal detector
      head 1 - travel/logistics flag
      head 2 - fatigue/condition signal
  - Output: 64-dim news embedding + 3 risk scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


class SportRiskAttentionHead(nn.Module):
    """
    A single sport-domain attention head.
    Learns to attend to the tokens most relevant to one risk category.
    """
    def __init__(self, hidden_dim: int, risk_type: str):
        super().__init__()
        self.risk_type = risk_type
        # Query vector: a learnable "what does this risk look like" vector
        self.query     = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.key_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.val_proj  = nn.Linear(hidden_dim, hidden_dim // 4, bias=False)
        self.scale     = hidden_dim ** 0.5

    def forward(self, token_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # token_embeddings: (B, seq_len, hidden_dim)
        keys   = self.key_proj(token_embeddings)                        # (B, seq, H)
        values = self.val_proj(token_embeddings)                        # (B, seq, H//4)
        query  = self.query.unsqueeze(0).unsqueeze(0)                   # (1, 1, H)
        scores = (query * keys).sum(dim=-1) / self.scale               # (B, seq)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights    = F.softmax(scores, dim=-1)                         # (B, seq)
        context    = (weights.unsqueeze(-1) * values).sum(dim=1)       # (B, H//4)
        risk_score = torch.sigmoid(context.mean(dim=-1, keepdim=True)) # (B, 1)
        return context, risk_score, weights


class SurgicalNewsEncoder(nn.Module):
    """
    Sentence-BERT with CLS pooling surgically replaced by 3 risk-detection heads.

    In production: load real SBERT weights, freeze layers 0-7, attach our heads.
    In CPU/dev mode: uses a lightweight surrogate transformer instead.
    """

    FREEZE_LAYERS = 8
    SBERT_MODEL   = "all-MiniLM-L6-v2"   # 22MB, runs well on CPU
    HIDDEN_DIM    = 384                    # MiniLM hidden size

    def __init__(self, out_dim: int = 64, use_pretrained: bool = True):
        super().__init__()
        self.out_dim = out_dim

        if use_pretrained and SBERT_AVAILABLE:
            self._load_sbert_backbone()
        else:
            self._build_surrogate_backbone()

        # MODEL SURGERY: 3 sport-domain risk heads replace CLS pooling
        self.risk_heads = nn.ModuleList([
            SportRiskAttentionHead(self.HIDDEN_DIM, "injury"),
            SportRiskAttentionHead(self.HIDDEN_DIM, "travel"),
            SportRiskAttentionHead(self.HIDDEN_DIM, "fatigue"),
        ])
        self.risk_names = ["injury", "travel", "fatigue"]

        # Project concatenated head outputs to fusion embedding dim
        head_out_dim = (self.HIDDEN_DIM // 4) * 3
        self.fusion_proj = nn.Sequential(
            nn.Linear(head_out_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def _load_sbert_backbone(self):
        """Load pretrained SBERT and freeze first FREEZE_LAYERS layers."""
        sbert = SentenceTransformer(self.SBERT_MODEL)
        # Extract the transformer component
        self.backbone = sbert[0].auto_model
        # Freeze layers 0..FREEZE_LAYERS-1
        encoder_layers = self.backbone.encoder.layer
        for i, layer in enumerate(encoder_layers):
            if i < self.FREEZE_LAYERS:
                for param in layer.parameters():
                    param.requires_grad = False
        # Always freeze embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        self.tokenizer  = sbert[0].tokenizer
        self.use_sbert  = True

    def _build_surrogate_backbone(self):
        """Lightweight surrogate for CPU dev / no-internet environments."""
        self.backbone   = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.HIDDEN_DIM, nhead=6, dim_feedforward=1024,
                dropout=0.1, batch_first=True
            ),
            num_layers=6
        )
        self.token_emb  = nn.Embedding(30522, self.HIDDEN_DIM)  # BERT vocab size
        self.pos_emb    = nn.Embedding(512, self.HIDDEN_DIM)
        self.use_sbert  = False

    def encode_text(self, texts: list, device: str = "cpu"):
        """Encode raw text strings to token embeddings (B, seq, hidden)."""
        if self.use_sbert:
            encoded = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = self.backbone(**encoded)
            return out.last_hidden_state, encoded["attention_mask"]
        else:
            # Surrogate: just embed random tokens (replace with real tokenizer in prod)
            B = len(texts)
            seq = 32
            ids  = torch.zeros(B, seq, dtype=torch.long, device=device)
            pos  = torch.arange(seq, device=device).unsqueeze(0).expand(B, -1)
            emb  = self.token_emb(ids) + self.pos_emb(pos)
            mask = torch.ones(B, seq, device=device)
            return self.backbone(emb), mask

    def forward(self, texts: list, device: str = "cpu"):
        token_embs, attn_mask = self.encode_text(texts, device)

        # Apply 3 risk heads (MODEL SURGERY output)
        contexts, risk_scores, attentions = [], [], []
        for head in self.risk_heads:
            ctx, score, attn_w = head(token_embs, attn_mask)
            contexts.append(ctx)
            risk_scores.append(score)
            attentions.append(attn_w)

        # Fuse head contexts into embedding
        concat  = torch.cat(contexts, dim=-1)          # (B, 3 * H//4)
        emb     = self.fusion_proj(concat)             # (B, out_dim=64)

        # Risk scores: (B, 3) — injury/travel/fatigue probabilities
        risks   = torch.cat(risk_scores, dim=-1)       # (B, 3)

        return {
            "embedding":   emb,       # 64-dim news embedding for fusion layer
            "risk_scores": risks,     # (B, 3) injury/travel/fatigue
            "attentions":  attentions # (3,) list of (B, seq) attention maps
        }
