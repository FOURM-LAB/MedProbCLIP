import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, PreTrainedModel
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    import timm
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False


@dataclass
class MedProbClipConfig:
    embed_dim: int = 512
    img_backbone: str = "vit_base_patch16_224"
    text_backbone: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    proj_hidden: int = 0
    freeze_img: bool = False
    freeze_text: bool = False
    max_txt_len: int = 256

# --- Re-usable components from PCME++ ---

class ProbProjector(nn.Module):
    """Linear (or tiny MLP) → (mu, logvar); dimension = embed_dim."""
    def __init__(self, in_dim: int, embed_dim: int, hidden: int = 0):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(),
                nn.Linear(hidden, 2 * embed_dim),
            )
        else:
            self.net = nn.Linear(in_dim, 2 * embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, torch.clamp(logvar, min=-6.0, max=6.0)

class ImageEncoder(nn.Module):
    def __init__(self, backbone: str, embed_dim: int, proj_hidden: int = 0, freeze: bool = False):
        super().__init__()
        assert HAS_TIMM, "Please `pip install timm`."
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        in_dim = self.backbone.num_features
        self.proj = ProbProjector(in_dim, embed_dim, proj_hidden)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        feat = self.backbone(x)
        mu, logvar = self.proj(feat)
        return mu, logvar

class TextEncoder(nn.Module):
    def __init__(self, mdl_name: str, embed_dim: int, proj_hidden: int = 0, freeze: bool = False):
        super().__init__()
        assert HAS_TRANSFORMERS, "Please `pip install transformers`."
        self.bert: PreTrainedModel = AutoModel.from_pretrained(mdl_name, use_safetensors=True, trust_remote_code=False)
        in_dim = self.bert.config.hidden_size
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.proj = ProbProjector(in_dim, embed_dim, proj_hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = out.last_hidden_state[:, 0]  # CLS token
        mu, logvar = self.proj(pooled)
        return mu, logvar

# --- Main MedProbClip Model ---

class MedProbClip(nn.Module):
    """
    MedProbClip: A probabilistic cross-modal model for multi-view learning.
    - Encodes multiple views of images and texts into probabilistic embeddings.
    - Similarity is calculated using CSD, designed for a multi-view contrastive loss.
    """
    def __init__(self, cfg: MedProbClipConfig):
        super().__init__()
        self.cfg = cfg
        self.img_enc = ImageEncoder(cfg.img_backbone, cfg.embed_dim, cfg.proj_hidden, cfg.freeze_img)
        self.txt_enc = TextEncoder(cfg.text_backbone, cfg.embed_dim, cfg.proj_hidden, cfg.freeze_text)
        
        # Learnable parameters for CSD scaling
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward_img(self, x):
        return self.img_enc(x)

    def forward_txt(self, input_ids, attention_mask):
        return self.txt_enc(input_ids, attention_mask)
    
    def forward(self, img1, txt1_ids, txt1_mask, img2, txt2_ids, txt2_mask):
        # Encode all four inputs into probabilistic embeddings
        mu_i1, lv_i1 = self.forward_img(img1)
        mu_t1, lv_t1 = self.forward_txt(txt1_ids, txt1_mask)
        mu_i2, lv_i2 = self.forward_img(img2)
        mu_t2, lv_t2 = self.forward_txt(txt2_ids, txt2_mask)
        
        return (mu_i1, lv_i1), (mu_t1, lv_t1), (mu_i2, lv_i2), (mu_t2, lv_t2)

    @staticmethod
    def csd(mu_x, logvar_x, mu_y, logvar_y, eps: float = 1e-6):
        sx = (logvar_x).exp().clamp_min(eps)
        sy = (logvar_y).exp().clamp_min(eps)
        dx = mu_x.unsqueeze(1) - mu_y.unsqueeze(0)
        s_sum = sx.unsqueeze(1) + sy.unsqueeze(0)
        d = 0.5 * ((dx * dx) / s_sum + (s_sum + eps).log()).sum(dim=-1)
        return d

    def match_logits(self, mu_i, logv_i, mu_t, logv_t):
        d = self.csd(mu_i, logv_i, mu_t, logv_t)
        return -self.a.clamp_min(1e-3) * d + self.b
