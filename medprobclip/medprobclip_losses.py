from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Re-usable loss components from PCME++ ---

def vib_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Calculates the KL Divergence between N(mu, sigma^2) and N(0, I)."""
    kl = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar)
    return kl.sum(dim=-1).mean()

def pseudo_positive_mask(logits: torch.Tensor, k: int = 8, thresh: float = None) -> torch.Tensor:
    """Identifies pseudo-positive pairs from high-scoring off-diagonal entries for label smoothing."""
    B = logits.size(0)
    device = logits.device
    with torch.no_grad():
        off = logits - 1e9 * torch.eye(B, device=device)
        if thresh is not None:
            sel = (off > thresh).float()
        else:
            topk = off.topk(k=min(k, max(B - 1, 1)), dim=1).indices
            sel = torch.zeros_like(off)
            sel.scatter_(1, topk, 1.0)
    sel = sel * (1 - torch.eye(B, device=device))
    return sel

# --- MedProbClip Multi-Component Loss ---

def _symmetric_pcme_loss(mu1, lv1, mu2, lv2, model):
    """Calculates a symmetric BCE loss for two sets of probabilistic embeddings."""
    logits = model.match_logits(mu1, lv1, mu2, lv2)
    target = torch.arange(logits.size(0), dtype=torch.long, device=logits.device)
    loss_1_to_2 = F.cross_entropy(logits, target)
    loss_2_to_1 = F.cross_entropy(logits.t(), target)
    return (loss_1_to_2 + loss_2_to_1) / 2.0


class MedProbClipLoss(nn.Module):
    """
    Multi-component probabilistic loss for MedProbClip.
    """
    def __init__(self, beta_vib_I: float = 1e-4, beta_vib_T: float = 1e-4, alpha_pp: float = 0.1, pp_k: int = 8, pp_thresh: float = None, lambda_I: float = 1.0, lambda_T: float = 1.0):
        super().__init__()
        self.beta_vib_I = beta_vib_I
        self.beta_vib_T = beta_vib_T
        self.alpha_pp = alpha_pp
        self.pp_k = pp_k
        self.pp_thresh = pp_thresh
        self.lambda_I = lambda_I
        self.lambda_T = lambda_T

    def forward(self, embeddings: Tuple, model) -> Dict[str, torch.Tensor]:
        (mu_i1, lv_i1), (mu_t1, lv_t1), (mu_i2, lv_i2), (mu_t2, lv_t2) = embeddings
        
        # --- 1. Inter-modal Loss ---
        # Instead of one large matrix, calculate 4 separate losses to correctly handle all positive pairs
        loss_i1_t1 = _symmetric_pcme_loss(mu_i1, lv_i1, mu_t1, lv_t1, model)
        loss_i1_t2 = _symmetric_pcme_loss(mu_i1, lv_i1, mu_t2, lv_t2, model)
        loss_i2_t1 = _symmetric_pcme_loss(mu_i2, lv_i2, mu_t1, lv_t1, model)
        loss_i2_t2 = _symmetric_pcme_loss(mu_i2, lv_i2, mu_t2, lv_t2, model)
        loss_inter = (loss_i1_t1 + loss_i1_t2 + loss_i2_t1 + loss_i2_t2) / 4.0

        # --- 2. Intra-modal Losses ---
        loss_intra_I = _symmetric_pcme_loss(mu_i1, lv_i1, mu_i2, lv_i2, model)
        loss_intra_T = _symmetric_pcme_loss(mu_t1, lv_t1, mu_t2, lv_t2, model)

        # --- 3. VIB Regularization ---
        kl_img = (vib_kl(mu_i1, lv_i1) + vib_kl(mu_i2, lv_i2)) / 2.0
        kl_txt = (vib_kl(mu_t1, lv_t1) + vib_kl(mu_t2, lv_t2)) / 2.0
        loss_vib = self.beta_vib_I * kl_img + self.beta_vib_T * kl_txt
        
        # --- 4. Combine Losses ---
        total_loss = loss_inter + self.lambda_I * loss_intra_I + self.lambda_T * loss_intra_T + loss_vib
        
        return {
            "loss": total_loss,
            "loss_inter": loss_inter.detach(),
            "loss_intra_I": loss_intra_I.detach(),
            "loss_intra_T": loss_intra_T.detach(),
            "loss_vib": loss_vib.detach(),
            "kl_img": kl_img.detach(),
            "kl_txt": kl_txt.detach(),
        }
