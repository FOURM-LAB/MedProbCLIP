import argparse, os, sys
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
from PIL import ImageFile

from medprobclip_model import MedProbClip, MedProbClipConfig
from medprobclip_losses import MedProbClipLoss
from medprobclip.cxr_study_dataset import CXRStudyDataset, collate_mvs

ImageFile.LOAD_TRUNCATED_IMAGES = True

def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def eval_retrieval_chunked(model, loader, device, topk=(1,5,10,100), chunk_q=128, chunk_g=512):
    """
    Evaluation for MedProbClip.
    Uses the FIRST image and text from each study for a standard retrieval task.
    """
    m = unwrap(model)
    m.eval()

    all_mu_i, all_lv_i, all_mu_t, all_lv_t = [], [], [], []
    for batch in tqdm(loader, desc="Encoding for Eval"):
        # NOTE: We only use img1 and text1 for a standard, comparable evaluation
        pixels = batch["img1"].to(device, non_blocking=True)
        ids = batch["input_ids1"].to(device, non_blocking=True)
        att = batch["attention_mask1"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            mu_i, lv_i = m.forward_img(pixels)
            mu_t, lv_t = m.forward_txt(ids, att)

        all_mu_i.append(mu_i)
        all_lv_i.append(lv_i)
        all_mu_t.append(mu_t)
        all_lv_t.append(lv_t)

    mu_i_all, lv_i_all = torch.cat(all_mu_i, 0), torch.cat(all_lv_i, 0)
    mu_t_all, lv_t_all = torch.cat(all_mu_t, 0), torch.cat(all_lv_t, 0)
    N = mu_i_all.size(0)

    def recall_direction(mu_queries, lv_queries, mu_gallery, lv_gallery):
        Kmax = max(topk)
        ranks = []
        for q_start in range(0, N, chunk_q):
            q_end = min(q_start + chunk_q, N)
            mu_q_chunk, lv_q_chunk = mu_queries[q_start:q_end], lv_queries[q_start:q_end]
            
            q_chunk_size = mu_q_chunk.size(0)
            topk_vals = torch.full((q_chunk_size, 0), -float("inf"), device=device)
            topk_idxs = torch.empty((q_chunk_size, 0), dtype=torch.long, device=device)

            for g_start in range(0, N, chunk_g):
                g_end = min(g_start + chunk_g, N)
                mu_g_chunk, lv_g_chunk = mu_gallery[g_start:g_end], lv_gallery[g_start:g_end]
                
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    sim_chunk = m.match_logits(mu_q_chunk, lv_q_chunk, mu_g_chunk, lv_g_chunk)

                current_topk_vals, current_topk_idxs_relative = sim_chunk.topk(k=min(Kmax, sim_chunk.size(1)), dim=1)
                merged_vals = torch.cat((topk_vals, current_topk_vals), dim=1)
                merged_idxs = torch.cat((topk_idxs, current_topk_idxs_relative + g_start), dim=1)
                topk_vals, new_order = merged_vals.topk(k=min(Kmax, merged_vals.size(1)), dim=1)
                topk_idxs = torch.gather(merged_idxs, dim=1, index=new_order)

            for i, q_abs_idx in enumerate(range(q_start, q_end)):
                if (topk_idxs[i] == q_abs_idx).any():
                    ranks.append((topk_idxs[i] == q_abs_idx).nonzero(as_tuple=False)[0, 0].item())
                else:
                    ranks.append(Kmax + 1)
        return torch.tensor(ranks, dtype=torch.long)

    ranks_i2t = recall_direction(mu_i_all, lv_i_all, mu_t_all, lv_t_all)
    ranks_t2i = recall_direction(mu_t_all, lv_t_all, mu_i_all, lv_i_all)

    def mk_metrics(ranks):
        return {f"R@{k}": (ranks < k).float().mean().item() * 100.0 for k in topk}

    mi, mt = mk_metrics(ranks_i2t), mk_metrics(ranks_t2i)
    out = {"R1_i2t": mi["R@1"], "R5_i2t": mi["R@5"], "R10_i2t": mi["R@10"], "R100_i2t": mi["R@100"],
           "R1_t2i": mt["R@1"], "R5_t2i": mt["R@5"], "R10_t2i": mt["R@10"], "R100_t2i": mt["R@100"]}
    out["RSUM"] = sum(out.values())
    return out

def train_one_epoch(model, criterion, loader, optimizer, scaler, device, epoch, grad_clip=1.0, log_interval=50, writer=None, scheduler=None):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch} Train")):
        img1 = batch["img1"].to(device, non_blocking=True)
        ids1 = batch["input_ids1"].to(device, non_blocking=True)
        att1 = batch["attention_mask1"].to(device, non_blocking=True)
        img2 = batch["img2"].to(device, non_blocking=True)
        ids2 = batch["input_ids2"].to(device, non_blocking=True)
        att2 = batch["attention_mask2"].to(device, non_blocking=True)
        
        with torch.amp.autocast("cuda", dtype=torch.float16):
            embeddings = model(img1, ids1, att1, img2, ids2, att2)
            losses = criterion(embeddings, unwrap(model))

        total_loss = losses["loss"].mean()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if writer and (step % log_interval == 0):
            m, g_step = unwrap(model), epoch * len(loader) + step
            writer.add_scalar("train/loss", total_loss.item(), g_step)
            # Log all individual loss components
            writer.add_scalar("train/loss_inter", losses["loss_inter"].mean().item(), g_step)
            writer.add_scalar("train/loss_intra_I", losses["loss_intra_I"].mean().item(), g_step)
            writer.add_scalar("train/loss_intra_T", losses["loss_intra_T"].mean().item(), g_step)
            writer.add_scalar("train/loss_vib", losses["loss_vib"].mean().item(), g_step)
            writer.add_scalar("train/kl_img", losses["kl_img"].mean().item(), g_step)
            writer.add_scalar("train/kl_txt", losses["kl_txt"].mean().item(), g_step)
            writer.add_scalar("train/model_a", m.a.item(), g_step)
            writer.add_scalar("train/model_b", m.b.item(), g_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], g_step)


def eval_img_transform(img_size=224):
    """
    Creates a standard, deterministic evaluation transform for images.
    """
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode
    return T.Compose([
        T.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25]),
    ])


def main():
    ap = argparse.ArgumentParser()
    # Add all the arguments from pcmepp_train.py
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--beta_vib_I", type=float, default=1e-4, help="Weight for image VIB KL divergence")
    ap.add_argument("--beta_vib_T", type=float, default=1e-4, help="Weight for text VIB KL divergence")
    ap.add_argument("--alpha_pp", type=float, default=0.0)
    ap.add_argument("--pp_k", type=int, default=8)
    ap.add_argument("--pp_thresh", type=float, default=None)
    ap.add_argument("--img_backbone", type=str, default="vit_base_patch16_224")
    ap.add_argument("--text_backbone", type=str, default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    ap.add_argument("--freeze_img", action="store_true")
    ap.add_argument("--freeze_text", action="store_true")
    ap.add_argument("--proj_hidden", type=int, default=0)
    ap.add_argument("--out", type=str, default="./checkpoints_medprobclip")
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--disable_lr_scheduler", action="store_false", dest="use_lr_scheduler")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trial_augment", type=str, default="")
    ap.add_argument("--alpha_warmup_epochs", type=int, default=1)
    ap.add_argument("--continue_from_epoch", type=int, default=0)
    # Add new lambda arguments
    ap.add_argument("--lambda_I", type=float, default=1.0, help="Weight for the intra-modal image loss.")
    ap.add_argument("--lambda_T", type=float, default=0.1, help="Weight for the intra-modal text loss.")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    if args.trial_augment: args.out = os.path.join(args.out, args.trial_augment)

    print("--- Hyperparameters ---", *[f"{k}: {v}" for k,v in vars(args).items()], "-----------------------", sep="\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out, "runs"))

    # Use mvs=True for multi-view sampling
    eval_transforms = eval_img_transform(args.img_size)
    train_ds = CXRStudyDataset(args.train_csv, tokenizer_name=args.text_backbone, max_len=args.max_len, mvs=True, img_size=args.img_size)
    val_ds = CXRStudyDataset(args.val_csv, tokenizer_name=args.text_backbone, max_len=args.max_len, mvs=True, img_size=args.img_size, img_transform1=eval_transforms, img_transform2=eval_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0), collate_fn=collate_mvs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_mvs)
    print(f"train_loader: {len(train_loader)} batches, val_loader: {len(val_loader)} batches.")

    test_loader = None
    if args.test_csv:
        test_ds = CXRStudyDataset(args.test_csv, tokenizer_name=args.text_backbone, max_len=args.max_len, mvs=True, img_size=args.img_size, img_transform1=eval_transforms, img_transform2=eval_transforms)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_mvs)
        print(f"test_loader: {len(test_loader)} batches.")

    cfg = MedProbClipConfig(embed_dim=512, img_backbone=args.img_backbone, text_backbone=args.text_backbone, proj_hidden=args.proj_hidden, freeze_img=args.freeze_img, freeze_text=args.freeze_text, max_txt_len=args.max_len)
    model = MedProbClip(cfg).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = MedProbClipLoss(
        beta_vib_I=args.beta_vib_I,
        beta_vib_T=args.beta_vib_T,
        alpha_pp=args.alpha_pp, 
        pp_k=args.pp_k, 
        pp_thresh=args.pp_thresh,
        lambda_I=args.lambda_I,
        lambda_T=args.lambda_T
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs) if args.use_lr_scheduler else None

    start_epoch, best_rsum = 0, -1.0
    if args.continue_from_epoch > 0:
        start_epoch = args.continue_from_epoch
        ckpt_path = os.path.join(args.out, "medprobclip_last.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            unwrap(model).load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scaler.load_state_dict(ckpt['scaler'])
            if scheduler and 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
            if 'best_rsum' in ckpt: best_rsum = ckpt['best_rsum']
            print(f"--> Resumed from epoch {ckpt.get('epoch', -1)}. Training will start from epoch {start_epoch}.")
        else:
            print(f"==> WARNING: Checkpoint not found at {ckpt_path}. Starting from scratch.")
            start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        if epoch < args.alpha_warmup_epochs:
            criterion.alpha_pp = args.alpha_pp * (epoch / args.alpha_warmup_epochs)
        else:
            criterion.alpha_pp = args.alpha_pp
        if writer: writer.add_scalar("train/alpha_pp_scheduled", criterion.alpha_pp, epoch)

        train_one_epoch(model, criterion, train_loader, optimizer, scaler, device, epoch, args.grad_clip, args.log_interval, writer, scheduler)
        
        if scheduler:
            scheduler.step()

        val_metrics = eval_retrieval_chunked(model, val_loader, device)
        print("VAL:", " ".join([f"{k}={v:.2f}" for k, v in val_metrics.items()]))
        for k, v in val_metrics.items(): writer.add_scalar(f"eval/val_{k}", v, epoch)

        if test_loader is not None:
            test_metrics = eval_retrieval_chunked(model, test_loader, device)
            print("TEST:", " ".join([f"{k}={v:.2f}" for k, v in test_metrics.items()]))
            for k, v in test_metrics.items(): writer.add_scalar(f"eval/test_{k}", v, epoch)

        if val_metrics["RSUM"] > best_rsum:
            best_rsum = val_metrics["RSUM"]
            ckpt = {"model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None, "cfg": vars(cfg), "metrics": val_metrics, "epoch": epoch, "best_rsum": best_rsum}
            torch.save(ckpt, os.path.join(args.out, "medprobclip_best.pt"))
            print("Saved best model.")
    
    last_ckpt = {"model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None, "cfg": vars(cfg), "epoch": epoch, "best_rsum": best_rsum}
    torch.save(last_ckpt, os.path.join(args.out, "medprobclip_last.pt"))

    if test_loader is not None:
        best_ckpt_path = os.path.join(args.out, "medprobclip_best.pt")
        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=device)
            unwrap(model).load_state_dict(ckpt['model'])
            print(f"Loaded best model from epoch {ckpt.get('epoch', 'N/A')} for final test.")
        
        test_metrics = eval_retrieval_chunked(model, test_loader, device)
        print("FINAL TEST:", " ".join([f"{k}={v:.2f}" for k, v in test_metrics.items()]))
        for k, v in test_metrics.items(): writer.add_scalar(f"final_test/{k}", v, 0)

if __name__ == "__main__":
    main()
