import argparse, os, sys
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
from PIL import ImageFile

from medprobclip_model import MedProbClip, MedProbClipConfig
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
    m = unwrap(model)
    m.eval()

    all_mu_i, all_lv_i, all_mu_t, all_lv_t = [], [], [], []
    for batch in tqdm(loader, desc="Encoding for Eval"):
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

    # --- Compute full similarity matrices ---
    sim_i2t = torch.zeros(N, N, device='cpu')
    for q_start in range(0, N, chunk_q):
        q_end = min(q_start + chunk_q, N)
        mu_q_chunk, lv_q_chunk = mu_i_all[q_start:q_end], lv_i_all[q_start:q_end]
        for g_start in range(0, N, chunk_g):
            g_end = min(g_start + chunk_g, N)
            mu_g_chunk, lv_g_chunk = mu_t_all[g_start:g_end], lv_t_all[g_start:g_end]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                sim_chunk = m.match_logits(mu_q_chunk, lv_q_chunk, mu_g_chunk, lv_g_chunk)
            sim_i2t[q_start:q_end, g_start:g_end] = sim_chunk.cpu()
    
    sim_t2i = sim_i2t.T

    # --- Calculate recall from similarity matrices ---
    gts = torch.arange(N)
    
    # i2t
    sorted_i2t = torch.argsort(sim_i2t, dim=1, descending=True)
    ranks_i2t = (sorted_i2t == gts.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]

    # t2i
    sorted_t2i = torch.argsort(sim_t2i, dim=1, descending=True)
    ranks_t2i = (sorted_t2i == gts.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
    
    def mk_metrics(ranks):
        return {f"R@{k}": (ranks < k).float().mean().item() * 100.0 for k in topk}

    mi, mt = mk_metrics(ranks_i2t), mk_metrics(ranks_t2i)
    out = {"R1_i2t": mi["R@1"], "R5_i2t": mi["R@5"], "R10_i2t": mi["R@10"], "R100_i2t": mi["R@100"],
           "R1_t2i": mt["R@1"], "R5_t2i": mt["R@5"], "R10_t2i": mt["R@10"], "R100_t2i": mt["R@100"]}
    out["RSUM"] = sum(out.values())
    # Return all embeddings along with metrics and similarities
    return out, sim_i2t, sim_t2i, mu_i_all, lv_i_all, mu_t_all, lv_t_all


def eval_img_transform(img_size=224):
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
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--checkpoint_path", required=True, help="Path to the model checkpoint to evaluate.")
    ap.add_argument("--output_dir", type=str, default=".", help="Directory to save the predictions.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--img_backbone", type=str, default="vit_base_patch16_224")
    ap.add_argument("--text_backbone", type=str, default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    ap.add_argument("--proj_hidden", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)

    print("--- Test Configuration ---")
    for k,v in vars(args).items():
        print(f"{k}: {v}")
    print("--------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_transforms = eval_img_transform(args.img_size)
    test_ds = CXRStudyDataset(args.test_csv, tokenizer_name=args.text_backbone, max_len=args.max_len, mvs=True, img_size=args.img_size, img_transform1=eval_transforms, img_transform2=eval_transforms)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_mvs)
    print(f"Test loader initialized with {len(test_loader)} batches.")

    cfg = MedProbClipConfig(embed_dim=512, img_backbone=args.img_backbone, text_backbone=args.text_backbone, proj_hidden=args.proj_hidden, max_txt_len=args.max_len)
    model = MedProbClip(cfg).to(device)

    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Checkpoint not found at {args.checkpoint_path}")
        return
        
    print(f"Loading best model from {args.checkpoint_path} for final test evaluation...")
    ckpt = torch.load(args.checkpoint_path, map_location=device)

    if 'cfg' in ckpt:
        print("\n--- Model Cfg from Checkpoint ---")
        for k, v in ckpt['cfg'].items():
            print(f"{k}: {v}")
        print("---------------------------------\n")

    unwrap(model).load_state_dict(ckpt['model'])
    print(f"Best model was from epoch {ckpt.get('epoch', 'N/A')} with R-SUM {ckpt.get('metrics', {}).get('RSUM', 'N/A')}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel evaluation.")
        model = nn.DataParallel(model)

    print("Evaluating on test set...")
    test_metrics, sim_i2t, sim_t2i, mu_i_all, lv_i_all, mu_t_all, lv_t_all = eval_retrieval_chunked(model, test_loader, device)

    print("\n----------- FINAL TEST RESULTS -----------")
    print(" ".join([f"{k}={v:.2f}" for k, v in test_metrics.items()]))
    print("----------------------------------------\n")

    # --- Save predictions and ground truth ---
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "medprobclip_predictions.pt")
    
    # Ground truth is simply the diagonal of the similarity matrix
    ground_truth = torch.arange(sim_i2t.shape[0])

    torch.save({
        'sim_i2t': sim_i2t,
        'sim_t2i': sim_t2i,
        'ground_truth': ground_truth,
        'metrics': test_metrics,
        # Add the detailed embeddings for uncertainty analysis
        'mu_i_all': mu_i_all.cpu(),
        'lv_i_all': lv_i_all.cpu(),
        'mu_t_all': mu_t_all.cpu(),
        'lv_t_all': lv_t_all.cpu()
    }, output_path)

    print(f"Predictions and metrics saved to {output_path}")


if __name__ == "__main__":
    main()
