import csv
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, PreTrainedTokenizerBase

ImageFile.LOAD_TRUNCATED_IMAGES = True


def collate_single(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "study_id": [b["study_id"] for b in batch],
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], 0),
        "pixel_values_orig": torch.stack([b["pixel_values_orig"] for b in batch], 0),
        "input_ids": torch.stack([b["input_ids"] for b in batch], 0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], 0),
        "text": [b["text"] for b in batch],
    }      
        

def collate_mvs(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "study_id": [b["study_id"] for b in batch],
        "img1": torch.stack([b["img1"] for b in batch], 0),
        "img2": torch.stack([b["img2"] for b in batch], 0),
        "input_ids1": torch.stack([b["input_ids1"] for b in batch], 0),
        "attention_mask1": torch.stack([b["attention_mask1"] for b in batch], 0),
        "input_ids2": torch.stack([b["input_ids2"] for b in batch], 0),
        "attention_mask2": torch.stack([b["attention_mask2"] for b in batch], 0),
        "img1_orig": torch.stack([b["img1_orig"] for b in batch], 0),
        "img2_orig": torch.stack([b["img2_orig"] for b in batch], 0),
        "text1": [b["text1"] for b in batch],
        "text2": [b["text2"] for b in batch],
    }        
    
class CXRStudyDataset(Dataset):
    """
    Study-level dataset that supports:
      - PCME++ (mvs=False): 1 image + 1 text per study
      - CXR-CLIP (mvs=True): 2 images + 2 texts per study
      - MedProbCLIP (mvs=True): same I/O as CXR-CLIP; probabilistic loss is in the model

    CSV schema (one row per study):
      study_id,image_paths,view_names,findings,impression
        - image_paths: semicolon-separated absolute paths
        - view_names:  semicolon-separated tags aligned with image_paths (may be empty; we handle UNK)
        - findings, impression: plain text (may be empty)

    Key options:
      - mvs=True returns two images & two texts. If only one exists, the second is an AUGMENTED variant.
      - text_policy in {"both","random","findings_only","impression_only"}
      - image transforms: primary + secondary pipelines (secondary slightly stronger by default)
      - text_aug_for_second=True: create augmented text2 if only one section exists
      - allow_duplicate_single_view=True: if False, raise when <2 images and mvs=True
      - require_nonempty_text=False: if True, raise when both texts are empty after augmentation

    Example usage:
        ``` 
        # For PCME++: 1 image + 1 text per study
        from typing import List, Dict
        import torch
        def collate_single(batch: List[Dict]) -> Dict[str, torch.Tensor]:
            return {
                "study_id": [b["study_id"] for b in batch],
                # augmented tensor used by the model
                "pixel_values": torch.stack([b["pixel_values"] for b in batch], 0),
                # original (unnormalized) for logging/vis
                "pixel_values_orig": torch.stack([b["pixel_values_orig"] for b in batch], 0),
                # tokenized
                "input_ids": torch.stack([b["input_ids"] for b in batch], 0),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch], 0),
                # raw text (single string per sample)
                "text": [b["text"] for b in batch],
            }    
              
        pcme_train_dataset = CXRStudyDataset(
            csv_path="csv_out/mimic_train.csv",
            tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
            max_len=256,
            mvs=False,                      # single (image, text)
            text_policy="both",             # or "findings_only" / "impression_only"
            img_size=224,                   # backbone resolution (e.g., ViT-B/16 uses 224)
        )

        pcme_train_loader = DataLoader(
            pcme_train_dataset, batch_size=64, shuffle=True, num_workers=8,
            pin_memory=True, persistent_workers=True, collate_fn=collate_single,
        )
        ``` 

        ``` 
        # For CXR-CLIP: 2 images + 2 texts per study
        from typing import List, Dict
        import torch
        def collate_mvs(batch: List[Dict]) -> Dict[str, torch.Tensor]:
            return {
                "study_id": [b["study_id"] for b in batch],
                # augmented tensors used by the model
                "img1": torch.stack([b["img1"] for b in batch], 0),
                "img2": torch.stack([b["img2"] for b in batch], 0),
                "input_ids1": torch.stack([b["input_ids1"] for b in batch], 0),
                "attention_mask1": torch.stack([b["attention_mask1"] for b in batch], 0),
                "input_ids2": torch.stack([b["input_ids2"] for b in batch], 0),
                "attention_mask2": torch.stack([b["attention_mask2"] for b in batch], 0),
                # originals as tensors (same H×W, unnormalized, for logging/vis)
                "img1_orig": torch.stack([b["img1_orig"] for b in batch], 0),
                "img2_orig": torch.stack([b["img2_orig"] for b in batch], 0),
                # raw texts (lists of strings)
                "text1": [b["text1"] for b in batch],
                "text2": [b["text2"] for b in batch],
            }        

        cxrclip_train = CXRStudyDataset(
            csv_path="csv_out/mimic_train.csv",
            tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
            max_len=256,
            mvs=True,                       # two (image, text) per study
            text_policy="both",             # consider "random" for extra text aug
            img_size=224,
            text_aug_for_second=True,       # if only one section exists, augment to make the 2nd
            allow_duplicate_single_view=True,  # if only one image exists, create augmented 2nd view
        )

        cxrclip_train_loader = DataLoader(
            cxrclip_train, batch_size=64, shuffle=True, num_workers=8,
            pin_memory=True, persistent_workers=True, collate_fn=collate_mvs,
        )        
        ``` 

        ``` 
        # For MedProbCLIP: 2 images + 2 texts per study
        from typing import List, Dict
        import torch
        def collate_mvs(batch: List[Dict]) -> Dict[str, torch.Tensor]:
            return {
                "study_id": [b["study_id"] for b in batch],
                # augmented tensors used by the model
                "img1": torch.stack([b["img1"] for b in batch], 0),
                "img2": torch.stack([b["img2"] for b in batch], 0),
                "input_ids1": torch.stack([b["input_ids1"] for b in batch], 0),
                "attention_mask1": torch.stack([b["attention_mask1"] for b in batch], 0),
                "input_ids2": torch.stack([b["input_ids2"] for b in batch], 0),
                "attention_mask2": torch.stack([b["attention_mask2"] for b in batch], 0),
                # originals as tensors (same H×W, unnormalized, for logging/vis)
                "img1_orig": torch.stack([b["img1_orig"] for b in batch], 0),
                "img2_orig": torch.stack([b["img2_orig"] for b in batch], 0),
                # raw texts (lists of strings)
                "text1": [b["text1"] for b in batch],
                "text2": [b["text2"] for b in batch],
            }     

        medprob_train = CXRStudyDataset(
            csv_path="csv_out/mimic_train.csv",
            tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
            max_len=256,
            mvs=True,
            text_policy="both",
            img_size=224,
            text_aug_for_second=True,
            allow_duplicate_single_view=True,
        )

        medprob_train_loader = DataLoader(
            medprob_train, batch_size=64, shuffle=True, num_workers=8,
            pin_memory=True, persistent_workers=True, collate_fn=collate_mvs,
        )        
        ``` 
     
    """

    # ---------- helpers (stateless) ----------
    @staticmethod
    def _parse_semicolon_list(s: str) -> List[str]:
        s = (s or "").strip()
        return [p.strip() for p in s.split(";") if p.strip()]

    @staticmethod
    def _load_image_to_rgb(path: str) -> Image.Image:
        # X-rays are grayscale; replicate to 3 channels for ViT/CLIP/Swin families
        try:
            img = Image.open(path).convert("L")
            return Image.merge("RGB", (img, img, img))
        except (IOError, Image.UnidentifiedImageError) as e:
            print(f"WARNING: Could not load image {path}. Error: {e}. Returning black image.")
            # Return a black image as a placeholder
            return Image.new("RGB", (224, 224), color = 'black')

    @staticmethod
    def image_transform_primary(img_size: int = 224) -> T.Compose:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ])

    @staticmethod
    def image_transform_secondary(img_size: int = 224) -> T.Compose:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.80, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.RandomRotation(degrees=5)], p=0.5),           # small, safe rotation
            T.RandomApply([T.ColorJitter(brightness=0.08, contrast=0.10)], p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.15),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ])

    @staticmethod
    def image_transform_original(img_size: int = 224) -> T.Compose:
        # Deterministic, no augmentation, no normalization — good for logging/visualization & stacking in collate.
        return T.Compose([
            T.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),  # [0,1] range
        ])

    _SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

    @staticmethod
    def augment_text_semiconservative(
        text: str,
        shuffle_sentences: bool = True,
        word_dropout_p: float = 0.03,
        mask_numbers: bool = True,
    ) -> str:
        """
        Semantics-preserving text augmentation for radiology sections.
        - sentence order shuffle (safe)
        - very light word dropout (skips negations)
        - number masking (12 -> <NUM>)
        """
        t = (text or "").strip()
        if not t:
            return t

        # 1) sentence shuffle
        if shuffle_sentences:
            sents = [s.strip() for s in CXRStudyDataset._SENT_SPLIT.split(t) if s.strip()]
            if len(sents) >= 2:
                random.shuffle(sents)
                t = " ".join(sents)

        # 2) word dropout (avoid negations)
        if word_dropout_p > 0.0:
            NEG = {"no", "not", "without", "absent", "neither", "nor"}
            kept = []
            for w in t.split():
                wl = re.sub(r'[^\w]', '', w).lower()
                if wl in NEG or random.random() >= word_dropout_p:
                    kept.append(w)
            t = " ".join(kept)

        # 3) mask numbers
        if mask_numbers:
            t = re.sub(r'\b\d+(\.\d+)?\b', '<NUM>', t)

        # final whitespace normalize & remove newlines
        return " ".join(t.split())

    @staticmethod
    def _view_aware_pick_two(img_paths: List[str], views: List[str]) -> Tuple[str, str]:
        """Prefer distinct views if possible; else pick two random."""
        n = len(img_paths)
        if n == 0:
            raise ValueError("Study has no images.")
        if n == 1:
            return img_paths[0], img_paths[0]  # duplication handled upstream via augmentation

        groups: Dict[str, List[int]] = {}
        for i, v in enumerate(views or []):
            groups.setdefault((v or "UNK").upper(), []).append(i)
        if len(groups) >= 2:
            keys = list(groups.keys())
            i1 = random.choice(keys)
            i2 = random.choice([k for k in keys if k != i1]) if len(keys) > 1 else i1
            p1 = img_paths[random.choice(groups[i1])]
            p2 = img_paths[random.choice(groups[i2])]
            if p1 != p2:
                return p1, p2
        # fallback
        i1, i2 = random.sample(range(n), 2)
        return img_paths[i1], img_paths[i2]

    @staticmethod
    def _choose_two_texts(findings: str, impression: str, policy: str) -> Tuple[str, str]:
        """
        policy:
          - "both": (findings, impression) with fallback so neither is empty if one exists
          - "random": random order / fallback
          - "findings_only": duplicate findings (fallback to impression if missing)
          - "impression_only": duplicate impression (fallback to findings if missing)
        """
        f = (findings or "").strip()
        i = (impression or "").strip()
        if policy == "findings_only":
            t1 = t2 = f | i if isinstance(f, str) else f or i
        elif policy == "impression_only":
            t1 = t2 = i or f
        elif policy == "random":
            t1, t2 = (f, i) if random.random() < 0.5 else (i, f)
            if not t1 and t2: t1 = t2
            if not t2 and t1: t2 = t1
        else:  # "both"
            t1, t2 = f, i
            if not t1 and t2: t1 = t2
            if not t2 and t1: t2 = t1
        return t1, t2

    # ---------- dataset API ----------
    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_len: int = 256,
        mvs: bool = True,
        text_policy: str = "both",
        img_size: int = 224,
        img_transform1: Optional[Callable] = None,
        img_transform2: Optional[Callable] = None,
        img_transform_orig: Optional[Callable] = None,
        text_aug_for_second: bool = True,
        allow_duplicate_single_view: bool = True,  # if False, raise if <2 images and mvs=True
        require_nonempty_text: bool = False,       # if True, raise when both texts empty
    ):
        super().__init__()
        self.rows = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                imgs = self._parse_semicolon_list(r.get("image_paths", ""))
                views = self._parse_semicolon_list(r.get("view_names", ""))
                if len(views) != len(imgs):
                    views = ["UNK"] * len(imgs)
                self.rows.append({
                    "study_id": (r.get("study_id") or "").strip(),
                    "image_paths": imgs,
                    "view_names": views,
                    "findings": r.get("findings", "") or "",
                    "impression": r.get("impression", "") or "",
                })

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True
        )
        self.max_len = max_len
        self.mvs = mvs
        self.text_policy = text_policy
        self.tf1 = img_transform1 or self.image_transform_primary(img_size)
        self.tf2 = img_transform2 or self.image_transform_secondary(img_size)
        self.tf_orig = img_transform_orig or self.image_transform_original(img_size)
        self.text_aug_for_second = text_aug_for_second
        self.allow_duplicate_single_view = allow_duplicate_single_view
        self.require_nonempty_text = require_nonempty_text

    def __len__(self) -> int:
        return len(self.rows)

    def _tok(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # text here should already be whitespace-normalized; still guard:
        text = " ".join((text or "").split())
        toks = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        return toks["input_ids"].squeeze(0), toks["attention_mask"].squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.rows[idx]
        imgs, views = r["image_paths"], r["view_names"]

        if self.mvs:
            # -------------------- IMAGES (augmented + originals) --------------------
            if len(imgs) == 0:
                raise ValueError("Study has no images.")
            if len(imgs) >= 2:
                p1, p2 = self._view_aware_pick_two(imgs, views)
                im1 = self._load_image_to_rgb(p1)
                im2 = self._load_image_to_rgb(p2)
                img1_orig = self.tf_orig(im1)  # [0,1], HxW fixed
                img2_orig = self.tf_orig(im2)
                img1 = self.tf1(im1)          # augmented + normalized
                img2 = self.tf2(im2)
            else:
                if not self.allow_duplicate_single_view:
                    raise IndexError("Need >=2 images when mvs=True; set allow_duplicate_single_view=True to augment-duplicate.")
                im = self._load_image_to_rgb(imgs[0])
                img1_orig = self.tf_orig(im)
                img2_orig = self.tf_orig(im)
                img1 = self.tf1(im)           # aug view A
                img2 = self.tf2(im)           # aug view B

            # -------------------- TEXTS (raw + tokenized) --------------------
            t1, t2 = self._choose_two_texts(r["findings"], r["impression"], self.text_policy)
            # If only one non-empty text, create augmented second
            if self.text_aug_for_second and (bool(t1) ^ bool(t2)):
                if t1 and not t2:
                    t2 = self.augment_text_semiconservative(t1)
                elif t2 and not t1:
                    t1 = self.augment_text_semiconservative(t2)

            # Raw strings before tokenization (whitespace-normalized)
            text1 = " ".join((t1 or "").split())
            text2 = " ".join((t2 or "").split())

            if self.require_nonempty_text:
                if len(text1) == 0 and len(text2) == 0:
                    raise IndexError("Both texts are empty.")

            input_ids1, attention_mask1 = self._tok(text1)
            input_ids2, attention_mask2 = self._tok(text2)

            return {
                "study_id": r["study_id"],
                # augmented tensors used for training
                "img1": img1, "img2": img2,
                # deterministic originals for logging/visualization
                "img1_orig": img1_orig, "img2_orig": img2_orig,
                # raw texts (pre-tokenization)
                "text1": text1, "text2": text2,
                # tokenized
                "input_ids1": input_ids1, "attention_mask1": attention_mask1,
                "input_ids2": input_ids2, "attention_mask2": attention_mask2,
            }

        # -------------------- mvs=False (single outputs) --------------------
        if len(imgs) == 0:
            raise ValueError("Study has no images.")
        p = imgs[0] if len(imgs) == 1 else random.choice(imgs)
        im = self._load_image_to_rgb(p)
        pixel_values = self.tf1(im)            # augmented + normalized
        pixel_values_orig = self.tf_orig(im)   # deterministic [0,1]

        f = (r["findings"] or "").strip()
        i = (r["impression"] or "").strip()
        if self.text_policy == "findings_only":
            text = f or i
        elif self.text_policy == "impression_only":
            text = i or f
        else:
            text = f if random.random() < 0.5 else i
            if not text:
                text = i if text is f else f
        text = " ".join((text or "").split())

        input_ids, attention_mask = self._tok(text)

        return {
            "study_id": r["study_id"],
            # augmented tensor + original
            "pixel_values": pixel_values,
            "pixel_values_orig": pixel_values_orig,
            # raw text (pre-tokenization)
            "text": text,
            # tokenized
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }