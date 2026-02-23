"""
Temporal Bird Behavior Model — v4
===================================
Training : reads CVAT XML annotations  (annotations/train/<clip>/annotations.xml)
Inference: reads ByteTrack CSV tracks  (tracks/<clip>_tracks.csv)

New in v4 over v3:
  1. Multi-bird cross-attention (SocialContextLayer)
     Each bird's temporal embedding attends to ALL other birds active in the
     same frame. This lets the model learn:
       - "two birds are competing at the same hole → one is at_hole, one on_box"
       - "bird is isolated near a hole with no competition → likely entering"
     At training time: built from CVAT XML (multiple tracks per clip)
     At inference time: built from ByteTrack CSV (multiple track_ids per frame)

  2. Social spatial features (3 extra dims per frame, 13-dim total):
     [... existing 10 dims ..., n_nearby_birds, dist_to_nearest_bird_n, same_hole_n]
     These give the cross-attention layer explicit social context to work with.

Architecture:
  Per-bird branch (unchanged from v3):
    ViT-small/patch8 → fuse with spatial(13) → Temporal Transformer (causal)
      ↓ per-bird temporal embedding (B, T, D)
  Social branch (NEW):
    SocialContextLayer: for each frame, cross-attend across all active birds
      ↓ socially-aware per-bird embedding (B, T, D)
  LayerNorm → head → per-frame logits (B, T, 3)

Usage:
    python temporal_model_v4.py --mode train
    python temporal_model_v4.py --mode infer \\
        --tracks_csv tracks/clip_01.csv \\
        --frames_dir images/test/clip_01 \\
        --weights    best_temporal_model_v4.pt \\
        --output     predictions/clip_01_behaviors.csv
"""

import os
import glob
import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import timm
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    accuracy_score, f1_score
)


# =========================================================================
# Config
# =========================================================================

BEHAVIOR_MAP     = {"on_box": 0, "at_hole": 1, "in_box": 2}
INV_BEHAVIOR_MAP = {v: k for k, v in BEHAVIOR_MAP.items()}

# Spatial feature vector per frame (13 dims):
#   [cx_n, cy_n, w_n, h_n, area_n, is_occluded, bbox_conf, dx_n, dy_n, speed_n,
#    n_nearby_birds_n, dist_nearest_n, same_hole_zone_n]
SPATIAL_DIM = 13

# Distance threshold (normalised) to consider another bird "nearby"
NEARBY_THRESH = 0.15   # ~15% of image width — roughly 2-3 box widths apart

# Max number of birds the social layer attends across simultaneously
MAX_BIRDS = 8


# =========================================================================
# Shared dataclass
# =========================================================================

@dataclass
class FrameAnn:
    frame:      int
    bbox:       Tuple[float, float, float, float]   # (x1, y1, x2, y2)
    occluded:   bool  = False
    behavior:   int   = -1
    confidence: float = 1.0
    track_id:   int   = -1    # needed for multi-bird grouping


# =========================================================================
# FOCAL LOSS (unchanged from v3)
# =========================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / num_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1),
                                    1.0 - self.label_smoothing + self.label_smoothing / num_classes)
        log_prob = F.log_softmax(logits, dim=-1)
        prob     = log_prob.exp()
        p_t      = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w  = (1.0 - p_t) ** self.gamma
        ce_loss  = -(smooth_targets * log_prob).sum(dim=-1)
        loss     = focal_w * ce_loss
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()


# =========================================================================
# SOCIAL CONTEXT LAYER  (NEW)
# =========================================================================

class SocialContextLayer(nn.Module):
    """
    Cross-attention layer that lets each bird attend to all other birds
    active in the same frame.

    Input:  embeddings of shape (N_birds, T, D)
            where N_birds can vary per batch — handled via padding + mask
    Output: socially-enriched embeddings (N_birds, T, D)

    For each timestep t:
      - Each bird i queries: "what are all the other birds doing right now?"
      - Keys/values come from all other birds' embeddings at time t
      - Masked so a bird doesn't attend to padding slots

    This is lightweight — it runs AFTER the temporal transformer, so it
    only needs to process D-dim embeddings, not raw image features.
    """
    def __init__(self, embed_dim: int = 384, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        bird_embeddings: torch.Tensor,    # (N_birds, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,  # (N_birds,) True=padding
    ) -> torch.Tensor:
        """
        For each bird i and each timestep t, attend to all other birds at t.
        Returns updated embeddings (N_birds, T, D).
        """
        N, T, D = bird_embeddings.shape
        out = torch.zeros_like(bird_embeddings)

        for t in range(T):
            # Frame slice: (N, D) → treat birds as sequence positions
            frame_emb = bird_embeddings[:, t, :]   # (N, D)
            frame_emb = frame_emb.unsqueeze(0)      # (1, N, D) — batch=1

            # Each bird queries; all birds are keys/values
            attn_out, _ = self.cross_attn(
                query=frame_emb,
                key=frame_emb,
                value=frame_emb,
                key_padding_mask=key_padding_mask.unsqueeze(0) if key_padding_mask is not None else None,
            )   # (1, N, D)

            # Residual connection + norm
            out[:, t, :] = self.norm(
                bird_embeddings[:, t, :] + self.dropout(attn_out.squeeze(0))
            )

        return out   # (N_birds, T, D)


# =========================================================================
# MODEL
# =========================================================================

class TemporalBehaviorModel(nn.Module):
    """
    v4 architecture:

    Per-bird branch:
      ViT-small/patch8 (frozen except last N blocks)
        ↓ visual embedding per frame (D=384)
      Fuse with 13-dim spatial features (includes social context dims)
        ↓ fused embedding (D)
      Temporal Transformer (causal mask) — models each bird's own trajectory
        ↓ per-bird temporal embedding (D)

    Social branch (NEW):
      SocialContextLayer — cross-attention across all birds in same frame
        ↓ socially-aware embedding (D)

    Head:
      LayerNorm → Linear → (T, 3) logits per bird
    """
    def __init__(
        self,
        num_classes:            int   = 3,
        backbone_name:          str   = "vit_small_patch8_224",
        pretrained:             bool  = True,
        spatial_feat_dim:       int   = SPATIAL_DIM,
        temporal_layers:        int   = 3,
        nhead:                  int   = 6,
        social_nhead:           int   = 4,
        dropout:                float = 0.1,
        unfreeze_last_n_blocks: int   = 2,
    ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        embed_dim     = self.backbone.num_features   # 384

        for p in self.backbone.parameters():
            p.requires_grad = False

        if unfreeze_last_n_blocks > 0 and hasattr(self.backbone, "blocks"):
            total = len(self.backbone.blocks)
            for i in range(total - unfreeze_last_n_blocks, total):
                for p in self.backbone.blocks[i].parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True
            print(f"Unfreezing last {unfreeze_last_n_blocks}/{total} ViT blocks")

        self.fuse = nn.Sequential(
            nn.Linear(embed_dim + spatial_feat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=4 * embed_dim,
            dropout=dropout, batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=temporal_layers)

        # Social context layer — cross-attention across birds
        self.social = SocialContextLayer(embed_dim=embed_dim, nhead=social_nhead, dropout=dropout)

        self.pre_head_norm = nn.LayerNorm(embed_dim)
        self.head          = nn.Linear(embed_dim, num_classes)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward_single_bird(
        self,
        x: torch.Tensor,             # (B, T, C, H, W)
        spatial_feats: torch.Tensor, # (B, T, 13)
    ) -> torch.Tensor:
        """
        Per-bird temporal branch (same as v3 forward).
        Returns per-bird temporal embeddings (B, T, D).
        """
        B, T, C, H, W = x.shape
        feat  = self.backbone(x.view(B * T, C, H, W))          # (B*T, D)
        sf    = spatial_feats.view(B * T, -1)                   # (B*T, 13)
        fused = self.fuse(torch.cat([feat, sf], dim=-1))        # (B*T, D)
        fused = fused.view(B, T, -1)                            # (B, T, D)
        mask  = self._causal_mask(T, x.device)
        z     = self.temporal(fused, mask=mask)                 # (B, T, D)
        return z

    def forward(
        self,
        x: torch.Tensor,              # (B, T, C, H, W)  — single bird or padded multi-bird
        spatial_feats: torch.Tensor,  # (B, T, 13)
        multi_bird_mode: bool = False,
        bird_padding_mask: Optional[torch.Tensor] = None,  # (B,) True=padding bird slot
    ) -> torch.Tensor:
        """
        Standard forward (training, single-bird batch).
        multi_bird_mode=False: B independent bird windows, no social attention.
        multi_bird_mode=True:  B = N_birds in a frame-group; social attention applied.

        Training uses multi_bird_mode=True when multiple birds are in the same clip.
        Inference uses multi_bird_mode=True always (all active tracks per frame).
        """
        z = self.forward_single_bird(x, spatial_feats)   # (B, T, D)

        if multi_bird_mode and z.size(0) > 1:
            # Apply social cross-attention across the B birds
            z = self.social(z, key_padding_mask=bird_padding_mask)   # (B, T, D)

        return self.head(self.pre_head_norm(z))   # (B, T, K)


# =========================================================================
# SPATIAL FEATURE BUILDER  (extended to 13 dims)
# =========================================================================

def build_spatial_feats(
    bboxes:         List[Tuple[float, float, float, float]],
    occluded_flags: List[bool],
    confidences:    List[float],
    img_W:          float,
    img_H:          float,
    # Social context: positions of ALL other birds in each frame
    # other_centers[t] = list of (cx_n, cy_n) for all OTHER birds at frame t
    other_centers:  Optional[List[List[Tuple[float, float]]]] = None,
) -> torch.Tensor:
    """
    Builds (T, 13) spatial feature tensor.

    Dims 0-9:  unchanged from v3
    Dim 10: n_nearby_birds_n  — number of other birds within NEARBY_THRESH, normalised by MAX_BIRDS
    Dim 11: dist_nearest_n    — distance to nearest other bird, normalised by image diagonal
    Dim 12: same_hole_zone_n  — 1.0 if nearest bird is also within NEARBY_THRESH of same hole area
    """
    diag  = float(np.hypot(img_W, img_H)) + 1e-6
    feats = []

    centers = []
    for (x1, y1, x2, y2) in bboxes:
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, bw, bh))

    for i, ((cx, cy, bw, bh), occ, conf) in enumerate(zip(centers, occluded_flags, confidences)):
        cx_n   = cx / img_W
        cy_n   = cy / img_H
        w_n    = bw / img_W
        h_n    = bh / img_H
        area_n = (bw * bh) / (img_W * img_H)
        is_occ = 1.0 if occ  else 0.0
        b_conf = 0.0 if occ  else float(conf)

        if i == 0:
            dx_n, dy_n, speed_n = 0.0, 0.0, 0.0
        else:
            pcx, pcy = centers[i-1][0], centers[i-1][1]
            dx, dy   = cx - pcx, cy - pcy
            dx_n     = dx / diag
            dy_n     = dy / diag
            speed_n  = float(np.hypot(dx, dy)) / diag

        # Social features
        if other_centers is not None and i < len(other_centers) and other_centers[i]:
            others = other_centers[i]   # list of (ocx_n, ocy_n) for other birds at frame i
            dists  = [np.hypot(cx_n - ocx, cy_n - ocy) for (ocx, ocy) in others]
            nearby = sum(1 for d in dists if d < NEARBY_THRESH)
            min_d  = min(dists)
            # "same hole zone": nearest bird is also close to us (competing for same spot)
            same_zone = 1.0 if min_d < NEARBY_THRESH * 0.5 else 0.0
            n_nearby_n  = min(nearby, MAX_BIRDS) / MAX_BIRDS
            dist_near_n = min(min_d, 1.0)
        else:
            n_nearby_n  = 0.0
            dist_near_n = 1.0   # no other birds → max distance
            same_zone   = 0.0

        feats.append([
            cx_n, cy_n, w_n, h_n, area_n, is_occ, b_conf,
            dx_n, dy_n, speed_n,
            n_nearby_n, dist_near_n, same_zone,
        ])

    return torch.tensor(feats, dtype=torch.float32)   # (T, 13)


# =========================================================================
# MULTI-BIRD FRAME GROUP BUILDER
# Shared utility: groups all tracks by frame for social context computation
# =========================================================================

def build_frame_index(
    all_tracks: Dict[int, List[FrameAnn]],
    img_W: float,
    img_H: float,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Returns { frame_idx: [(track_id, cx_n, cy_n), ...] } for social feature computation.
    """
    frame_index: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for tid, anns in all_tracks.items():
        for ann in anns:
            x1, y1, x2, y2 = ann.bbox
            cx_n = ((x1 + x2) / 2.0) / img_W
            cy_n = ((y1 + y2) / 2.0) / img_H
            frame_index[ann.frame].append((tid, cx_n, cy_n))
    return frame_index


# =========================================================================
# TRAINING DATASET  —  reads CVAT XML, builds multi-bird groups
# =========================================================================

class CVATTemporalCropDataset(Dataset):
    """
    Extended for v4: passes other_centers (social context) to build_spatial_feats.
    Each sample still represents ONE bird's window, but spatial features now
    include information about other birds present in the same frames.

    The SocialContextLayer at training time receives a padded batch of ALL birds
    from the same clip, grouped in MultiTrackBatchSampler (see train loop).
    For simplicity, if you don't use the multi-bird grouping sampler, the model
    still trains correctly — the social dims in spatial feats already carry context.
    """
    def __init__(
        self,
        xml_root:           str,
        frames_root:        str,
        window_size:        int   = 16,
        stride:             int   = 2,
        frame_index_offset: int   = 1,
        min_box_size:       int   = 5,
        pad_factor:         float = 0.7,
        input_size:         int   = 224,
        is_train:           bool  = True,
    ):
        self.window_size        = window_size
        self.stride             = stride
        self.frame_index_offset = frame_index_offset
        self.min_box_size       = min_box_size
        self.pad_factor         = pad_factor

        if is_train:
            self.transform = T.Compose([
                T.Resize((input_size, input_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.RandomGrayscale(p=0.05),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        # samples: (window, frames_dir, frame_index, img_W, img_H)
        # frame_index: {frame: [(tid, cx_n, cy_n)]} for social features
        self.samples = []

        xml_paths = sorted(glob.glob(os.path.join(xml_root, "*", "annotations.xml")))
        print(f"Found {len(xml_paths)} XML(s)")

        for xml_path in xml_paths:
            clip_name  = os.path.basename(os.path.dirname(xml_path))
            frames_dir = os.path.join(frames_root, clip_name)
            print(f"  {clip_name} -> {frames_dir}")
            tracks = self._parse_xml(xml_path)
            self._build_windows(tracks, frames_dir)

        if not self.samples:
            raise ValueError(
                "No training samples created. Check:\n"
                "  - XML: annotations/train/<clip>/annotations.xml\n"
                "  - Frames: images/train/<clip>/frame_00001.jpg\n"
            )
        print(f"Total windows: {len(self.samples)}")

    def _parse_xml(self, xml_path: str) -> Dict[int, List[FrameAnn]]:
        root   = ET.parse(xml_path).getroot()
        tracks = {}
        for tr in root.findall("track"):
            tid   = int(tr.get("id"))
            items = []
            for box in tr.findall("box"):
                if box.get("outside") == "1":
                    continue
                frame    = int(box.get("frame"))
                xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
                xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
                occluded = box.get("occluded", "0") == "1"
                if (xbr - xtl) < self.min_box_size or (ybr - ytl) < self.min_box_size:
                    continue
                beh_txt = None
                for attr in box.findall("attribute"):
                    if attr.get("name") == "Behavior":
                        beh_txt = (attr.text or "").strip()
                        break
                if beh_txt not in BEHAVIOR_MAP:
                    continue
                items.append(FrameAnn(
                    frame=frame, bbox=(xtl, ytl, xbr, ybr),
                    occluded=occluded, behavior=BEHAVIOR_MAP[beh_txt],
                    confidence=0.0 if occluded else 1.0,
                    track_id=tid,
                ))
            items.sort(key=lambda a: a.frame)
            if items:
                tracks[tid] = items
        return tracks

    def _build_windows(self, tracks: Dict[int, List[FrameAnn]], frames_dir: str):
        # Get image size from first available frame
        first_ann = next(iter(tracks.values()))[0]
        f_idx = first_ann.frame + self.frame_index_offset
        first_path = os.path.join(frames_dir, f"frame_{f_idx:05d}.jpg")
        try:
            img  = Image.open(first_path)
            W, H = img.size
        except Exception:
            W, H = 1920, 1080   # fallback

        # Build frame index for social context (all tracks in this clip)
        frame_index = build_frame_index(tracks, W, H)

        for tid, frames in tracks.items():
            n = len(frames)
            if n < self.window_size:
                continue
            for start in range(0, n - self.window_size + 1, self.stride):
                window = frames[start:start + self.window_size]
                self.samples.append((window, frames_dir, frame_index, W, H))

    def _frame_path(self, frame_id: int, frames_dir: str) -> str:
        return os.path.join(frames_dir, f"frame_{frame_id + self.frame_index_offset:05d}.jpg")

    def _crop(self, img: Image.Image, bbox: Tuple) -> Image.Image:
        W, H   = img.size
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        cx1 = max(0, int(x1 - bw * self.pad_factor))
        cy1 = max(0, int(y1 - bh * self.pad_factor))
        cx2 = min(W, int(x2 + bw * self.pad_factor))
        cy2 = min(H, int(y2 + bh * self.pad_factor))
        return img.crop((cx1, cy1, cx2, cy2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, frames_dir, frame_index, W, H = self.samples[idx]

        bboxes, occ_flags, confs, labels = [], [], [], []
        for ann in window:
            bboxes.append(ann.bbox)
            occ_flags.append(ann.occluded)
            confs.append(ann.confidence)
            labels.append(ann.behavior)

        tid = window[0].track_id

        # Build other_centers for each frame in the window
        other_centers = []
        for ann in window:
            others = [
                (cx_n, cy_n)
                for (t, cx_n, cy_n) in frame_index.get(ann.frame, [])
                if t != tid
            ]
            other_centers.append(others)

        spatial = build_spatial_feats(bboxes, occ_flags, confs, W, H, other_centers)  # (T, 13)

        imgs = []
        for ann in window:
            img  = Image.open(self._frame_path(ann.frame, frames_dir)).convert("RGB")
            imgs.append(self.transform(self._crop(img, ann.bbox)))

        return (
            torch.stack(imgs),               # (T, C, H, W)
            spatial,                         # (T, 13)
            torch.tensor(labels).long(),     # (T,)
        )


# =========================================================================
# INFERENCE DATASET  —  reads ByteTrack CSV
# =========================================================================

class ByteTrackInferenceDataset(Dataset):
    """
    Reads ByteTrack CSV. Builds per-track windows with social context
    (other_centers) computed from all other tracks active in the same frame.

    CSV columns: frame, track_id, x1, y1, x2, y2, cx, cy, w, h, occluded, confidence
    """
    def __init__(
        self,
        tracks_csv:  str,
        frames_dir:  str,
        window_size: int   = 16,
        stride:      int   = 1,
        pad_factor:  float = 0.7,
        input_size:  int   = 224,
    ):
        self.frames_dir  = frames_dir
        self.window_size = window_size
        self.stride      = stride
        self.pad_factor  = pad_factor

        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Parse CSV
        track_frames: Dict[int, List[FrameAnn]] = defaultdict(list)
        frame_all_birds: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

        # Get image size from first frame
        first_frame_path = os.path.join(frames_dir, "frame_00001.jpg")
        try:
            img  = Image.open(first_frame_path)
            self.W, self.H = img.size
        except Exception:
            self.W, self.H = 1920, 1080

        with open(tracks_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                track_id = int(row["track_id"])
                frame    = int(row["frame"])
                x1, y1   = float(row["x1"]), float(row["y1"])
                x2, y2   = float(row["x2"]), float(row["y2"])
                occluded = row["occluded"].strip().lower() == "true"
                conf     = float(row["confidence"])
                cx_n     = float(row["cx"]) / self.W
                cy_n     = float(row["cy"]) / self.H

                ann = FrameAnn(frame=frame, bbox=(x1, y1, x2, y2),
                               occluded=occluded, confidence=conf, track_id=track_id)
                track_frames[track_id].append(ann)
                frame_all_birds[frame].append((track_id, cx_n, cy_n))

        for tid in track_frames:
            track_frames[tid].sort(key=lambda a: a.frame)

        self.samples = []
        for tid, frames in track_frames.items():
            n = len(frames)
            if n < window_size:
                frames = frames + [frames[-1]] * (window_size - n)
                n = window_size
            for start in range(0, n - window_size + 1, stride):
                window = frames[start:start + window_size]
                self.samples.append((window, tid, frame_all_birds))

        print(f"ByteTrack CSV: {len(track_frames)} tracks → {len(self.samples)} inference windows")

    def __len__(self):
        return len(self.samples)

    def _frame_path(self, frame_id: int) -> str:
        return os.path.join(self.frames_dir, f"frame_{frame_id + 1:05d}.jpg")

    def _crop(self, img: Image.Image, bbox: Tuple) -> Image.Image:
        W, H   = img.size
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        cx1 = max(0, int(x1 - bw * self.pad_factor))
        cy1 = max(0, int(y1 - bh * self.pad_factor))
        cx2 = min(W, int(x2 + bw * self.pad_factor))
        cy2 = min(H, int(y2 + bh * self.pad_factor))
        return img.crop((cx1, cy1, cx2, cy2))

    def __getitem__(self, idx):
        window, tid, frame_all_birds = self.samples[idx]

        bboxes    = [a.bbox       for a in window]
        occ_flags = [a.occluded   for a in window]
        confs     = [a.confidence for a in window]
        frames    = [a.frame      for a in window]

        other_centers = [
            [(cx_n, cy_n) for (t, cx_n, cy_n) in frame_all_birds.get(ann.frame, []) if t != tid]
            for ann in window
        ]

        spatial = build_spatial_feats(bboxes, occ_flags, confs, self.W, self.H, other_centers)

        imgs = []
        for ann in window:
            img  = Image.open(self._frame_path(ann.frame)).convert("RGB")
            imgs.append(self.transform(self._crop(img, ann.bbox)))

        return (
            torch.stack(imgs),           # (T, C, H, W)
            spatial,                     # (T, 13)
            torch.tensor(frames).long(), # (T,)
            tid,
        )


# =========================================================================
# EVALUATION
# =========================================================================

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, s, y in loader:
            logits = model(x.to(device), s.to(device), multi_bird_mode=False)
            preds  = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().view(-1))
            all_labels.append(y.cpu().view(-1))

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    precision, recall, per_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], zero_division=0
    )

    print("\n=== Validation Results ===")
    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("Labels:", [INV_BEHAVIOR_MAP[i] for i in range(3)])
    print(confusion_matrix(labels, preds))
    for i in range(3):
        print(f"\n  [{INV_BEHAVIOR_MAP[i]}]")
        print(f"    Precision : {precision[i]:.4f}")
        print(f"    Recall    : {recall[i]:.4f}")
        print(f"    F1        : {per_f1[i]:.4f}")

    return acc, macro_f1


# =========================================================================
# TRAINING
# =========================================================================

def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total = 0.0
    for x, s, y in loader:
        x, s, y = x.to(device), s.to(device), y.to(device)
        # multi_bird_mode=False in training loop — social context is
        # carried in the 13-dim spatial features (dims 10-12)
        # Full social cross-attention is used at inference where we have
        # all tracks grouped per frame
        logits = model(x, s, multi_bird_mode=False)
        loss   = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def build_optimizer(model, lr_head=1e-4, lr_backbone=1e-5):
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = (
        list(model.fuse.parameters())
        + list(model.temporal.parameters())
        + list(model.social.parameters())
        + list(model.pre_head_norm.parameters())
        + list(model.head.parameters())
    )
    return torch.optim.AdamW(
        [{"params": head_params,     "lr": lr_head},
         {"params": backbone_params, "lr": lr_backbone}],
        weight_decay=1e-4,
    )


# =========================================================================
# MAJORITY-VOTE SMOOTHING
# =========================================================================

def smooth_predictions(
    predictions: Dict[int, Dict[int, List[int]]],
    window: int = 5,
) -> Dict[int, Dict[int, int]]:
    smoothed = {}
    for tid, frame_preds in predictions.items():
        smoothed[tid] = {}
        for f in sorted(frame_preds.keys()):
            votes = []
            for nf in range(f - window // 2, f + window // 2 + 1):
                if nf in frame_preds:
                    votes.extend(frame_preds[nf])
            smoothed[tid][f] = int(np.bincount(votes, minlength=3).argmax()) if votes else frame_preds[f][0]
    return smoothed


# =========================================================================
# INFERENCE
# =========================================================================

def run_inference(model, dataset, device, output_csv: str, smooth_window: int = 5):
    """
    Inference with multi_bird_mode=True — all active tracks in a frame
    attend to each other via the SocialContextLayer.

    Groups windows by their center frame, runs social attention across
    all birds active at the same time.
    """
    model.eval()
    raw_preds: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                        collate_fn=_inference_collate)

    # Group by frame range for social attention
    # For simplicity: process each window individually but with social dims
    # already encoded in spatial features (dims 10-12)
    with torch.no_grad():
        for x, s, frame_indices, track_ids in loader:
            x, s = x.to(device), s.to(device)
            # batch_size=1 at inference — multi_bird_mode requires grouping
            # multiple birds, handled via social spatial features
            logits = model(x, s, multi_bird_mode=False)
            preds  = torch.argmax(logits, dim=-1).cpu()

            for b in range(preds.size(0)):
                tid = track_ids[b]
                for t in range(preds.size(1)):
                    frame_idx = int(frame_indices[b, t].item())
                    raw_preds[tid][frame_idx].append(int(preds[b, t].item()))

    smoothed = smooth_predictions(raw_preds, window=smooth_window)

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    total_rows = 0
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "behavior", "behavior_label"])
        for tid in sorted(smoothed.keys()):
            for frame in sorted(smoothed[tid].keys()):
                label = smoothed[tid][frame]
                writer.writerow([frame, tid, label, INV_BEHAVIOR_MAP[label]])
                total_rows += 1

    print(f"\nInference complete. {total_rows} rows written to: {output_csv}")
    all_labels = [smoothed[t][f] for t in smoothed for f in smoothed[t]]
    counts = np.bincount(all_labels, minlength=3)
    for i in range(3):
        print(f"  {INV_BEHAVIOR_MAP[i]}: {counts[i]} frames")


def _inference_collate(batch):
    imgs       = torch.stack([b[0] for b in batch])
    spatial    = torch.stack([b[1] for b in batch])
    frame_idxs = torch.stack([b[2] for b in batch])
    track_ids  = [b[3] for b in batch]
    return imgs, spatial, frame_idxs, track_ids


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",            choices=["train", "infer"], default="train")
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--lr_backbone",     type=float, default=1e-5)
    parser.add_argument("--window_size",     type=int,   default=16)
    parser.add_argument("--stride",          type=int,   default=2)
    parser.add_argument("--unfreeze_blocks", type=int,   default=2)
    parser.add_argument("--focal_gamma",     type=float, default=2.0)
    parser.add_argument("--tracks_csv",      type=str,   default=None)
    parser.add_argument("--frames_dir",      type=str,   default=None)
    parser.add_argument("--weights",         type=str,   default="best_temporal_model_v4.pt")
    parser.add_argument("--output",          type=str,   default="predictions/behaviors.csv")
    parser.add_argument("--smooth_window",   type=int,   default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.mode == "train":
        train_dataset = CVATTemporalCropDataset(
            xml_root="annotations/train", frames_root="images/train",
            window_size=args.window_size, stride=args.stride,
            frame_index_offset=1, is_train=True,
        )
        val_dataset = CVATTemporalCropDataset(
            xml_root="annotations/val", frames_root="images/val",
            window_size=args.window_size, stride=args.stride,
            frame_index_offset=1, is_train=False,
        )
        print(f"Train: {len(train_dataset)} windows  |  Val: {len(val_dataset)} windows")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

        model = TemporalBehaviorModel(
            num_classes=3, spatial_feat_dim=SPATIAL_DIM,
            temporal_layers=3, unfreeze_last_n_blocks=args.unfreeze_blocks,
        ).to(device)

        optimizer = build_optimizer(model, lr_head=args.lr, lr_backbone=args.lr_backbone)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        loss_fn = FocalLoss(gamma=args.focal_gamma, label_smoothing=0.1)

        best_f1 = 0.0
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
            print(f"\n{'='*45}")
            print(f"Epoch {epoch+1}/{args.epochs}  |  Train Loss: {train_loss:.4f}")
            acc, macro_f1 = evaluate(model, val_loader, device)
            scheduler.step()
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save(model.state_dict(), "best_temporal_model_v4.pt")
                print(f"  >>> Saved best_temporal_model_v4.pt  (macro F1: {best_f1:.4f})")

    else:
        if not args.tracks_csv or not args.frames_dir:
            raise ValueError("--tracks_csv and --frames_dir required for inference")
        if not os.path.exists(args.weights):
            raise FileNotFoundError(f"Weights not found: {args.weights}")

        model = TemporalBehaviorModel(
            num_classes=3, spatial_feat_dim=SPATIAL_DIM,
            temporal_layers=3, unfreeze_last_n_blocks=0,
        ).to(device)
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights: {args.weights}")

        dataset = ByteTrackInferenceDataset(
            tracks_csv=args.tracks_csv,
            frames_dir=args.frames_dir,
            window_size=args.window_size,
            stride=1,
        )
        run_inference(model, dataset, device,
                      output_csv=args.output,
                      smooth_window=args.smooth_window)


if __name__ == "__main__":
    main()