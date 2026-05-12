"""
AID2026 — Dataset & DataLoader
Handles MIVIA-AID video loading, clip extraction, and augmentation.
"""

import os
import csv
import json
import math
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF

# Try decord first (fast), fall back to torchvision
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    import torchvision.io as tvio


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_FRAMES   = 16      # frames per clip fed to VideoMAE
CLIP_STRIDE   = 8       # frame stride within a clip (temporal resolution)
SPATIAL_SIZE  = 224     # spatial resolution
CLIP_HOP      = 8       # hop between clip start frames (overlap = CLIP_FRAMES - CLIP_HOP)
MAX_CLIPS     = 64      # cap clips per video for memory


# ---------------------------------------------------------------------------
# Video Reading
# ---------------------------------------------------------------------------

def read_video_frames(path: str, max_frames: int = 4096) -> Tuple[np.ndarray, float]:
    """
    Returns:
        frames: (T, H, W, 3) uint8 numpy array
        fps: float
    """
    if DECORD_AVAILABLE:
        vr = VideoReader(path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        n = min(len(vr), max_frames)
        frames = vr.get_batch(list(range(n))).asnumpy()  # (T,H,W,3)
        return frames, fps
    else:
        video, _, info = tvio.read_video(path, pts_unit="sec")
        fps = info.get("video_fps", 25.0)
        frames = video.numpy()  # (T,H,W,3)
        if len(frames) > max_frames:
            frames = frames[:max_frames]
        return frames, fps


def extract_clips(
    frames: np.ndarray,
    clip_frames: int = CLIP_FRAMES,
    clip_stride: int = CLIP_STRIDE,
    clip_hop: int = CLIP_HOP,
    max_clips: int = MAX_CLIPS,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extracts overlapping clips from a video.
    Returns:
        clips:        (N, clip_frames, H, W, 3)
        start_frames: list of N start frame indices
    """
    T = len(frames)
    # Start positions for each clip
    starts = list(range(0, T - clip_frames * clip_stride + 1, clip_hop))
    if not starts:
        starts = [0]
    starts = starts[:max_clips]

    clips = []
    for s in starts:
        indices = [min(s + i * clip_stride, T - 1) for i in range(clip_frames)]
        clip = frames[indices]  # (clip_frames, H, W, 3)
        clips.append(clip)

    return np.stack(clips), starts  # (N, clip_frames, H, W, 3), [int...]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(train: bool, size: int = SPATIAL_SIZE):
    ops = []
    if train:
        ops += [
            T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.8, 1.2), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.05),
        ]
    else:
        ops += [
            T.Resize(int(size * 1.12), antialias=True),
            T.CenterCrop(size),
        ]
    ops += [
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(ops)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MIVIADataset(Dataset):
    """
    MIVIA-AID dataset.

    CSV format expected:
        Id Video, Duration, Start, End
        (Start/End empty for negative samples)

    Args:
        csv_path:    path to annotations CSV
        videos_dir:  directory containing video files
        train:       training mode (enables augmentations)
        clip_frames: frames per clip
        clip_stride: frame stride within clip
        clip_hop:    frames between clip starts
        max_clips:   max clips per video
    """

    def __init__(
        self,
        csv_path: str,
        videos_dir: str,
        train: bool = True,
        clip_frames: int = CLIP_FRAMES,
        clip_stride: int = CLIP_STRIDE,
        clip_hop: int = CLIP_HOP,
        max_clips: int = MAX_CLIPS,
    ):
        self.videos_dir = Path(videos_dir)
        self.train = train
        self.clip_frames = clip_frames
        self.clip_stride = clip_stride
        self.clip_hop = clip_hop
        self.max_clips = max_clips
        self.transform = build_transforms(train)

        # Load annotations
        self.samples = self._load_csv(csv_path)
        print(f"[Dataset] Loaded {len(self.samples)} samples "
              f"({sum(s['label'] for s in self.samples)} positive) "
              f"from {csv_path}")

    def _load_csv(self, path: str) -> List[Dict]:
        samples = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row["Id Video"].strip()
                duration = float(row.get("Duration", 0) or 0)
                start_str = row.get("Start", "").strip()
                end_str = row.get("End", "").strip()
                label = 1 if start_str else 0
                onset_sec = float(start_str) if start_str else -1.0
                end_sec = float(end_str) if end_str else -1.0
                samples.append({
                    "video_id": vid,
                    "duration": duration,
                    "label": label,
                    "onset_sec": onset_sec,
                    "end_sec": end_sec,
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        vid_path = self._find_video(sample["video_id"])

        # Read video
        frames, fps = read_video_frames(str(vid_path))
        total_frames = len(frames)

        # Extract clips
        clips_np, start_frames = extract_clips(
            frames,
            self.clip_frames,
            self.clip_stride,
            self.clip_hop,
            self.max_clips,
        )
        num_clips = len(clips_np)

        # Compute onset clip index
        if sample["label"] == 1 and sample["onset_sec"] >= 0:
            onset_frame = int(sample["onset_sec"] * fps)
            # Find closest clip
            dists = [abs(sf - onset_frame) for sf in start_frames]
            onset_clip = int(np.argmin(dists))
        else:
            onset_clip = -1

        # Apply transforms: clips_np is (N, T, H, W, 3)
        clips_tensor = self._apply_transforms(clips_np)  # (N, C, T, H, W)

        return {
            "clips": clips_tensor,                              # (N, C, T, H, W)
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "onset_clip": torch.tensor(onset_clip, dtype=torch.long),
            "onset_sec": torch.tensor(sample["onset_sec"], dtype=torch.float),
            "fps": torch.tensor(fps, dtype=torch.float),
            "start_frames": torch.tensor(start_frames, dtype=torch.long),
            "video_id": sample["video_id"],
        }

    def _find_video(self, video_id: str) -> Path:
        """Try common extensions."""
        for ext in ["", ".mp4", ".avi", ".mov", ".mkv"]:
            p = self.videos_dir / f"{video_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Video not found: {video_id} in {self.videos_dir}")

    def _apply_transforms(self, clips_np: np.ndarray) -> torch.Tensor:
        """
        clips_np: (N, T, H, W, 3) uint8
        Returns: (N, 3, T, H, W) float tensor
        """
        N, T, H, W, C = clips_np.shape
        out = []
        for n in range(N):
            # (T, H, W, 3) → (T, 3, H, W)
            frames = torch.from_numpy(clips_np[n]).permute(0, 3, 1, 2)
            # Apply same spatial transform consistently per clip
            # For training: sample random params once per clip
            t_frames = []
            for f in range(T):
                t_frames.append(self.transform(frames[f]))
            clip_t = torch.stack(t_frames, dim=1)  # (3, T, H, W)
            out.append(clip_t)
        return torch.stack(out)  # (N, 3, T, H, W)


# ---------------------------------------------------------------------------
# Collate Function (handles variable num_clips across videos)
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Pads clips to same length within batch.
    """
    max_clips = max(b["clips"].shape[0] for b in batch)
    C, T, H, W = batch[0]["clips"].shape[1:]

    clips_padded = []
    padding_masks = []
    for b in batch:
        n = b["clips"].shape[0]
        pad = max_clips - n
        if pad > 0:
            padding = torch.zeros(pad, C, T, H, W)
            clips_padded.append(torch.cat([b["clips"], padding], dim=0))
        else:
            clips_padded.append(b["clips"])
        mask = torch.zeros(max_clips, dtype=torch.bool)
        mask[:n] = True
        padding_masks.append(mask)

    return {
        "clips": torch.stack(clips_padded),          # (B, max_clips, C, T, H, W)
        "padding_mask": torch.stack(padding_masks),   # (B, max_clips) True=valid
        "label": torch.stack([b["label"] for b in batch]),
        "onset_clip": torch.stack([b["onset_clip"] for b in batch]),
        "onset_sec": torch.stack([b["onset_sec"] for b in batch]),
        "fps": torch.stack([b["fps"] for b in batch]),
        "start_frames": [b["start_frames"] for b in batch],
        "video_ids": [b["video_id"] for b in batch],
    }


def build_dataloaders(
    train_csv: str,
    val_csv: str,
    videos_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = MIVIADataset(train_csv, videos_dir, train=True, **dataset_kwargs)
    val_ds   = MIVIADataset(val_csv,   videos_dir, train=False, **dataset_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader
