"""
AID2026 — Official Submission Script
Usage:
    python test.py --videos foo_videos/ --results foo_results/

Outputs one CSV per video in foo_results/ with 'start' column (seconds or empty).
"""

import os
import sys
import csv
import time
import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.cuda.amp import autocast

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (relative to this script — works in Colab after unzip)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
MODEL_CKPT   = SCRIPT_DIR / "checkpoints" / "checkpoint_best.pt"
SRC_DIR      = SCRIPT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from model   import IncidentDetector, OnsetDetector
from dataset import read_video_frames, extract_clips, build_transforms, CLIP_FRAMES, CLIP_STRIDE, CLIP_HOP, MAX_CLIPS


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load checkpoint
        print(f"[Inference] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg  = ckpt.get("cfg", {})

        # Build model
        self.model = IncidentDetector(
            backbone_name    = cfg.get("backbone",       "MCG-NJU/videomae-small"),
            num_frames       = cfg.get("clip_frames",    CLIP_FRAMES),
            context_clips    = cfg.get("context_clips",  8),
            freeze_backbone_layers=0,  # no freezing at inference
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # Threshold from training
        self.threshold = float(ckpt.get("threshold", 0.5))
        self.onset_detector = OnsetDetector(base_threshold=self.threshold)

        # Config
        self.clip_frames = cfg.get("clip_frames", CLIP_FRAMES)
        self.clip_stride = cfg.get("clip_stride", CLIP_STRIDE)
        self.clip_hop    = cfg.get("clip_hop",    CLIP_HOP)
        self.max_clips   = cfg.get("max_clips",   MAX_CLIPS)
        self.transform   = build_transforms(train=False)

        print(f"[Inference] Model ready. Threshold={self.threshold:.3f} Device={self.device}")

    @torch.no_grad()
    def predict(self, video_path: str) -> Optional[float]:
        """
        Returns:
            onset_sec (float) if incident detected, else None
        """
        # Read video
        frames, fps = read_video_frames(video_path)

        # Extract clips
        clips_np, start_frames = extract_clips(
            frames,
            self.clip_frames,
            self.clip_stride,
            self.clip_hop,
            self.max_clips,
        )
        num_clips = len(clips_np)

        # Build tensor: (1, N, C, T, H, W)
        clips_list = []
        for n in range(num_clips):
            clip_frames_np = clips_np[n]  # (T, H, W, 3)
            t_frames = []
            for f in range(clip_frames_np.shape[0]):
                frame_t = torch.from_numpy(clip_frames_np[f]).permute(2, 0, 1)  # (3, H, W)
                t_frames.append(self.transform(frame_t))
            clip_t = torch.stack(t_frames, dim=1)  # (3, T, H, W)
            clips_list.append(clip_t)

        clips_tensor = torch.stack(clips_list).unsqueeze(0).to(self.device)  # (1, N, C, T, H, W)

        # Forward pass
        with autocast(dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.model(clips_tensor)

        scores = out["scores"][0].cpu().float().numpy()  # (N,)

        # Detect onset
        detected_clip = self.onset_detector.detect(scores)

        if detected_clip is None:
            return None

        # Convert to seconds
        det_frame = start_frames[detected_clip] if detected_clip < len(start_frames) else 0
        onset_sec = det_frame / fps
        return float(onset_sec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AID2026 Incident Detection")
    parser.add_argument("--videos",  type=str, required=True, help="Path to folder of test videos")
    parser.add_argument("--results", type=str, required=True, help="Path to output results folder")
    parser.add_argument("--ckpt",    type=str, default=str(MODEL_CKPT), help="Model checkpoint path")
    parser.add_argument("--device",  type=str, default=None, help="cuda / cpu (auto-detect if omitted)")
    args = parser.parse_args()

    videos_dir  = Path(args.videos)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Supported extensions
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    video_files = sorted([
        f for f in videos_dir.iterdir()
        if f.suffix.lower() in VIDEO_EXTS
    ])

    if not video_files:
        print(f"[Warning] No video files found in {videos_dir}")
        return

    print(f"[Test] Found {len(video_files)} videos")

    # Load model
    engine = InferenceEngine(ckpt_path=args.ckpt, device=args.device)

    # Results CSV
    results_csv = results_dir / "results.csv"
    t_total = time.time()

    with open(results_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Id Video", "start"])  # header

        for i, vp in enumerate(video_files):
            t0 = time.time()
            try:
                onset_sec = engine.predict(str(vp))
            except Exception as e:
                print(f"  [Error] {vp.name}: {e}")
                onset_sec = None

            elapsed = time.time() - t0
            start_val = f"{onset_sec:.3f}" if onset_sec is not None else ""
            writer.writerow([vp.stem, start_val])

            status = f"INCIDENT @ {onset_sec:.2f}s" if onset_sec is not None else "no incident"
            print(f"  [{i+1:04d}/{len(video_files)}] {vp.name:<40} {status:<30} ({elapsed:.2f}s)")

    total_time = time.time() - t_total
    print(f"\n[Done] Results saved to {results_csv}")
    print(f"[Done] Total time: {total_time:.1f}s  ({total_time/len(video_files):.2f}s/video)")


if __name__ == "__main__":
    main()
