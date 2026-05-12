"""
AID2026 — Training Loop
Features:
  - Mixed precision (bfloat16) for Colab A100/T4
  - Cosine LR schedule with warm-up
  - Threshold optimization on validation set
  - Early stopping on F1
  - Gradient clipping
  - Checkpoint saving (best F1)
"""

import os
import time
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, precision_score, recall_score

from model import IncidentDetector, OnsetDetector
from dataset import build_dataloaders


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    all_scores: np.ndarray,    # (N, max_clips)
    all_labels: np.ndarray,    # (N,)
    all_onset_clips: np.ndarray,  # (N,) -1 for negatives
    all_fps: np.ndarray,       # (N,)
    all_start_frames: list,    # list of (num_clips,) arrays
    threshold: float,
    tolerance_sec: float = 5.0,
    max_delay_sec: float = 5.0,
) -> Dict:
    """
    Implements AID2026 evaluation protocol exactly:
      TP: detection in positive video within [onset - tol, onset + max_delay]
      FP: detection in negative video OR outside window
      FN: positive video with no detection
    """
    TPs, FPs, FNs = 0, 0, 0
    delays = []

    onset_detector = OnsetDetector(base_threshold=threshold)

    for i in range(len(all_labels)):
        scores_i = all_scores[i]  # (max_clips,)
        label = int(all_labels[i])
        fps = float(all_fps[i])
        onset_clip_gt = int(all_onset_clips[i])
        start_frames = all_start_frames[i]  # numpy array

        # Detect onset
        detected_clip = onset_detector.detect(scores_i)

        if label == 0:
            # Negative video
            if detected_clip is not None:
                FPs += 1
            # else: TN (not counted)
        else:
            # Positive video — check timing
            if detected_clip is None:
                FNs += 1
            else:
                # Convert clips to seconds
                clip_hop_frames = 8  # must match dataset
                onset_frame_gt = int(start_frames[onset_clip_gt]) if onset_clip_gt < len(start_frames) else 0
                det_frame = int(start_frames[detected_clip]) if detected_clip < len(start_frames) else 0

                onset_sec_gt = onset_frame_gt / fps
                det_sec = det_frame / fps

                delay = det_sec - onset_sec_gt
                early = -delay  # positive means detected before onset

                if early > tolerance_sec:
                    FPs += 1  # too early
                elif delay > max_delay_sec:
                    FPs += 1  # too late
                else:
                    TPs += 1
                    delays.append(max(0.0, delay))

    precision = TPs / (TPs + FPs + 1e-8)
    recall    = TPs / (TPs + FNs + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    avg_delay = float(np.mean(delays)) if delays else float(max_delay_sec)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "avg_delay_sec": avg_delay,
        "TPs": TPs, "FPs": FPs, "FNs": FNs,
    }


def find_best_threshold(
    all_scores: np.ndarray,
    all_labels: np.ndarray,
    all_onset_clips: np.ndarray,
    all_fps: np.ndarray,
    all_start_frames: list,
) -> Tuple[float, Dict]:
    """Grid search over threshold values to maximize F1."""
    best_f1 = -1.0
    best_thr = 0.5
    best_metrics = {}
    for thr in np.arange(0.2, 0.85, 0.05):
        m = compute_metrics(
            all_scores, all_labels, all_onset_clips,
            all_fps, all_start_frames, threshold=thr
        )
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(thr)
            best_metrics = m
    return best_thr, best_metrics


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Device: {self.device}")

        # Model
        self.model = IncidentDetector(
            backbone_name=cfg["backbone"],
            num_frames=cfg["clip_frames"],
            context_clips=cfg["context_clips"],
            dropout=cfg["dropout"],
            freeze_backbone_layers=cfg["freeze_layers"],
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Trainer] Trainable parameters: {n_params:,}")

        # Data
        self.train_loader, self.val_loader = build_dataloaders(
            train_csv=cfg["train_csv"],
            val_csv=cfg["val_csv"],
            videos_dir=cfg["videos_dir"],
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            clip_frames=cfg["clip_frames"],
            clip_hop=cfg["clip_hop"],
            max_clips=cfg["max_clips"],
        )

        # Optimizer & scheduler
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        total_steps = len(self.train_loader) * cfg["epochs"]
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=cfg["lr"],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )

        # AMP
        self.scaler = GradScaler(enabled=cfg.get("amp", True))

        # Paths
        self.ckpt_dir = Path(cfg["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_f1 = 0.0
        self.patience_counter = 0
        self.best_threshold = 0.5

    # -----------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(self.train_loader):
            clips       = batch["clips"].to(self.device)         # (B, N, C, T, H, W)
            labels      = batch["label"].to(self.device)
            onset_clips = batch["onset_clip"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16, enabled=self.cfg.get("amp", True)):
                out = self.model(clips, labels=labels, onset_frames=onset_clips)
                loss = out["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()

            if step % 20 == 0:
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(f"  [E{epoch} S{step}/{len(self.train_loader)}] "
                      f"loss={loss.item():.4f} lr={lr:.2e} t={elapsed:.1f}s")

        return total_loss / len(self.train_loader)

    # -----------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        all_scores, all_labels, all_onset_clips = [], [], []
        all_fps, all_start_frames = [], []

        for batch in self.val_loader:
            clips       = batch["clips"].to(self.device)
            labels      = batch["label"]
            onset_clips = batch["onset_clip"]
            fps         = batch["fps"]
            start_frames= batch["start_frames"]

            with autocast(dtype=torch.bfloat16, enabled=self.cfg.get("amp", True)):
                out = self.model(clips)

            scores = out["scores"].cpu().float().numpy()  # (B, max_clips)
            all_scores.append(scores)
            all_labels.append(labels.numpy())
            all_onset_clips.append(onset_clips.numpy())
            all_fps.append(fps.numpy())
            for sf in start_frames:
                all_start_frames.append(sf.numpy())

        all_scores      = np.concatenate(all_scores, axis=0)
        all_labels      = np.concatenate(all_labels, axis=0)
        all_onset_clips = np.concatenate(all_onset_clips, axis=0)
        all_fps         = np.concatenate(all_fps, axis=0)

        # Find best threshold on validation set
        best_thr, metrics = find_best_threshold(
            all_scores, all_labels, all_onset_clips,
            all_fps, all_start_frames
        )
        metrics["threshold"] = best_thr
        return metrics

    # -----------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, metrics: Dict, tag: str = "best"):
        path = self.ckpt_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
            "threshold": metrics.get("threshold", 0.5),
            "cfg": self.cfg,
        }, path)
        print(f"  [Checkpoint] Saved {path}  (F1={metrics['f1']:.4f})")

    # -----------------------------------------------------------------------

    def fit(self):
        print(f"\n{'='*60}")
        print(f"  AID2026 Training — {self.cfg['epochs']} epochs")
        print(f"{'='*60}\n")

        for epoch in range(1, self.cfg["epochs"] + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.best_threshold = val_metrics["threshold"]

            print(f"\n[Epoch {epoch}] loss={train_loss:.4f} | "
                  f"F1={val_metrics['f1']:.4f} P={val_metrics['precision']:.4f} "
                  f"R={val_metrics['recall']:.4f} | "
                  f"delay={val_metrics['avg_delay_sec']:.2f}s | "
                  f"thr={val_metrics['threshold']:.2f}")

            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, tag="best")
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_metrics, tag="last")

            # Save metrics log
            log_path = self.ckpt_dir / "metrics_log.jsonl"
            with open(log_path, "a") as f:
                f.write(json.dumps({"epoch": epoch, "train_loss": train_loss, **val_metrics}) + "\n")

            # Early stopping
            if self.patience_counter >= self.cfg.get("patience", 10):
                print(f"\n[Early Stop] No improvement for {self.patience_counter} epochs.")
                break

        print(f"\n{'='*60}")
        print(f"  Training complete. Best Val F1: {self.best_f1:.4f}")
        print(f"  Best threshold: {self.best_threshold:.2f}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def get_default_config() -> Dict:
    return {
        # Model
        "backbone":        "MCG-NJU/videomae-small",
        "clip_frames":     16,
        "context_clips":   8,
        "dropout":         0.3,
        "freeze_layers":   8,
        # Data
        "train_csv":       "data/train.csv",
        "val_csv":         "data/val.csv",
        "videos_dir":      "data/videos",
        "clip_hop":        8,
        "max_clips":       48,
        # Training
        "batch_size":      4,
        "num_workers":     2,
        "epochs":          30,
        "lr":              2e-4,
        "weight_decay":    1e-4,
        "patience":        8,
        "amp":             True,
        "checkpoint_dir":  "checkpoints",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    parser.add_argument("--train_csv",   type=str)
    parser.add_argument("--val_csv",     type=str)
    parser.add_argument("--videos_dir",  type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--epochs",      type=int)
    parser.add_argument("--batch_size",  type=int)
    parser.add_argument("--backbone",    type=str)
    args = parser.parse_args()

    cfg = get_default_config()

    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))

    # CLI overrides
    for k in ["train_csv", "val_csv", "videos_dir", "checkpoint_dir",
              "epochs", "batch_size", "backbone"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    trainer = Trainer(cfg)
    trainer.fit()
