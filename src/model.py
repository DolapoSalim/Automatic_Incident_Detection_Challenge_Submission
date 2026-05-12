"""
AID2026 — Incident Detection Model
Architecture: VideoMAE-V2 Small backbone + Temporal Scoring Head + Change-Point Detection
Design philosophy:
  - Clip-level binary scoring for speed
  - Change-point detection on score stream for precise onset localization
  - INT8-quantization-friendly design for Colab FPS maximization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel, VideoMAEConfig
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# 1. Temporal Scoring Head
# ---------------------------------------------------------------------------

class TemporalScoringHead(nn.Module):
    """
    Takes per-clip CLS embeddings and outputs an anomaly score in [0, 1].
    Uses a lightweight MLP + temporal context via a 1D conv across a buffer
    of recent clip embeddings.
    """

    def __init__(self, hidden_dim: int = 768, context_clips: int = 8, dropout: float = 0.3):
        super().__init__()
        self.context_clips = context_clips
        self.hidden_dim = hidden_dim

        # Temporal context aggregation over recent clips
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, clip_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_embeddings: (B, T, D) — batch of T clip embeddings
        Returns:
            scores: (B, T) — anomaly score per clip
        """
        B, T, D = clip_embeddings.shape
        # Sliding window scores: for each position t, look back context_clips
        scores = []
        for t in range(T):
            start = max(0, t - self.context_clips + 1)
            window = clip_embeddings[:, start:t+1, :]  # (B, W, D)
            # Pad if window smaller than context_clips
            if window.shape[1] < self.context_clips:
                pad = self.context_clips - window.shape[1]
                window = F.pad(window, (0, 0, pad, 0))
            x = window.permute(0, 2, 1)  # (B, D, W)
            x = self.temporal_conv(x)     # (B, 128, W)
            score = torch.sigmoid(self.classifier(x))  # (B, 1)
            scores.append(score)
        return torch.cat(scores, dim=1)  # (B, T)


# ---------------------------------------------------------------------------
# 2. Main Incident Detection Model
# ---------------------------------------------------------------------------

class IncidentDetector(nn.Module):
    """
    Full pipeline:
      1. VideoMAE backbone extracts per-clip features
      2. Temporal scoring head produces anomaly scores
      3. At inference: change-point detection on score stream → incident onset
    """

    def __init__(
        self,
        backbone_name: str = "MCG-NJU/videomae-small",
        num_frames: int = 16,
        tubelet_size: int = 2,
        context_clips: int = 8,
        dropout: float = 0.3,
        freeze_backbone_layers: int = 8,  # freeze first N transformer layers
    ):
        super().__init__()
        self.num_frames = num_frames

        # --- Backbone ---
        config = VideoMAEConfig.from_pretrained(backbone_name)
        config.num_frames = num_frames
        config.tubelet_size = tubelet_size
        self.backbone = VideoMAEModel.from_pretrained(backbone_name, config=config, ignore_mismatched_sizes=True)

        # Freeze early layers for efficiency
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < freeze_backbone_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        # --- Scoring Head ---
        hidden_dim = config.hidden_size
        self.head = TemporalScoringHead(
            hidden_dim=hidden_dim,
            context_clips=context_clips,
            dropout=dropout,
        )

        # --- Learnable threshold (sigmoid-scaled) ---
        self.log_threshold = nn.Parameter(torch.tensor(0.0))

    @property
    def threshold(self) -> float:
        return torch.sigmoid(self.log_threshold).item()

    def extract_clip_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, T, H, W) — standard VideoMAE input
        Returns:
            cls_embeds: (B, D)
        """
        outputs = self.backbone(pixel_values=pixel_values)
        # Mean-pool patch tokens (or use CLS-equivalent: mean of sequence)
        return outputs.last_hidden_state.mean(dim=1)

    def forward(
        self,
        clips: torch.Tensor,          # (B, num_clips, C, T, H, W)
        labels: Optional[torch.Tensor] = None,  # (B,) 0/1 binary
        onset_frames: Optional[torch.Tensor] = None,  # (B,) frame index of onset (-1 if negative)
    ):
        """
        Full forward pass over a sequence of clips.
        Returns dict with loss (if training) and scores.
        """
        B, num_clips, C, T, H, W = clips.shape

        # Extract features for all clips in parallel (flatten batch × clips)
        clips_flat = clips.view(B * num_clips, C, T, H, W)
        feats_flat = self.extract_clip_features(clips_flat)  # (B*num_clips, D)
        feats = feats_flat.view(B, num_clips, -1)            # (B, num_clips, D)

        # Temporal scores
        scores = self.head(feats)  # (B, num_clips)

        output = {"scores": scores, "threshold": self.threshold}

        if labels is not None and onset_frames is not None:
            loss = self._compute_loss(scores, labels, onset_frames, num_clips)
            output["loss"] = loss

        return output

    def _compute_loss(
        self,
        scores: torch.Tensor,       # (B, num_clips)
        labels: torch.Tensor,       # (B,)
        onset_frames: torch.Tensor, # (B,) clip index of onset, -1 for negatives
        num_clips: int,
    ) -> torch.Tensor:
        """
        Combined loss:
          - Binary cross-entropy on video-level prediction (max score)
          - Temporal onset loss: push score spike to onset clip
        """
        # Video-level prediction: max score across clips
        video_scores = scores.max(dim=1).values  # (B,)
        bce_loss = F.binary_cross_entropy(video_scores, labels.float())

        # Temporal onset focal loss for positive samples
        pos_mask = labels == 1
        onset_loss = torch.tensor(0.0, device=scores.device)
        if pos_mask.any():
            pos_scores = scores[pos_mask]          # (B_pos, num_clips)
            pos_onsets = onset_frames[pos_mask]    # (B_pos,)

            # Build soft target: gaussian peaked at onset clip
            B_pos = pos_scores.shape[0]
            clip_indices = torch.arange(num_clips, device=scores.device).float()
            sigma = max(num_clips * 0.05, 1.0)
            targets = torch.zeros_like(pos_scores)
            for i in range(B_pos):
                o = pos_onsets[i].float()
                gauss = torch.exp(-0.5 * ((clip_indices - o) / sigma) ** 2)
                targets[i] = gauss / (gauss.sum() + 1e-8)

            # KL divergence between predicted distribution and onset target
            log_probs = F.log_softmax(pos_scores, dim=1)
            onset_loss = F.kl_div(log_probs, targets, reduction="batchmean")

        return bce_loss + 0.5 * onset_loss


# ---------------------------------------------------------------------------
# 3. Change-Point Detection for Inference
# ---------------------------------------------------------------------------

class OnsetDetector:
    """
    Runs CUSUM-style change-point detection on the clip score stream
    to find the precise onset of an incident.
    Uses an adaptive threshold derived from the model's learned threshold.
    """

    def __init__(
        self,
        base_threshold: float = 0.5,
        cusum_slack: float = 0.1,
        min_run_length: int = 2,
        smoothing_window: int = 3,
    ):
        self.base_threshold = base_threshold
        self.cusum_slack = cusum_slack
        self.min_run_length = min_run_length
        self.smoothing_window = smoothing_window

    def detect(self, scores: np.ndarray) -> Optional[int]:
        """
        Args:
            scores: 1D array of anomaly scores per clip
        Returns:
            onset_clip_index (int) or None if no incident detected
        """
        # Smooth scores
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        smoothed = np.convolve(scores, kernel, mode="same")

        # CUSUM
        cusum = 0.0
        run_length = 0
        for i, s in enumerate(smoothed):
            cusum = max(0.0, cusum + s - self.base_threshold - self.cusum_slack)
            if s > self.base_threshold:
                run_length += 1
            else:
                run_length = 0

            if cusum > 1.0 and run_length >= self.min_run_length:
                # Walk back to find true onset (first clip above threshold)
                onset = i
                while onset > 0 and smoothed[onset - 1] > self.base_threshold * 0.7:
                    onset -= 1
                return onset

        return None
