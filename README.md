# AID2026 — Automatic Incident Detection Pipeline

**Competition:** Automatic Incident Detection Challenge 2026 (AID2026)  
**Primary metric:** F1-Score on test set  
**Secondary metrics:** Notification Delay, Processing FPS, GPU Memory

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                           │
│                                                                 │
│  Video ──► Overlapping Clips ──► VideoMAE-Small Backbone        │
│                                        │                        │
│                               Per-clip CLS Embeddings           │
│                                        │                        │
│                         Temporal Scoring Head (1D Conv + MLP)   │
│                                        │                        │
│                            Anomaly Score Stream [0,1]           │
│                                        │                        │
│                       CUSUM Change-Point Detector               │
│                                        │                        │
│                        Onset Second (or No Detection)           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Choice | Rationale |
|--------|-----------|
| **VideoMAE-Small backbone** | 22M params, strong transfer, fits A100 with batch=4 |
| **Overlapping 16-frame clips** | Captures motion patterns; overlap reduces onset miss rate |
| **Temporal context window** | 1D conv over last 8 clips → detects gradual incident buildup |
| **CUSUM change-point detection** | Robust onset localization; handles score noise better than hard threshold |
| **BF16 AMP training** | ~2× speed on A100, negligible accuracy loss |
| **Learnable threshold** | Avoids manual tuning; adapts to dataset distribution |
| **KL-div onset loss** | Pushes model to spike score precisely at annotated onset |

---

## Project Structure

```
aid2026/
├── test.py                     # Official submission entry point
├── src/
│   ├── model.py                # IncidentDetector + OnsetDetector
│   ├── dataset.py              # MIVIADataset + DataLoader
│   └── train.py                # Trainer with F1 optimization
├── scripts/
│   └── prepare_data.py         # Train/val split generation
├── notebooks/
│   └── aid2026_training.ipynb  # Full Colab training notebook
├── checkpoints/                # Saved model weights (created at training)
├── configs/                    # JSON training configs
└── data/                       # CSVs + split info (not videos)
```

---

## Quick Start

### 1. Prepare Data

```bash
python scripts/prepare_data.py \
    --csv data/annotations.csv \
    --output_dir data/ \
    --videos_dir data/videos/ \
    --val_ratio 0.15
```

### 2. Train

```bash
python src/train.py \
    --train_csv data/train.csv \
    --val_csv   data/val.csv \
    --videos_dir data/videos/ \
    --checkpoint_dir checkpoints/ \
    --epochs 30 \
    --batch_size 4
```

Or use the JSON config:
```bash
python src/train.py --config configs/train_config.json
```

### 3. Official Test Submission

```bash
python test.py --videos foo_videos/ --results foo_results/
```

---

## Evaluation Protocol (AID2026)

| Category | Condition |
|----------|-----------|
| **True Positive** | Detected in positive video within valid window: `[onset - tol, onset + max_delay]` |
| **False Positive** | Detection in negative video OR outside valid window |
| **False Negative** | Positive video with no detection |

**Final ranking:** F1-Score = 2 × (P × R) / (P + R)

---

## Dependencies

```
torch >= 2.2
transformers >= 4.40
torchvision
decord          # fast video loading
scikit-learn    # metrics
opencv-python-headless
einops
timm
```

Install:
```bash
pip install transformers accelerate decord torchvision scikit-learn opencv-python-headless einops timm
```

---

## Expected Performance

On MIVIA-AID validation set (15% split):

| Metric | Expected Range |
|--------|---------------|
| F1-Score | 0.78 – 0.88 |
| Precision | 0.80 – 0.90 |
| Recall | 0.76 – 0.86 |
| Avg Delay | 1.5 – 3.0s |

---

## Optimization Tips for Higher F1

1. **Unfreeze more backbone layers** after 10 epochs (set `freeze_layers=4`)
2. **Test-time augmentation**: average scores from horizontally flipped clips
3. **Ensemble**: train 3 seeds, average score streams before CUSUM
4. **Larger backbone**: swap to `MCG-NJU/videomae-base` if VRAM allows
5. **Post-processing**: calibrate threshold on validation set per-class if class balance shifts
