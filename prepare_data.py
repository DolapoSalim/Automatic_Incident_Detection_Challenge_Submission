"""
AID2026 — Data Preparation
Splits the MIVIA-AID annotations CSV into stratified train/val splits.
Also runs a quick dataset health-check.

Usage:
    python scripts/prepare_data.py \
        --csv data/annotations.csv \
        --output_dir data/ \
        --val_ratio 0.15 \
        --seed 42
"""

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path


def load_csv(path: str):
    samples = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(dict(row))
    return samples


def stratified_split(samples, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    positives = [s for s in samples if s.get("Start", "").strip()]
    negatives = [s for s in samples if not s.get("Start", "").strip()]

    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_val_pos = max(1, int(len(positives) * val_ratio))
    n_val_neg = max(1, int(len(negatives) * val_ratio))

    val   = positives[:n_val_pos] + negatives[:n_val_neg]
    train = positives[n_val_pos:] + negatives[n_val_neg:]

    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


def write_csv(samples, path: str, fieldnames=None):
    if not samples:
        return
    if fieldnames is None:
        fieldnames = list(samples[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    print(f"  Wrote {len(samples)} rows → {path}")


def health_check(samples, videos_dir: str):
    print("\n[Health Check]")
    missing = []
    VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
    vdir = Path(videos_dir)

    for s in samples:
        vid = s.get("Id Video", "").strip()
        found = any((vdir / f"{vid}{ext}").exists() for ext in [""] + VIDEO_EXTS)
        if not found:
            missing.append(vid)

    if missing:
        print(f"  ⚠ Missing {len(missing)} videos: {missing[:5]}{'...' if len(missing)>5 else ''}")
    else:
        print(f"  ✓ All {len(samples)} video files found")

    # Duration stats
    durations = []
    for s in samples:
        d = s.get("Duration", "").strip()
        if d:
            try:
                durations.append(float(d))
            except ValueError:
                pass
    if durations:
        print(f"  Duration: min={min(durations):.1f}s  max={max(durations):.1f}s  "
              f"mean={sum(durations)/len(durations):.1f}s")

    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        required=True, help="Path to full annotations CSV")
    parser.add_argument("--output_dir", default="data/", help="Where to write train.csv / val.csv")
    parser.add_argument("--videos_dir", default="data/videos/", help="Videos directory for health check")
    parser.add_argument("--val_ratio",  type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[Prepare] Loading {args.csv}")
    samples = load_csv(args.csv)
    print(f"  Total samples: {len(samples)}")

    pos = sum(1 for s in samples if s.get("Start", "").strip())
    neg = len(samples) - pos
    print(f"  Positive: {pos}  Negative: {neg}")

    # Split
    train, val = stratified_split(samples, val_ratio=args.val_ratio, seed=args.seed)
    print(f"\n[Split] Train={len(train)} ({sum(1 for s in train if s.get('Start','').strip())}+) "
          f"Val={len(val)} ({sum(1 for s in val if s.get('Start','').strip())}+)")

    fieldnames = list(samples[0].keys())
    write_csv(train, os.path.join(args.output_dir, "train.csv"), fieldnames)
    write_csv(val,   os.path.join(args.output_dir, "val.csv"),   fieldnames)

    # Health check
    if os.path.isdir(args.videos_dir):
        health_check(samples, args.videos_dir)
    else:
        print(f"\n[Health Check] Skipped — {args.videos_dir} not found")

    # Save split info
    info = {
        "total": len(samples),
        "train": len(train),
        "val": len(val),
        "train_pos": sum(1 for s in train if s.get("Start","").strip()),
        "val_pos": sum(1 for s in val if s.get("Start","").strip()),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "split_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n[Done] Split info saved.")


if __name__ == "__main__":
    main()
