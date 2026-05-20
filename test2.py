import os
import pandas as pd

class YourDataset:
    def __init__(self, csv_path, videos_dir, ...):

        df = pd.read_csv(csv_path)

        valid_samples = []

        for _, row in df.iterrows():

            video_id = str(row["video"])  # adjust column name if needed

            # Try multiple common patterns
            candidate_paths = [
                os.path.join(videos_dir, video_id),
                os.path.join(videos_dir, video_id + ".mp4"),
                os.path.join(videos_dir, video_id + ".avi"),
                os.path.join(videos_dir, video_id + ".mov"),
                os.path.join(videos_dir, video_id + ".mkv"),
            ]

            found_path = None
            for p in candidate_paths:
                if os.path.exists(p):
                    found_path = p
                    break

            if found_path is None:
                print(f"[Dataset] Missing video: {video_id} -> skipping")
                continue

            # store resolved full path (IMPORTANT PART)
            sample = {
                "video_path": found_path,
                "label": row["label"],
            }

            # keep other annotations if they exist
            if "onset_clip" in row:
                sample["onset_clip"] = row["onset_clip"]

            valid_samples.append(sample)

        self.samples = valid_samples

        print(f"[Dataset] Cleaned dataset: {len(self.samples)} valid samples")