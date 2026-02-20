import os
import csv
import ast

CRASH_TXT_PATH = "data/raw_videos/CCD/Crash-1500.txt"
OUTPUT_CSV = "data/processed_frames/CCD/ccd_frame_labels.csv"

PRE_ACCIDENT_WINDOW = 10
TOTAL_FRAMES = 50

def parse_annotations():
    rows = []
    skipped = 0

    with open(CRASH_TXT_PATH, "r") as f:
        for line in f:
            line = line.strip()

            # Split CSV-style line
            parts = line.split(",", 1)

            video_id = parts[0]
            rest = parts[1]

            # Extract the list inside brackets
            label_list_str = rest[rest.find("["):rest.find("]") + 1]
            frame_labels = ast.literal_eval(label_list_str)

            if len(frame_labels) != TOTAL_FRAMES or 1 not in frame_labels:
                skipped += 1
                continue

            accident_frame = frame_labels.index(1)

            for frame_idx in range(TOTAL_FRAMES):
                if frame_idx < accident_frame - PRE_ACCIDENT_WINDOW:
                    label = 0   # normal
                elif accident_frame - PRE_ACCIDENT_WINDOW <= frame_idx < accident_frame:
                    label = 1   # pre-accident
                else:
                    label = 2   # accident

                rows.append([video_id, frame_idx, label])

    print(f"Skipped {skipped} videos with invalid labels")
    return rows

def save_csv(rows):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "frame_idx", "label"])
        writer.writerows(rows)

if __name__ == "__main__":
    rows = parse_annotations()
    save_csv(rows)
    print(f"✅ Saved frame-level labels to {OUTPUT_CSV}")
