"""
Usage:
    python Visualize.py \
        --frames      images/test/clip1 \
        --tracks_csv  tracks/clip1.csv \
        --preds_csv   predictions/clip_01_behaviors.csv \
        --output_dir  visualizations/clip_01 \
        --video       visualizations/clip_01.mp4
"""

import cv2
import csv
import os
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional

BEHAVIOR_COLORS = {
    "on_box":  (34,  197,  94),   # green
    "at_hole": (249, 115,  22),   # orange
    "in_box":  (239,  68,  68),   # red
}

BEHAVIOR_MAP     = {"on_box": 0, "at_hole": 1, "in_box": 2}
INV_BEHAVIOR_MAP = {v: k for k, v in BEHAVIOR_MAP.items()}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2

def draw_solid_box(frame, x1, y1, x2, y2, color, thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_dashed_box(frame, x1, y1, x2, y2, color, dash_len=10, thickness=2):
    """Dashed rectangle for occluded/in_box birds."""
    for x in range(x1, x2, dash_len * 2):
        cv2.line(frame, (x, y1), (min(x + dash_len, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash_len, x2), y2), color, thickness)
    for y in range(y1, y2, dash_len * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash_len, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash_len, y2)), color, thickness)


def draw_label(frame, text, x, y, color, bg=True):
    """Draw text with a colored background pill."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    pad = 4
    if bg:
        cv2.rectangle(frame,
                      (x - pad, y - th - pad),
                      (x + tw + pad, y + baseline + pad),
                      color, -1)
        text_color = (0, 0, 0)   # black text on colored bg
    else:
        text_color = color
    cv2.putText(frame, text, (x, y), FONT, FONT_SCALE, text_color, THICKNESS, cv2.LINE_AA)


def draw_legend(frame):
    """Top-right corner: behavior color legend."""
    h, w = frame.shape[:2]
    items = [("on_box",  BEHAVIOR_COLORS["on_box"]),
             ("at_hole", BEHAVIOR_COLORS["at_hole"]),
             ("in_box",  BEHAVIOR_COLORS["in_box"])]

    x_start = w - 160
    y_start = 15
    for i, (label, color) in enumerate(items):
        y = y_start + i * 28
        cv2.rectangle(frame, (x_start, y), (x_start + 18, y + 18), color, -1)
        cv2.putText(frame, label, (x_start + 24, y + 14),
                    FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)


def draw_frame_number(frame, frame_idx):
    """Top-left: frame number."""
    cv2.putText(frame, f"Frame {frame_idx:05d}", (10, 28),
                FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frame {frame_idx:05d}", (10, 28),
                FONT, 0.65, (0, 0, 0), 1, cv2.LINE_AA)


def draw_bottom_bar(frame, counts: Dict[str, int]):
    """Bottom bar: running behavior frame counts."""
    h, w = frame.shape[:2]
    bar_h = 30
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (20, 20, 20), -1)

    x = 10
    for behavior, color in BEHAVIOR_COLORS.items():
        text = f"{behavior}: {counts[behavior]}"
        cv2.putText(frame, text, (x, h - 9),
                    FONT, 0.48, color, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.48, 1)
        x += tw + 25


def annotate_frame(
    frame:         np.ndarray,
    frame_idx:     int,
    track_data:    Dict[int, Tuple[float, float, float, float, bool]],  # tid → (x1,y1,x2,y2,occluded)
    pred_data:     Dict[int, str],                                       # tid → behavior_label
    counts:        Dict[str, int],
) -> np.ndarray:
    """
    Draws all annotations on a single frame.

    track_data:  { track_id: (x1, y1, x2, y2, occluded) }
    pred_data:   { track_id: behavior_label }
    counts:      running totals for bottom bar
    """
    for tid, (x1, y1, x2, y2, occluded) in track_data.items():
        behavior = pred_data.get(tid, "on_box")
        color    = BEHAVIOR_COLORS.get(behavior, (200, 200, 200))

        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

        # Box — dashed if occluded (bird inside box), solid otherwise
        if occluded:
            draw_dashed_box(frame, ix1, iy1, ix2, iy2, color, thickness=2)
        else:
            draw_solid_box(frame, ix1, iy1, ix2, iy2, color, thickness=2)

        # Behavior label above the box
        label_text = f"{behavior}"
        draw_label(frame, label_text, ix1, max(iy1 - 6, 20), color, bg=True)

        # Track ID inside the box (small, bottom-left of box)
        cv2.putText(frame, f"#{tid}", (ix1 + 4, min(iy2 - 4, iy2)),
                    FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        counts[behavior] += 1

    draw_frame_number(frame, frame_idx)
    draw_legend(frame)
    draw_bottom_bar(frame, counts)

    return frame

def load_tracks_csv(tracks_csv: str) -> Dict[int, Dict[int, Tuple]]:
    """
    Returns { frame_idx: { track_id: (x1, y1, x2, y2, occluded) } }
    """
    data = defaultdict(dict)
    with open(tracks_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame    = int(row["frame"])
            tid      = int(row["track_id"])
            x1, y1   = float(row["x1"]), float(row["y1"])
            x2, y2   = float(row["x2"]), float(row["y2"])
            occluded = row["occluded"].strip().lower() == "true"
            data[frame][tid] = (x1, y1, x2, y2, occluded)
    return data


def load_preds_csv(preds_csv: str) -> Dict[int, Dict[int, str]]:
    """
    Returns { frame_idx: { track_id: behavior_label } }
    """
    data = defaultdict(dict)
    with open(preds_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            tid   = int(row["track_id"])
            label = row["behavior_label"].strip()
            data[frame][tid] = label
    return data


def get_sorted_frame_paths(frames_dir: str):
    exts  = (".jpg", ".jpeg", ".png")
    paths = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith(exts)
    ])
    if not paths:
        raise FileNotFoundError(f"No frames found in: {frames_dir}")
    return paths


def run_visualization(
    frames_dir:  str,
    tracks_csv:  str,
    preds_csv:   str,
    output_dir:  str,
    video_path:  Optional[str] = None,
    fps:         float = 25.0,
):
    os.makedirs(output_dir, exist_ok=True)

    frame_paths  = get_sorted_frame_paths(frames_dir)
    track_data   = load_tracks_csv(tracks_csv)
    pred_data    = load_preds_csv(preds_csv)

    print(f"Frames:     {len(frame_paths)}")
    print(f"Track rows: {sum(len(v) for v in track_data.values())}")
    print(f"Pred rows:  {sum(len(v) for v in pred_data.values())}")
    print(f"Output dir: {output_dir}")

    # Running behavior counts for bottom bar
    counts = {"on_box": 0, "at_hole": 0, "in_box": 0}

    # Video writer (optional)
    writer = None
    if video_path:
        first = cv2.imread(frame_paths[0])
        h, w  = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        print(f"Video:      {video_path}")

    for file_idx, frame_path in enumerate(frame_paths):
        frame_idx = file_idx   # 0-based, matches CSV

        img = cv2.imread(frame_path)
        if img is None:
            print(f"  Warning: could not read {frame_path}")
            continue

        # Get track bboxes for this frame
        this_tracks = track_data.get(frame_idx, {})

        # Get predictions for this frame
        # Fall back to last known prediction if model hasn't output this frame yet
        this_preds = pred_data.get(frame_idx, {})

        # Annotate
        img = annotate_frame(img, frame_idx, this_tracks, this_preds, counts)

        # Save annotated frame
        out_name = os.path.basename(frame_path)
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, img)

        if writer:
            writer.write(img)

        if (file_idx + 1) % 100 == 0:
            print(f"  Processed {file_idx+1}/{len(frame_paths)} frames...")

    if writer:
        writer.release()

    print(f"\nDone. {len(frame_paths)} annotated frames saved to: {output_dir}")
    print(f"\nBehavior totals across all frames x tracks:")
    for b, c in counts.items():
        print(f"  {b}: {c}")

    if video_path and os.path.exists(video_path):
        print(f"Video saved: {video_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Draw behavior boxes on test clip frames"
    )
    parser.add_argument("--frames",     required=True,
                        help="Directory of test clip frames (frame_00001.jpg ...)")
    parser.add_argument("--tracks_csv", required=True,
                        help="ByteTrack CSV for this clip (from bytetrack_pipeline.py)")
    parser.add_argument("--preds_csv",  required=True,
                        help="Predictions CSV (from temporal model inference)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save annotated frames")
    parser.add_argument("--video",      default=None,
                        help="Optional: output video path (.mp4)")
    parser.add_argument("--fps",        type=float, default=25.0,
                        help="FPS for output video (default 25)")
    args = parser.parse_args()

    for path, label in [
        (args.frames,     "frames dir"),
        (args.tracks_csv, "tracks CSV"),
        (args.preds_csv,  "predictions CSV"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    run_visualization(
        frames_dir=args.frames,
        tracks_csv=args.tracks_csv,
        preds_csv=args.preds_csv,
        output_dir=args.output_dir,
        video_path=args.video,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
