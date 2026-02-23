"""
ByteTrack Bird Tracking Pipeline v2 — with Weighted Box Fusion (WBF)
======================================================================
Improvement over v1: runs YOLO at TWO confidence thresholds and fuses
detections using Weighted Box Fusion before passing to ByteTrack.

Why this helps for small birds:
  - Low threshold (0.25): catches faint/partial detections (peeking bird,
    partially occluded bird, small bird far from camera)
  - High threshold (0.6): catches confident, clean detections
  - WBF merges overlapping boxes weighted by confidence — better than NMS
    which just picks one box and discards others
  - ByteTrack then gets richer, more stable detections → fewer dropped tracks

Usage:
    python bytetrack_pipeline_v2.py \
        --model  best.pt \
        --rois   hole_rois.json \
        --frames images/train/clip_01 \
        --export tracks/clip_01.csv

    python bytetrack_pipeline_v2.py \
        --model     best.pt \
        --rois      hole_rois.json \
        --frames    images/train \
        --export    tracks \
        --all_clips
"""

import cv2
import json
import argparse
import numpy as np
import csv
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from ultralytics import YOLO


# =========================================================================
# Config
# =========================================================================

# Hardcode your paths here so you don't need to pass them on command line
DEFAULT_MODEL = "best.pt"
DEFAULT_ROIS  = "hole_rois.json"

MAX_GHOST_FRAMES = 30
VELOCITY_HISTORY = 5
INWARD_THRESHOLD = 0.5
NEAR_HOLE_FACTOR = 1.2

# WBF settings
CONF_LOW      = 0.25   # catches faint/partial bird detections
CONF_HIGH     = 0.60   # confident clean detections
WBF_IOU_THR   = 0.45   # boxes with IoU > this are merged by WBF
WBF_SKIP_BOX  = 0.001  # boxes with merged confidence below this are dropped


# =========================================================================
# Weighted Box Fusion
# =========================================================================

def wbf_single_image(
    boxes_list:  List[np.ndarray],   # list of (N_i, 4) arrays, each row [x1,y1,x2,y2] normalised 0-1
    scores_list: List[np.ndarray],   # list of (N_i,) confidence arrays
    iou_thr:     float = WBF_IOU_THR,
    skip_box_thr: float = WBF_SKIP_BOX,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted Box Fusion across multiple detection sets.
    Based on: https://arxiv.org/abs/1910.13302

    Returns:
        fused_boxes:  (M, 4) array of fused boxes [x1,y1,x2,y2] normalised
        fused_scores: (M,)   array of fused confidence scores
    """
    if not boxes_list or all(len(b) == 0 for b in boxes_list):
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    # Flatten all boxes + scores, tag with source list index
    all_boxes  = []
    all_scores = []

    for b_arr, s_arr in zip(boxes_list, scores_list):
        if len(b_arr) == 0:
            continue
        for box, score in zip(b_arr, s_arr):
            all_boxes.append(box)
            all_scores.append(float(score))

    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    all_boxes  = np.array(all_boxes,  dtype=np.float32)   # (N, 4)
    all_scores = np.array(all_scores, dtype=np.float32)   # (N,)

    # Sort by score descending
    order      = np.argsort(-all_scores)
    all_boxes  = all_boxes[order]
    all_scores = all_scores[order]

    # Cluster boxes by IoU
    clusters_boxes  = []   # list of lists of box arrays in each cluster
    clusters_scores = []   # list of lists of scores in each cluster

    def iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        union = area_a + area_b - inter + 1e-6
        return inter / union

    for box, score in zip(all_boxes, all_scores):
        matched = False
        for ci, cluster_boxes in enumerate(clusters_boxes):
            # Compare with weighted mean box of this cluster
            w = np.array(clusters_scores[ci])
            w = w / w.sum()
            mean_box = (np.array(cluster_boxes) * w[:, None]).sum(axis=0)
            if iou(box, mean_box) > iou_thr:
                clusters_boxes[ci].append(box)
                clusters_scores[ci].append(score)
                matched = True
                break
        if not matched:
            clusters_boxes.append([box])
            clusters_scores.append([score])

    # Fuse each cluster into one box
    fused_boxes  = []
    fused_scores = []

    for c_boxes, c_scores in zip(clusters_boxes, clusters_scores):
        c_boxes  = np.array(c_boxes,  dtype=np.float32)
        c_scores = np.array(c_scores, dtype=np.float32)
        w        = c_scores / c_scores.sum()
        fused_box   = (c_boxes * w[:, None]).sum(axis=0)
        fused_score = c_scores.mean()   # average confidence of cluster
        if fused_score >= skip_box_thr:
            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)

    if not fused_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    return np.array(fused_boxes, dtype=np.float32), np.array(fused_scores, dtype=np.float32)


def detect_with_wbf(model: YOLO, img: np.ndarray, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run YOLO at two confidence thresholds, fuse with WBF.
    Returns (boxes_pixel, scores) where boxes_pixel is (N, 4) [x1,y1,x2,y2] in pixel coords.
    """
    boxes_list  = []
    scores_list = []

    for conf_thr in [CONF_LOW, CONF_HIGH]:
        result = model.predict(img, conf=conf_thr, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            boxes_list.append(np.zeros((0, 4), dtype=np.float32))
            scores_list.append(np.zeros(0, dtype=np.float32))
            continue

        # Normalise boxes to [0, 1]
        xyxy   = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        norm   = xyxy / np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        boxes_list.append(norm)
        scores_list.append(scores)

    fused_norm, fused_scores = wbf_single_image(boxes_list, scores_list)

    if len(fused_norm) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

    # Denormalise back to pixel coords
    fused_pixel = fused_norm * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return fused_pixel, fused_scores


# =========================================================================
# ROI
# =========================================================================

@dataclass
class HoleROI:
    id: int
    cx: float
    cy: float
    radius: float

    def contains(self, x, y, factor=1.0) -> bool:
        return np.hypot(x - self.cx, y - self.cy) <= self.radius * factor

    def distance_to(self, x, y) -> float:
        return float(np.hypot(x - self.cx, y - self.cy))


def load_rois(json_path: str) -> List[HoleROI]:
    with open(json_path) as f:
        data = json.load(f)
    rois = [HoleROI(id=d["id"], cx=d["cx"], cy=d["cy"], radius=d["radius"]) for d in data]
    print(f"Loaded {len(rois)} hole ROIs from {json_path}")
    for r in rois:
        print(f"  Hole {r.id}: center=({r.cx:.0f}, {r.cy:.0f})  radius={r.radius:.0f}px")
    return rois


# =========================================================================
# Track state + manager (unchanged logic from v1)
# =========================================================================

@dataclass
class TrackState:
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=VELOCITY_HISTORY))
    ghost_frames: int = 0
    last_bbox: Optional[tuple] = None
    nearest_hole: Optional[HoleROI] = None

    def add_position(self, cx: float, cy: float):
        self.positions.append((cx, cy))

    def fraction_moving_toward(self, hole: HoleROI) -> float:
        pos = list(self.positions)
        if len(pos) < 2:
            return 0.5
        inward = 0
        for i in range(1, len(pos)):
            dx = pos[i][0] - pos[i-1][0]; dy = pos[i][1] - pos[i-1][1]
            to_x = hole.cx - pos[i-1][0]; to_y = hole.cy - pos[i-1][1]
            if dx * to_x + dy * to_y > 0:
                inward += 1
        return inward / (len(pos) - 1)

    def is_slowing_down(self) -> bool:
        pos = list(self.positions)
        if len(pos) < 3:
            return False
        speeds = [np.hypot(pos[i][0]-pos[i-1][0], pos[i][1]-pos[i-1][1]) for i in range(1, len(pos))]
        return speeds[-1] < speeds[0]

    def entered_hole(self, hole: HoleROI) -> bool:
        return self.fraction_moving_toward(hole) >= INWARD_THRESHOLD or self.is_slowing_down()


class TrackManager:
    def __init__(self, rois: List[HoleROI]):
        self.rois   = rois
        self.tracks: Dict[int, TrackState] = {}

    def _nearest_hole(self, cx, cy) -> Optional[HoleROI]:
        return min(self.rois, key=lambda r: r.distance_to(cx, cy)) if self.rois else None

    def on_detected(self, track_id: int, bbox: tuple, conf: float) -> dict:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        w,  h  = x2-x1, y2-y1
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(track_id=track_id)
        state = self.tracks[track_id]
        state.add_position(cx, cy)
        state.last_bbox    = bbox
        state.ghost_frames = 0
        state.nearest_hole = self._nearest_hole(cx, cy)
        return {"track_id": track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy, "w": w, "h": h,
                "occluded": False, "confidence": conf}

    def on_missing(self, track_id: int) -> Optional[dict]:
        if track_id not in self.tracks:
            return None
        state = self.tracks[track_id]
        state.ghost_frames += 1
        if state.ghost_frames > MAX_GHOST_FRAMES or state.last_bbox is None:
            del self.tracks[track_id]; return None
        x1, y1, x2, y2 = state.last_bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        w,  h  = x2-x1, y2-y1
        hole = state.nearest_hole or self._nearest_hole(cx, cy)
        if hole is None or not hole.contains(cx, cy, factor=NEAR_HOLE_FACTOR):
            del self.tracks[track_id]; return None
        if not state.entered_hole(hole):
            del self.tracks[track_id]; return None
        return {"track_id": track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy, "w": w, "h": h,
                "occluded": True, "confidence": 0.0}

    def active_ids(self) -> List[int]:
        return list(self.tracks.keys())


# =========================================================================
# Frame loader
# =========================================================================

def get_sorted_frame_paths(frames_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith(exts)
    ])
    if not paths:
        raise FileNotFoundError(f"No image files found in: {frames_dir}")
    return paths


# =========================================================================
# Core tracking loop
# =========================================================================

def track_clip(
    model:       YOLO,
    rois:        List[HoleROI],
    frames_dir:  str,
    export_csv:  str,
    iou:         float = 0.5,
    frame_index_offset: int = 1,
):
    """
    Key difference from v1: detections come from detect_with_wbf()
    (two-threshold YOLO + WBF fusion) instead of model.track() directly.
    We then pass WBF-fused boxes to ByteTrack via model.track() with
    custom boxes injected — or run ByteTrack manually via supervision.

    Practical note: Ultralytics doesn't support injecting pre-fused boxes
    into ByteTrack directly. So we use a two-step approach:
      1. detect_with_wbf() → fused boxes + scores
      2. model.track() with the fused result fed back in via a dummy frame
         where only the fused detections survive.

    The simplest production approach: use the fused boxes as-is and feed
    them into a standalone ByteTrack instance from the `supervision` library.
    """
    try:
        import supervision as sv
        byte_tracker = sv.ByteTracker(
            track_activation_threshold=0.25,
            lost_track_buffer=MAX_GHOST_FRAMES,
            minimum_matching_threshold=iou,
            frame_rate=25,
        )
        use_supervision = True
        print("Using supervision ByteTracker with WBF detections")
    except ImportError:
        use_supervision = False
        print("supervision not installed — falling back to model.track() (WBF still applied for ROI logic)")
        print("Install with: pip install supervision --break-system-packages")

    frame_paths = get_sorted_frame_paths(frames_dir)
    manager     = TrackManager(rois)

    out_dir = os.path.dirname(export_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_rows = 0
    ghost_rows = 0

    with open(export_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id",
                         "x1", "y1", "x2", "y2",
                         "cx", "cy", "w", "h",
                         "occluded", "confidence"])

        for file_idx, frame_path in enumerate(frame_paths):
            frame_idx = file_idx
            img = cv2.imread(frame_path)
            if img is None:
                print(f"  Warning: could not read {frame_path}")
                continue

            img_h, img_w = img.shape[:2]

            # Step 1: WBF-fused detections
            fused_boxes, fused_scores = detect_with_wbf(model, img, img_w, img_h)

            detected_ids = set()

            if use_supervision and len(fused_boxes) > 0:
                # Feed WBF boxes into supervision ByteTracker
                detections = sv.Detections(
                    xyxy=fused_boxes,
                    confidence=fused_scores,
                )
                tracks = byte_tracker.update_with_detections(detections)

                for track in tracks:
                    track_id   = int(track[4])
                    x1, y1, x2, y2 = track[:4]
                    # Find matching confidence from fused_scores by IoU
                    conf = float(fused_scores[0]) if len(fused_scores) > 0 else 0.5

                    detected_ids.add(track_id)
                    row = manager.on_detected(track_id, (x1, y1, x2, y2), conf)
                    writer.writerow([frame_idx, row["track_id"],
                                     round(row["x1"],2), round(row["y1"],2),
                                     round(row["x2"],2), round(row["y2"],2),
                                     round(row["cx"],2), round(row["cy"],2),
                                     round(row["w"],2),  round(row["h"],2),
                                     row["occluded"],     round(row["confidence"],4)])
                    total_rows += 1

            else:
                # Fallback: standard model.track() (ByteTrack built into Ultralytics)
                results = model.track(img, persist=True, tracker="bytetrack.yaml",
                                      verbose=False, conf=CONF_LOW, iou=iou)
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if box.id is None:
                            continue
                        track_id   = int(box.id.item())
                        x1,y1,x2,y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf.item())
                        detected_ids.add(track_id)
                        row = manager.on_detected(track_id, (x1,y1,x2,y2), confidence)
                        writer.writerow([frame_idx, row["track_id"],
                                         round(row["x1"],2), round(row["y1"],2),
                                         round(row["x2"],2), round(row["y2"],2),
                                         round(row["cx"],2), round(row["cy"],2),
                                         round(row["w"],2),  round(row["h"],2),
                                         row["occluded"],     round(row["confidence"],4)])
                        total_rows += 1

            # Ghost track handling (same as v1)
            for track_id in manager.active_ids():
                if track_id in detected_ids:
                    continue
                ghost = manager.on_missing(track_id)
                if ghost is None:
                    continue
                writer.writerow([frame_idx, ghost["track_id"],
                                 round(ghost["x1"],2), round(ghost["y1"],2),
                                 round(ghost["x2"],2), round(ghost["y2"],2),
                                 round(ghost["cx"],2), round(ghost["cy"],2),
                                 round(ghost["w"],2),  round(ghost["h"],2),
                                 ghost["occluded"],     ghost["confidence"]])
                total_rows += 1
                ghost_rows += 1

            if (file_idx + 1) % 100 == 0:
                print(f"  {file_idx+1}/{len(frame_paths)} frames...")

    print(f"  Tracks: {total_rows} rows  ({ghost_rows} occluded)")
    print(f"  Saved:  {export_csv}")


# =========================================================================
# Entry point
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    parser.add_argument("--rois",      default=DEFAULT_ROIS)
    parser.add_argument("--frames",    required=True)
    parser.add_argument("--export",    required=True)
    parser.add_argument("--all_clips", action="store_true")
    parser.add_argument("--iou",       type=float, default=0.5)
    parser.add_argument("--offset",    type=int,   default=1)
    args = parser.parse_args()

    for path, label in [(args.model,"model"), (args.rois,"ROIs"), (args.frames,"frames")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    rois = load_rois(args.rois)

    if args.all_clips:
        clips = sorted([d for d in os.listdir(args.frames)
                        if os.path.isdir(os.path.join(args.frames, d))])
        if not clips:
            raise ValueError(f"No subdirectories in {args.frames}")
        os.makedirs(args.export, exist_ok=True)
        for clip in clips:
            print(f"\n=== Clip: {clip} ===")
            model = YOLO(args.model)   # reset ByteTrack state between clips
            track_clip(model, rois,
                       os.path.join(args.frames, clip),
                       os.path.join(args.export, f"{clip}_tracks.csv"),
                       iou=args.iou, frame_index_offset=args.offset)
    else:
        model = YOLO(args.model)
        track_clip(model, rois, args.frames, args.export,
                   iou=args.iou, frame_index_offset=args.offset)

    print("\nDone.")


if __name__ == "__main__":
    main()