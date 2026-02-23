"""
ROI Annotation Tool — Bird Box Entrance Marker
===============================================
Usage:
    python annotate_rois.py --frame path/to/frame_00001.jpg --output hole_rois.json

Controls:
    Left click          → place a hole center marker
    Right click         → remove the nearest marker
    Scroll wheel        → resize the ROI radius for the NEXT click
    'r'                 → reset all markers
    's' or Enter        → save and exit
    'q' or Escape       → quit without saving
    'z'                 → undo last marker

What gets saved (hole_rois.json):
    A list of dicts, one per hole:
    {
        "id": 0,
        "cx": 412,          # pixel x of hole center
        "cy": 183,          # pixel y of hole center
        "radius": 40        # ROI radius in pixels — bird bbox center must be within this
    }
"""

import cv2
import json
import argparse
import numpy as np
import os

# -------------------------
# State
# -------------------------
markers = []          # list of (cx, cy, radius)
current_radius = 40   # default ROI radius in pixels
COLORS = [
    (0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (0, 128, 255), (255, 255, 0), (0, 255, 128),
]

def get_color(i):
    return COLORS[i % len(COLORS)]

def draw_frame(base_img):
    img = base_img.copy()
    h, w = img.shape[:2]

    for i, (cx, cy, r) in enumerate(markers):
        color = get_color(i)
        cv2.circle(img, (cx, cy), r, color, 2)
        cv2.circle(img, (cx, cy), 4, color, -1)
        cv2.putText(img, f"Box {i}", (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # HUD
    hud_lines = [
        f"Markers: {len(markers)}   Radius: {current_radius}px",
        "Left click=add   Right click=remove nearest",
        "Scroll=resize radius   Z=undo   R=reset",
        "S / Enter = SAVE & EXIT   Q / Esc = QUIT",
    ]
    for j, line in enumerate(hud_lines):
        y = 22 + j * 22
        cv2.rectangle(img, (0, y - 16), (500, y + 5), (0, 0, 0), -1)
        cv2.putText(img, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    # Preview circle at current radius
    cx_prev = w - current_radius - 20
    cy_prev = 30
    cv2.circle(img, (cx_prev, cy_prev), current_radius, (200, 200, 200), 2)
    cv2.putText(img, "next ROI", (cx_prev - 25, cy_prev + current_radius + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return img


def nearest_marker_idx(x, y):
    if not markers:
        return None
    dists = [np.hypot(x - cx, y - cy) for (cx, cy, _) in markers]
    idx = int(np.argmin(dists))
    return idx if dists[idx] < 80 else None


def mouse_callback(event, x, y, flags, param):
    global current_radius, markers

    if event == cv2.EVENT_LBUTTONDOWN:
        markers.append((x, y, current_radius))
        print(f"  Added hole {len(markers)-1}: center=({x},{y}) radius={current_radius}px")

    elif event == cv2.EVENT_RBUTTONDOWN:
        idx = nearest_marker_idx(x, y)
        if idx is not None:
            removed = markers.pop(idx)
            print(f"  Removed marker at ({removed[0]},{removed[1]})")

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            current_radius = min(current_radius + 5, 200)
        else:
            current_radius = max(current_radius - 5, 10)


def main():
    global current_radius, markers

    parser = argparse.ArgumentParser()
    parser.add_argument("--frame",  required=True, help="Path to a reference frame (jpg/png)")
    parser.add_argument("--output", default="hole_rois.json", help="Output JSON path")
    parser.add_argument("--load",   default=None,  help="Load existing JSON to edit it")
    args = parser.parse_args()

    if not os.path.exists(args.frame):
        raise FileNotFoundError(f"Frame not found: {args.frame}")

    base_img = cv2.imread(args.frame)
    if base_img is None:
        raise ValueError(f"Could not load image: {args.frame}")

    h, w = base_img.shape[:2]
    print(f"Image size: {w}x{h}")
    print(f"Output: {args.output}")

    # Load existing ROIs if editing
    if args.load and os.path.exists(args.load):
        with open(args.load) as f:
            existing = json.load(f)
        markers = [(d["cx"], d["cy"], d["radius"]) for d in existing]
        print(f"Loaded {len(markers)} existing markers from {args.load}")

    cv2.namedWindow("ROI Annotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Annotator", min(w, 1400), min(h, 900))
    cv2.setMouseCallback("ROI Annotator", mouse_callback)

    print("\n=== ROI Annotator Ready ===")
    print("Click each nest box HOLE ENTRANCE center.")
    print("Use scroll wheel to adjust the ROI radius before clicking.\n")

    saved = False
    while True:
        display = draw_frame(base_img)
        cv2.imshow("ROI Annotator", display)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('s'), 13):  # S or Enter
            rois = [
                {"id": i, "cx": cx, "cy": cy, "radius": r}
                for i, (cx, cy, r) in enumerate(markers)
            ]
            with open(args.output, "w") as f:
                json.dump(rois, f, indent=2)
            print(f"\nSaved {len(rois)} ROIs to: {args.output}")
            for roi in rois:
                print(f"  Box {roi['id']}: center=({roi['cx']},{roi['cy']}) radius={roi['radius']}px")
            saved = True
            break

        elif key in (ord('q'), 27):  # Q or Escape
            print("Quit without saving.")
            break

        elif key == ord('z'):  # Undo
            if markers:
                removed = markers.pop()
                print(f"  Undo: removed ({removed[0]},{removed[1]})")

        elif key == ord('r'):  # Reset
            markers.clear()
            print("  Reset all markers.")

    cv2.destroyAllWindows()

    if saved:
        print("\nNext step:")
        print(f"  python bytetrack_pipeline.py --rois {args.output} --source your_video.mp4")


if __name__ == "__main__":
    main()