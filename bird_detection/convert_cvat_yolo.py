import os
import xml.etree.ElementTree as ET

# ===== CONFIG =====
ANNOTATIONS_BASE = "annotations/test"
IMAGES_BASE = "images/test"
LABELS_BASE = "labels/test"

IMG_WIDTH = 1920
IMG_HEIGHT = 1080
# ===================

for clip_name in os.listdir(ANNOTATIONS_BASE):

    clip_annotation_path = os.path.join(
        ANNOTATIONS_BASE, clip_name, "annotations.xml"
    )

    if not os.path.exists(clip_annotation_path):
        print(f"Skipping {clip_name}, no annotations.xml found")
        continue

    print(f"Processing {clip_name}")

    labels_output = os.path.join(LABELS_BASE, clip_name)
    os.makedirs(labels_output, exist_ok=True)

    tree = ET.parse(clip_annotation_path)
    root = tree.getroot()

    for track in root.findall("track"):
        for box in track.findall("box"):

            frame = int(box.get("frame")) + 1
            outside = int(box.get("outside"))

            if outside == 1:
                continue

            behavior = None
            for attr in box.findall("attribute"):
                if attr.get("name") == "Behavior":
                    behavior = attr.text

            # Skip invisible birds
            if behavior == "in_box":
                continue

            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            x_center = ((xtl + xbr) / 2) / IMG_WIDTH
            y_center = ((ytl + ybr) / 2) / IMG_HEIGHT
            width = (xbr - xtl) / IMG_WIDTH
            height = (ybr - ytl) / IMG_HEIGHT

            label_line = f"0 {x_center} {y_center} {width} {height}\n"

            frame_name = f"frame_{frame:05d}.txt"
            label_path = os.path.join(labels_output, frame_name)

            with open(label_path, "a") as f:
                f.write(label_line)

print("All clips converted successfully.")
