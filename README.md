# Bird Behavior Tracking

Automated pipeline for detecting and classifying bird behaviors in a fixed-camera aviary using YOLOv8, ByteTrack, and a temporal transformer model.

## Behaviors Detected
| Label | Description |
|-------|-------------|
| `on_box` | Bird sitting on top of nest box |
| `at_hole` | Bird at the hole entrance |
| `in_box` | Bird inside the nest box (occluded) |

---

## Project Structure
```
bird_detection/
├── annotations/
│   ├── train/
│   │   ├── clip1/annotations.xml
│   │   └── clip2/annotations.xml
│   └── val/
│       └── clip1/annotations.xml
├── images/
│   ├── train/
│   │   ├── clip1/frame_00001.jpg ...
│   │   └── clip2/frame_00001.jpg ...
│   └── val/
│       └── clip1/frame_00001.jpg ...
├── train_yolo.py
├── BoxROI.py
├── Bytetrack.py
├── temporal.py
└── Visualize.py

```

---

## Requirements
```bash
pip install ultralytics timm torch torchvision supervision scikit-learn numpy pillow opencv-python
```

---

## Step-by-Step Pipeline

### Step 1 : Annotate Hole ROIs (run once)

Mark the nest box hole entrance positions on a reference frame. Since the camera is fixed this only needs to be done once per camera angle for Each Enclosure.
```bash
python BoxROI.py \
    --frame images/train/clip1/frame_00001.jpg \
    --output hole_rois.json
```
---
### Step 2 : Train YOLO model
```bash
python train_yolo.py \
    --epochs 60 \
    --batch 8 \
    --imgsz 832
```
- Model: YOLOv8s trained from scratch
- Best weights saved automatically to `runs/bird_detector/weights/best.pt`
- Use `best.pt` as the `--model` argument in all subsequent steps

### Step 3 : Train the Temporal Model

Annotations must be in CVAT XML format under `annotations/train/` and `annotations/val/`. Frames must be in the matching `images/train/` and `images/val/` directories.
```bash
python temporal.py --mode train \
    --epochs 25 \
    --batch_size 8 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --window_size 16 \
    --stride 8 \
    --unfreeze_blocks 2 \
    --focal_gamma 2.0
```
Best model saved automatically to `best_temporal_model_v4.pt` based on validation macro F1.

### Step 4 : Run ByteTrack on Test Clip
```bash
python Bytetrack.py \
    --model  best.pt \
    --rois   hole_rois.json \
    --frames images/test/clip1 \
    --export tracks/clip1.csv
```

For multiple clips at once:
```bash
python Bytetrack.py \
    --model     best.pt \
    --rois      hole_rois.json \
    --frames    images/test \
    --export    tracks \
    --all_clips
```
---

### Step 5 : Run Behavior Inference
```bash
python temporal.py --mode infer \
    --tracks_csv  tracks/clip1.csv \
    --frames_dir  images/test/clip1 \
    --weights     best_temporal_model_v4.pt \
    --output      predictions/clip_01_behaviors.csv \
    --window_size 16
```
---

### Step 6 : Visualize Results
```bash
python Visualize.py \
    --frames      images/test/clip1 \
    --tracks_csv  tracks/clip1.csv \
    --preds_csv   predictions/clip_01_behaviors.csv \
    --output_dir  visualizations/clip_01 \
    --video       visualizations/clip_01.mp4 \
    --fps         5.0
```

## Training Notes

- Extract frames at 5fps from original video before training
- Annotate using [CVAT](https://cvat.org) and export as CVAT XML format
- Use `--stride 8` with small datasets to avoid overlapping windows inflating metrics
- Best checkpoint is saved automatically — no need to run to final epoch
---

## Acknowledgements

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [timm](https://github.com/huggingface/pytorch-image-models)
