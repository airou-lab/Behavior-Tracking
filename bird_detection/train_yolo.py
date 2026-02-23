import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=832)
    args = parser.parse_args()

    model = YOLO("yolov8s.pt")

    model.train(
        data="data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=8,
        project="runs",
        name="bird_detector"
    )

if __name__ == "__main__":
    main()

