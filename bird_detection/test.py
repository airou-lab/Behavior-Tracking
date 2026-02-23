from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/runs/bird_detector4/weights/best.pt")

    # ----------- 1️⃣ Evaluate on TEST set -----------
    metrics = model.val(
        data="bird_detection/data.yaml",
        split="test",
        imgsz=832,
        batch=8,
        device="cpu",
        save_json=True,
        plots=True
    )

    print("\nEvaluation on TEST set complete.\n")
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

    # ----------- 2️⃣ Save images with predicted boxes -----------
    model.predict(
        source="bird_detection/images/test/clip7_06082024",   # your test images folder
        imgsz=832,
        conf=0.05,      # lower if you want higher recall
        device="cpu",
        save=True,      # THIS saves boxed images
        save_txt=False,
        save_conf=True  # writes confidence values
    )

    print("\nPrediction images saved.")

if __name__ == "__main__":
    main()
