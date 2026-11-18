
from pathlib import Path
from ultralytics import YOLO

# Exact file locations
model_path = Path(r"D:\lunaa\model\best.pt")
image_path = Path(r"D:\lunaa\image\image.png")

# Check model
if not model_path.exists():
    print("ERROR: best.pt is missing!")
    raise SystemExit

# Check image
if not image_path.exists():
    print("ERROR: image.png is missing!")
    raise SystemExit

# Load trained model
model = YOLO(str(model_path))

# Detect lunar boulders
results = model.predict(
    source=str(image_path),
    conf=0.25,
    imgsz=640,
    save=True
)

print("Lunar boulder detection completed!")
