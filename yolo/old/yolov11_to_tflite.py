import os
from ultralytics import YOLO

# 1. Define the absolute paths to avoid directory relative issues
WORKING_DIR = "/mnt/e/coral_npu/yolov11"
MODEL_PATH = os.path.join(WORKING_DIR, "yolo11n.pt")

print(f"[INFO] Loading model from: {MODEL_PATH}")

# 2. Instantiate the YOLOv11 model structure using the file in that directory
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    raise FileNotFoundError(f"Could not find yolo11n.pt at {MODEL_PATH}. Check file location!")

print("[INFO] Initiating TFLite INT8 Quantization Export...")

# 3. Export to full INT8 format using built-in calibration dataset
# Ultralytics automatically handles dataset calibration using 'coco8.yaml'
output_path = model.export(
    format="tflite",
    int8=True,
    data="coco8.yaml"
)

print(f"\n[SUCCESS] Model successfully exported to: {output_path}")
