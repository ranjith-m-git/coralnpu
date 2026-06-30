import os
from ultralytics import YOLO

def run_step1():
    print("=== STEP 1: Exporting PyTorch to SavedModel ===")
    model_path = "/mnt/e/coral_npu/yolov11/yolo11n.pt"
    if not os.path.exists(model_path):
        print(f"Error: Missing {model_path}")
        return
    
    pytorch_model = YOLO(model_path)
    # This creates a folder named 'yolo11n_saved_model'
    pytorch_model.export(format="saved_model", int8=True, data="coco8.yaml", nms=False)
    print("Step 1 Complete! SavedModel folder created.")

if __name__ == "__main__":
    run_step1()
