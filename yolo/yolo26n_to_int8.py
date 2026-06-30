from ultralytics import YOLO

# 1. Load YOLO26 model and export to Edge TPU (INT8)
model = YOLO("yolo26n.pt")
model.export(format="edgetpu") 

# 2. Load the exported Edge TPU model
# Note: The export process creates a file ending in '_full_integer_quant_edgetpu.tflite'
# Load the standard INT8 CPU model
model_path = "yolo26n_saved_model/yolo26n_full_integer_quant.tflite"
model = YOLO(model_path, task="detect")

# Run inference and explicitly tell it to save the output image
#results = model("images/car.jpg", save=True)
results = model("images/Busy_Street_in_Causeway_Bay.jpg", save=True)
