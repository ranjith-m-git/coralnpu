from ultralytics import YOLO

# Load the standard INT8 CPU model
model_path = "yolo26n_saved_model/yolo26n_full_integer_quant.tflite"
model = YOLO(model_path, task="detect")

# Run inference and explicitly tell it to save the output image
results = model("images/car.jpg", save=True)
