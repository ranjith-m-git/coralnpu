import cv2
import numpy as np
import tensorflow as tf

# 1. Load your local image file 
raw_img = cv2.imread("images/cat.jpg")
if raw_img is None:
    raise FileNotFoundError("Could not open or find 'images/cat.jpg'")

h_orig, w_orig, _ = raw_img.shape

# 2. Initialize the Interpreter
interpreter = tf.lite.Interpreter(model_path="yolo26n_full_integer_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Preprocess the image to 640x640 
input_img = cv2.resize(raw_img, (640, 640))
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

# Normalize image from 0-255 integers to 0.0 - 1.0 floating point numbers
input_img = input_img.astype(np.float32) / 255.0

# Extract input quantization rules
in_quant = input_details[0]['quantization_parameters']
in_scale = in_quant['scales'][0] if len(in_quant['scales']) > 0 else 1.0
in_zero_point = in_quant['zero_points'][0] if len(in_quant['zero_points']) > 0 else 0

# Convert FLOAT32 down into INT8 properly without overflowing
input_img = np.round(input_img / in_scale + in_zero_point).astype(np.int8)
input_img = np.expand_dims(input_img, axis=0)

# 4. Run Inference
interpreter.set_tensor(input_details[0]['index'], input_img)
interpreter.invoke()

# 5. Extract and De-quantize Output
raw_output_data = interpreter.get_tensor(output_details[0]['index'])
raw_output_data = np.squeeze(raw_output_data)  # Drops batch dimension safely

# Get output de-quantization attributes from list index 0
out_quant = output_details[0]['quantization_parameters']
out_scale = out_quant['scales'][0] if len(out_quant['scales']) > 0 else 1.0
out_zero_point = out_quant['zero_points'][0] if len(out_quant['zero_points']) > 0 else 0

# Turn raw INT8 matrix back into FLOAT32 real numbers
output_data = (raw_output_data.astype(np.float32) - out_zero_point) * out_scale

# --- MATRIX AUTO-CORRECTION ---
print(f"DEBUG 1: Raw output matrix shape from model is: {output_data.shape}")
if output_data.shape == (8400, 84):
    print("DEBUG 2: Matrix is flipped (8400, 84). Transposing to standard format (84, 8400)...")
    output_data = output_data.T

# Helper function to compute Sigmoid activation safely
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

# 6. Parse the Corrected Matrix
boxes = []
confidences = []
class_ids = []

# Complete dictionary of standard classes for matching context
coco_classes = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 15: "cat", 16: "dog"} 

# Set confidence threshold ultra-low to capture quantized detections
CONF_THRESH = 0.01  
NMS_THRESH = 0.45

for i in range(8400):
    # Slice exact object bounding scores
    raw_scores = output_data[4:, i]
    prob_scores = sigmoid(raw_scores)  
    
    class_id = np.argmax(prob_scores)
    confidence = prob_scores[class_id]
    
    if confidence > CONF_THRESH:
        # Extract bounding box center metrics
        xc, yc, w, h = output_data[0:4, i]
        
        # Scale back coordinates directly from 640x640 frame boundaries
        x1 = int((xc - w / 2) / 640 * w_orig)
        y1 = int((yc - h / 2) / 640 * h_orig)
        box_w = int(w / 640 * w_orig)
        box_h = int(h / 640 * h_orig)
        
        # Guard against zero dimension artifacts
        if box_w > 1 and box_h > 1:
            boxes.append([x1, y1, box_w, box_h])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))

# 7. Non-Maximum Suppression (NMS) to clean up remaining duplicates
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=CONF_THRESH, nms_threshold=NMS_THRESH)

print("\n--- DETECTED OBJECTS ---")
if len(indices) > 0:
    for idx in indices.flatten():
        c_id = class_ids[idx]
        name = coco_classes.get(c_id, f"Class {c_id}")
        conf = confidences[idx] * 100
        print(f"Found: {name} ({conf:.1f}% Confidence) at Box Location: {boxes[idx]}")
        
        # 8. VISUAL OVERLAY: Draw boxes onto the original image
        x, y, w, h = boxes[idx]
        cv2.rectangle(raw_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        label = f"{name}: {conf:.1f}%"
        cv2.putText(raw_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the output visualization image
    output_path = "output_result.jpg"
    cv2.imwrite(output_path, raw_img)
    print(f"\nVisual validation saved successfully to: {output_path}")
else:
    print("No objects detected above threshold. Your model layers require a larger calibration run.")
