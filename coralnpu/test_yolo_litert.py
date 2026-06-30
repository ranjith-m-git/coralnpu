import numpy as np
import tensorflow as tf

# Load the model
interpreter = tf.lite.Interpreter(model_path="yolo11n_int8.tflite")
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_type = input_details[0]['dtype']

print(f"Model successfully loaded!")
print(f"Expected Input Type: {input_type}")
print(f"Expected Input Shape: {input_shape}")

# Create the correct data type based on what the model actually expects
if input_type == np.float32:
    # Create float data between 0.0 and 1.0 (Standard for Float32 YOLO)
    input_data = np.random.rand(*input_shape).astype(np.float32)
else:
    # Create signed int8 data between -128 and 127
    input_data = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)

# Run Inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Extract and print all outputs
print("\n--- Inference Successful! ---")
for i, output in enumerate(output_details):
    output_data = interpreter.get_tensor(output['index'])
    print(f"Output {i} Name: {output['name']}")
    print(f"Output {i} Shape: {output_data.shape}")
