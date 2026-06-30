import os
import cv2
import numpy as np

# 1. Define the function using clean variable nicknames (arguments)
def process_image(input_path, output_path):
    print(f"Loading image from: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not find image at {input_path}")
        return
        
    # Resize to the exact shape expected by your YOLO11 model (640x640)
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Quantize to signed 8-bit integers (INT8: -128 to 127)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_int8 = ((img_float - 0.5) * 255.0).astype(np.int8)
    
    # Save the raw pixel array byte stream
    img_int8.tofile(output_path)
    print(f"Success! Raw input array file generated at: {output_path}")

if __name__ == "__main__":
    # 2. Put your real file path strings down here inside the execution block
    # Make sure you have a real image file named 'cat.jpg' sitting inside an 'images' folder!
    input_file = "images/cat.jpg" 
    output_file = "cat_input.bin"  # This file will generate in your active folder
    
    process_image(input_file, output_file)
