import numpy as np
import tensorflow as tf

def run_step2():
    print("=== STEP 2: Running Low-Level Strict Full-Integer Quantization ===")
    saved_model_dir = "/mnt/e/coral_npu/yolov11/yolo11n_saved_model"
    
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_data_gen():
        for _ in range(10):
            dummy_frame = np.random.rand(1, 640, 640, 3).astype(np.float32)
            yield [dummy_frame]
            
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8   
    converter.inference_output_type = tf.int8  
    
    print("Converting model tensors to pure INT8 format...")
    pure_tflite_model = converter.convert()
    
    output_filename = "/mnt/e/coral_npu/yolov11/yolo11n_pure_int8.tflite"
    with open(output_filename, "wb") as f:
        f.write(pure_tflite_model)
        
    print(f"SUCCESS! Pure integer model saved as: {output_filename}")
    
    # Verification
    verifier = tf.lite.Interpreter(model_path=output_filename)
    verifier.allocate_tensors()
    print("Verified Input Data Type:", verifier.get_input_details()[0]['dtype'])
    print("Verified Output Data Type:", verifier.get_output_details()[0]['dtype'])

if __name__ == "__main__":
    run_step2()
