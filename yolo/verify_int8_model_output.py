import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="yolo26n_saved_model/yolo26n_full_integer_quant.tflite")
#interpreter = tf.lite.Interpreter(model_path="yolo26n_saved_model/yolo26n_int8.tflite")
interpreter.allocate_tensors()

# Inspecting the Entrance Gate
print("--- ENTRANCE GATE (INPUT) ---")
print(interpreter.get_input_details()[0])

# Inspecting the Exit Gate
print("\n--- EXIT GATE (OUTPUT) ---")
print(interpreter.get_output_details()[0])
