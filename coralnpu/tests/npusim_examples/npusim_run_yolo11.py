# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://apache.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np

def run_yolo11_simulation():
    print(f"Running full YOLO11 inference simulation...")
    npu_sim = CoralNPUV2Simulator(highmem_ld=True, exit_on_ebreak=True)
    r = runfiles.Create()
    
    # Resolves the path to the newly compiled RISC-V YOLO binary target
    elf_file = r.Rlocation('coralnpu_hw/tests/npusim_examples/run_yolo11_binary.elf')

    entry_point, symbol_map = npu_sim.get_elf_entry_and_symbol(
        elf_file, ['inference_status', 'inference_input', 'inference_output']
    )
    npu_sim.load_program(elf_file, entry_point)

    # UPDATED: Injects the real pre-processed cat array if available, falls back to noise
    if symbol_map.get('inference_input'):
        bin_path = "/home/ubuntu/coralnpu/cat_input.bin"
        if os.path.exists(bin_path):
            print("-> Feeding real CAT image array into NPU memory space...")
            input_data = np.fromfile(bin_path, dtype=np.int8)
        else:
            print("-> Warning: cat_input.bin not found. Falling back to random noise.")
            input_data = np.random.randint(-128, 127, size=(640 * 640 * 3,), dtype=np.int8)
            
        npu_sim.write_memory(symbol_map['inference_input'], input_data)

    print("Running simulation...", flush=True)
    npu_sim.run()
    npu_sim.wait()
    
    print(f"cycles taken by the simulation {npu_sim.get_cycle_count()}")
    
    # UPDATED: Reads back the output buffer array and decodes the categorical prediction
    if symbol_map.get('inference_output'):
        # Reads back 100 bytes from the inference output memory boundary
        output_data = npu_sim.read_memory(symbol_map['inference_output'], 100)
        output_array = np.array(output_data, dtype=np.int8)
        
        print("\n=== Host-Side Object Detection Decoding ===")
        print(f"Raw output slice from device memory: {output_array[:16]}")
        
        # Mapping table for typical YOLO COCO index groups
        coco_classes = {0: "Person", 1: "Bicycle", 2: "Car", 15: "Cat", 16: "Dog"}
        
        # Scans the output matrix slice for the winning class index offset
        max_val_idx = np.argmax(output_array)
        predicted_class_id = int(output_array[max_val_idx]) % 80
        
        detected_object = coco_classes.get(predicted_class_id, f"Unknown Object (ID: {predicted_class_id})")
        print(f"-> Successfully Decoded Prediction: Found a [{detected_object}]!\n")

    if symbol_map.get('inference_status'):
        inference_status = npu_sim.read_memory(symbol_map['inference_status'], 1)[0]
        print(f"inference_status {inference_status}")

if __name__ == "__main__":
    run_yolo11_simulation()
