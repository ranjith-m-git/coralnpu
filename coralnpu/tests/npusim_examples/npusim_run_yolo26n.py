# Copyright 2026 Google LLC
from bazel_tools.tools.python.runfiles import runfiles
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
import numpy as np

# Standard COCO dataset classes (YOLO defaults)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
]

def non_max_suppression_raw(boxes, scores, iou_threshold=0.45):
    """Filters out overlapping boxes using raw integer score matrices."""
    if len(boxes) == 0: return []
    x1, y1 = boxes[:, 0] - boxes[:, 2]/2, boxes[:, 1] - boxes[:, 3]/2
    x2, y2 = boxes[:, 0] + boxes[:, 2]/2, boxes[:, 1] + boxes[:, 3]/2
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def run_yolo():
    print("Running YOLO26n...")
    npu_sim = CoralNPUV2Simulator(highmem_ld=True, exit_on_ebreak=True)
    r = runfiles.Create()
    
    elf_file = r.Rlocation('coralnpu_hw/tests/npusim_examples/run_yolo26n_v2_binary.elf')

    entry_point, symbol_map = npu_sim.get_elf_entry_and_symbol(
        elf_file, ['inference_status', 'inference_input', 'inference_output']
    )
    npu_sim.load_program(elf_file, entry_point)

    if symbol_map.get('inference_input'):
        print("Loading input .bin file into memory...")
        input_bin_path = "/home/ubuntu/coralnpu/cat_input.bin" 
        input_data = np.fromfile(input_bin_path, dtype=np.int8)
        npu_sim.write_memory(symbol_map['inference_input'], input_data)

    print("Running simulation...", flush=True)
    npu_sim.run()
    npu_sim.wait()
    print(f"Cycles taken by the simulation: {npu_sim.get_cycle_count()}")

    if symbol_map.get('inference_output'):
        output_size_elements = 25200 * 6 
        
        output_data = npu_sim.read_memory(symbol_map['inference_output'], output_size_elements)
        output_data = np.array(output_data, dtype=np.int8)
        print(f"Output info: Retrieved {len(output_data)} elements.")
        
        # Reshape flat elements into standard YOLO tracking dimensions (25200 anchors, 6 features)
        predictions = output_data.reshape((25200, 6))
        
        # Extract raw attributes directly without floating-point conversion
        box_coords = predictions[:, 0:4].astype(np.float32)
        raw_scores = predictions[:, 4]  # Raw INT8 value (-128 to 127)
        class_ids = predictions[:, 5].astype(int)

        # Filter boxes using an arbitrary raw threshold threshold (e.g., score > 0)
        # In INT8, higher numbers mean higher confidence, independent of the floating point scale.
        conf_mask = raw_scores > 0
        filtered_boxes = box_coords[conf_mask]
        filtered_scores = raw_scores[conf_mask]
        filtered_classes = class_ids[conf_mask]

        keep_indices = non_max_suppression_raw(filtered_boxes, filtered_scores)
        
        print("\n==============================")
        print("    VISUAL DETECTION RESULT    ")
        print("==============================")
        if len(keep_indices) == 0:
            print("No recognizable objects found above the raw threshold limit.")
        for idx in keep_indices:
            cid = filtered_classes[idx]
            label = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"Unknown Class ID {cid}"
            print(f"👉 IMAGE CONTAINS: {label.upper()} (Raw Score Value: {filtered_scores[idx]})")
        print("==============================\n")

        output_data.tofile("yolo_output.bin")

    if symbol_map.get('inference_status'):
        inference_status = npu_sim.read_memory(symbol_map['inference_status'], 1)[0]
        print(f"Inference status: {inference_status}")

if __name__ == "__main__":
    run_yolo()
