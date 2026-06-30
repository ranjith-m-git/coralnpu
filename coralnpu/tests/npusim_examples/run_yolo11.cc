// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>

#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/opt/rvv_opt.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

// 1. UPDATED: Includes your pure, fully quantized INT8 model array header file
#include "tests/npusim_examples/yolo11n_int8_model.h"

namespace {
// Set capacity to 30 ops to safely hold all YOLO11 operator variations
using YoloOpResolver = tflite::MicroMutableOpResolver<30>;

// RESTORED: Bringing back the optimized Google Coral NPU custom operators
using coralnpu_v2::opt::litert_micro::Register_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D;

TfLiteStatus RegisterOps(YoloOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D(Register_CONV_2D()));
  TF_LITE_ENSURE_STATUS(
      op_resolver.AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D()));
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  TF_LITE_ENSURE_STATUS(op_resolver.AddShape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPack());
  
  TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic()); 
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSub());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTranspose());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSplit());
  TF_LITE_ENSURE_STATUS(op_resolver.AddBatchMatMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddResizeNearestNeighbor());

  // FIXED: Added the missing quantization scaling operator
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());

  return kTfLiteOk;
}

}  // namespace

extern "C" {
// Expanded to 24MB to stop runtime out-of-memory heap faults
constexpr size_t kTensorArenaSize = 24 * 1024 * 1024;  
int8_t inference_status = -1;

uint8_t inference_input[640 * 640 * 3]
    __attribute__((section(".data"), aligned(16)));

int8_t inference_output[2000] __attribute__((section(".data"), aligned(16)));

uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
}

int main(int argc, char** argv) {
  // Pulls the verified string name generated inside your header via sed
  const tflite::Model* model = tflite::GetModel(yolo11n_int8_tflite);
  
  YoloOpResolver op_resolver;
  RegisterOps(op_resolver);
  printf("Halted after op resolver\n");
  
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  printf("Halted after Interpreter setup\n");
  
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("Error during AllocateTensors\n");
    return -1;
  }
  
  TfLiteTensor* input = interpreter.input(0);
  if (input == nullptr) {
    printf("Error getting input tensor\n");
    return -1;
  }
  coralnpu_v2::opt::Memcpy(input->data.data, inference_input, input->bytes);

  if (interpreter.Invoke() != kTfLiteOk) {
    printf("Error during Invoke\n");
    return -1;
  }

  TfLiteTensor* output = interpreter.output(0);
  if (output == nullptr) {
    printf("Error getting output tensor\n");
    return -1;
  }
  
  // Safely extract the target byte arrays up to the configured limits
  size_t bytes_to_copy = output->bytes < 1000 ? output->bytes : 1000;
  coralnpu_v2::opt::Memcpy(inference_output, output->data.data, bytes_to_copy);
  
  printf("Invoke successful\n");
  inference_status = 0;
  return 0;
}
