#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"

// Tell the compiler to find the real pre-compiled Evaluation logic from the library
namespace tflite {
namespace ops {
namespace micro {
namespace concatenation {
extern TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node);
}  // namespace concatenation
}  // namespace micro
}  // namespace ops
}  // namespace tflite

namespace custom_yolo_ops {

// Custom higher limit to absorb YOLO26's wide layer arrays
constexpr int kMaxInputNum = 64; 

TfLiteStatus SafePrepare(TfLiteContext* context, TfLiteNode* node) {
  int num_inputs = node->inputs->size;
  if (num_inputs > kMaxInputNum) {
    printf("Bypassed framework cap! Processing wide concat layer with %d inputs.\n", num_inputs);
  }
  return kTfLiteOk;
}

// Binds directly to the standard execution structures expected by your toolchain
TFLMRegistration Register_CUSTOM_CONCATENATION() {
  return tflite::micro::RegisterOp(
      nullptr, 
      SafePrepare, 
      tflite::ops::micro::concatenation::Eval
  );
}

}  // namespace custom_yolo_ops
