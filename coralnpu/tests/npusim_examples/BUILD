# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@coralnpu_hw//third_party/python:requirements.bzl", "requirement")
load("//rules:coralnpu_v2.bzl", "coralnpu_v2_binary")

py_binary(
    name = "npusim_run_mobilenet",
    srcs = ["npusim_run_mobilenet.py"],
    data = [
        ":run_full_mobilenet_v1_binary",
    ],
    deps = [
        "//sw/coralnpu_sim:coralnpu_v2_sim_utils_lib",
        "@bazel_tools//tools/python/runfiles",
        requirement("numpy"),
    ],
)

coralnpu_v2_binary(
    name = "run_yolo26n_binary",
    srcs = [
        "run_yolo26n.cc",
        "yolo26n_int8_model.h", 
    ],
    dtcm_size_kbytes = 32768,  
    itcm_size_kbytes = 8192,  
    semihosting = True,
    visibility = ["//visibility:public"],
    deps = [
        "//sw/opt:rvv_opt",
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:depthwise_conv",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro:micro_log",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers", 
        "@tflite_micro//tensorflow/lite/micro:system_setup",
    ],
)


###############################################################
# Add this to your existing BUILD file

py_binary(
    name = "npusim_run_yolo26n",
    srcs = ["npusim_run_yolo26n.py"],
    data = [
        ":run_yolo26n_v2_binary",  # <-- Changed Name Here
    ],
    deps = [
        "//sw/coralnpu_sim:coralnpu_v2_sim_utils_lib",
        "@bazel_tools//tools/python/runfiles",
        requirement("numpy"),
    ],
)

coralnpu_v2_binary(
    name = "run_yolo26n_v2_binary", # <-- Changed Name Here
    srcs = [
        "run_yolo26n.cc",
        "yolo26n_int8_model.h", 
    ],
    #dtcm_size_kbytes = 32768,  
    #itcm_size_kbytes = 8192,
    dtcm_size_kbytes = 65536,  
    itcm_size_kbytes = 16384,      
    semihosting = True,
    visibility = ["//visibility:public"],
    deps = [
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:depthwise_conv",
        "//sw/opt:rvv_opt",
        "@tflite_micro//tensorflow/lite/core/c:common",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro:system_setup",
        "@tflite_micro//tensorflow/lite/micro/kernels:micro_ops", 
    ],
)


###############################################################
# Add this code block at the bottom of tests/npusim_examples/BUILD

## coralnpu_v2_binary(
##     name = "run_yolo11_binary",
##     srcs = ["run_yolo11.cc"],
##     hdrs = ["yolo11n_int8_model.h"],
##     dtcm_size_kbytes = 32768,  
##     itcm_size_kbytes = 32768,  
##     semihosting = True,
##     visibility = ["//visibility:public"],
##     deps = [
##         "//sw/opt/litert-micro:conv",
##         "//sw/opt/litert-micro:depthwise_conv",
##         "//sw/opt:rvv_opt",
##         "@tflite_micro//tensorflow/lite/core/c:common",
##         "@tflite_micro//tensorflow/lite/micro:micro_framework",
##         "@tflite_micro//tensorflow/lite/micro:system_setup",
##         "@tflite_micro//tensorflow/lite/micro/kernels:micro_ops", # CRUCIAL: Added for standard reference ops
##     ],
## )
## 
## 
## 
## py_binary(
##     name = "npusim_run_yolo11",
##     srcs = ["npusim_run_yolo11.py"],
##     data = [":run_yolo11_binary"],
##     deps = [
##         "//sw/coralnpu_sim:coralnpu_v2_sim_utils_lib",  # CORRECTED PATH
##         "@bazel_tools//tools/python/runfiles",
##         requirement("numpy"),                           # Added dependency
##     ],
## )


