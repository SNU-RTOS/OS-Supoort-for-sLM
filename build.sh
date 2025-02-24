#!/bin/bash
cd /home/rtos/workspace/ghpark/ai-edge-torch
bazel run -c opt //ai_edge_torch/generative/examples/cpp:text_generator_main 
# bazel run -c opt --copt=-DTFLITE_MMAP_DISABLED //ai_edge_torch/generative/examples/cpp:text_generator_main 
