#!/bin/bash
# git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
# git checkout 117a62ac439ed87eb26f67208be60e01c21960de
patch -p1 < ../ai-edge-torch/bazel/org_tensorflow_system_python.diff
cd ../ai-edge-torch 
bazel build -c opt //ai_edge_torch/generative/examples/cpp:text_generator_main