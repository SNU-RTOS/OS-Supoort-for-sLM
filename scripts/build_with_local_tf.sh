#!/bin/bash

if [ ! -f "./external" ]; then
    mkdir -p ./external
fi

cd external

## Clone ai-edge-torch
if [ ! -f "./ai-edge-torch" ]; then
    echo "[INFO] Installing ai-edge-torch"
    git clone https://github.com/SNU-RTOS/ai-edge-torch.git
    rm ./ai-edge-torch/ai_edge_torch/generative/examples/cpp
    ln -s src ./ai-edge-torch/ai_edge_torch/generative/examples/cpp
fi

## Clone tensorflow
if [ ! -f "./tensorflow" ]; then
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    git checkout 117a62ac439ed87eb26f67208be60e01c21960de
    patch -p1 < ../ai-edge-torch/bazel/org_tensorflow_system_python.diff
    cd ..
fi

## Build text_generator_main (Text genertion app)
cd ./ai-edge-torch 
bazel build -c opt //ai_edge_torch/generative/examples/cpp:text_generator_main