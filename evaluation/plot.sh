#!/bin/bash

# TTFT / ATNT / AMPF 다 보여줌
python3.10 show-plot.py \
    --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/Thread1_8_128_512in_32out \
    --ram-sizes 256M 512M 1G 2G 4G 8G

# 256M 제외외 - 이게 진짜임!!! 25-02-23
python3.10 show-plot.py \
    --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/Naive_4Thread_4Core \
    --ram-sizes 512M 1G 2G 4G 8G
    # --input-tokens 8 128 512

python3.10 show-plot.py \
    --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/Naive_4Thread_4Core \
    --ram-sizes 2G Reordering

python3.10 show-plot.py \
    --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/RUBIKPi_8G \
    --ram-sizes T4C4_Naive T4C4_Preload T4C4_NoMmap T4C4_NoMmap_Preload

python3.10 show-plot.py \
    --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/Gemma_Naive_4Thread_4Core \
    --ram-sizes 512M 1G 2G 4G 8G

# 여기서부턴 가짜임! 25-02-23
python3.10 show-plot.py \
    --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/New_Naive \
    --ram-sizes 512M 1G 2G 4G 8G

python3.10 show-plot.py \
    --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/Direct_IO \
    --ram-sizes 512M 1G 2G 4G 8G

python3.10 show-plot.py \
    --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/Thread1_output \
    --ram-sizes Naive Reordering Pinning Reorder+Pin