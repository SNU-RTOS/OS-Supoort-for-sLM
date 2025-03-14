#!/bin/bash

source /home/rtos/workspace/ghpark/SOURCE_THIS

MODEL_NAME=gemma-2-2b-it
EXPORT_PATH="/home/rtos/workspace/ghpark/export"
SCRIPT_PATH="/home/rtos/workspace/ghpark/ai-edge-torch/ai_edge_torch/generative/examples/gemma"


echo "========================================"

OUTPUT_PATH="${EXPORT_PATH}/${MODEL_NAME}-fp32"
exec > >(tee "log-${MODEL_NAME}-fp32.log") 2>&1  # stdout + stderr을 로그 파일에 저장 & 실시간 출력
if [ -d "${OUTPUT_PATH}" ]; then
  echo "Directory '${OUTPUT_PATH}' already exists. Skipping creation."
else
  echo "Directory '${OUTPUT_PATH}' does not exist. Creating now."
  mkdir -p "${OUTPUT_PATH}"
  echo "Directory '${OUTPUT_PATH}' created successfully."
fi

python3 ${SCRIPT_PATH}/convert_gemma2_to_tflite.py \
     --checkpoint_path  /home/rtos/workspace/ghpark/models/${MODEL_NAME} \
     --output_path ${OUTPUT_PATH} \
     
    
echo "[INFO] ${OUTPUT_PATH}  exported"



echo "========================================"

OUTPUT_PATH="${EXPORT_PATH}/${MODEL_NAME}-q8"
exec > >(tee "log-${MODEL_NAME}-q8.log") 2>&1  # stdout + stderr을 로그 파일에 저장 & 실시간 출력
if [ -d "${OUTPUT_PATH}" ]; then
  echo "Directory '${OUTPUT_PATH}' already exists. Skipping creation."
else
  echo "Directory '${OUTPUT_PATH}' does not exist. Creating now."
  mkdir -p "${OUTPUT_PATH}"
  echo "Directory '${OUTPUT_PATH}' created successfully."
fi

python3 ${SCRIPT_PATH}/convert_gemma2_to_tflite.py \
     --checkpoint_path  /home/rtos/workspace/ghpark/models/${MODEL_NAME} \
     --output_path ${OUTPUT_PATH} \
     --quantize

echo "[INFO] ${OUTPUT_PATH}  exported"


exit