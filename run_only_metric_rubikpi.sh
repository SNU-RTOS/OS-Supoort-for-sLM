#!/bin/bash

clear_caches() {
    echo "[INFO] Dropping OS Page Caches.."
    sync && echo 3 | tee /proc/sys/vm/drop_caches > /dev/null
    echo "[INFO] Clearing CPU Caches"

    if [[ -f "./clear_cache_arm" ]]; then
        ./clear_cache_arm
    else
        echo "[WARNING] CPU cache clearing script not found: ./clear_cache_arm"
    fi

    echo "[INFO] DONE: Clearing Caches"
    echo ""
}

MODEL_PATH="./models/llama-3.2-3b-it-q8"
MODEL_NAME="llama_q8_ekv1024"

echo "[INFO] Start LLM inference"
echo "[INFO] Model: ${MODEL_NAME}"

FILE="sample_prompt.txt" # prompts_v8.txt 로 파일명 변경 (변수 사용 권장)

if [ ! -f "$FILE" ]; then
    echo "Error: 파일 '$FILE'이 존재하지 않습니다."
    exit 1
fi

NUM_REPEATS=0
PROMPT_ITEM_SIZE=5

CGROUP_MMAX=(
    "2G"
    "4G"
    "8G"
)

for MMAX in "${CGROUP_MMAX[@]}"; do
    
    RESULTS_DIR="result_dp_${MMAX}"
    
    if [ ! -d "$RESULTS_DIR" ]; then
        mkdir "$RESULTS_DIR"
    fi

    for ((i=0; i<=NUM_REPEATS; i++)); do
        echo "========== Iteration $i/$NUM_REPEATS =========="

        prompt_id=1 # prompt_id 변수 초기화
        
        while read -r line; do # while read 루프 시작: 파일에서 한 줄씩 읽어옴
            if [[ "$line" =~ ^([0-9]+),\"(.*)\"$ ]]; then
                clear_caches

                token_count="${BASH_REMATCH[1]}"
                prompt="${BASH_REMATCH[2]}"

                # PERF_FILE="${RESULTS_DIR}/perf_${token_count}_$((${i}*${PROMPT_ITEM_SIZE}+${prompt_id})).data"
                OUTPUT_FILE="${RESULTS_DIR}/output_${token_count}_$((${i}*${PROMPT_ITEM_SIZE}+${prompt_id})).txt"
                # CSV_FILE="${RESULTS_DIR}/memusage_${token_count}_$((${i}*${PROMPT_ITEM_SIZE}+${prompt_id})).csv"
                
                # 백그라운드 실행 및 PID 추출
                systemd-run --scope -p MemoryMax=$MMAX ./text_generator_main \
                    --tflite_model="${MODEL_PATH}/${MODEL_NAME}.tflite" \
                    --sentencepiece_model="${MODEL_PATH}/tokenizer.model" \
                    --max_decode_steps=32 \
                    --start_token="<bos>" \
                    --stop_token="<eos>" \
                    --num_threads=8 \
                    --prompt="$prompt"
                    # --weight_cache_path="${MODEL_PATH}/${MODEL_NAME}.xnnpack_cache"

                PID=$!
                #echo "Profiling PID: $PID..."
                #echo "[INFO] Results saved: $PERF_FILE, $OUTPUT_FILE, $CSV_FILE"

		#while kill -0 $PID 2>/dev/null; do
		#	sleep 1
		#done

                echo "==================================="
                echo ""
            fi

            prompt_id=$((prompt_id + 1)) # prompt_id 증가
            if (( prompt_id > ${PROMPT_ITEM_SIZE} )); then # prompt_id가 10보다 크면 1로 초기화
                prompt_id=1
            fi

        done <"$FILE" # 입력 재지향: sample_prompt.txt 파일로부터 입력 받음
    done

done



