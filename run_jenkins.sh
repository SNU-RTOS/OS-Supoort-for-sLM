#!/bin/bash

source .env

MODEL_PATH="${MODEL_PATH}/llama-3.2-3b-it-q8"
MODEL_NAME="llama_q8_ekv1024"

echo "[INFO] Start LLM inference"
echo "[INFO] Model: ${MODEL_NAME}"

FILE="./${PROMPT_PATH}/sample_prompt_8_3.txt"

if [ ! -f "$FILE" ]; then
    echo "Error: File '$FILE' does not exist."
    exit 1
fi

NUM_REPEATS=2
PROMPT_ITEM_SIZE=3

CGROUP_MMAX=(
    "512M"
    "1G"
    "2G"
    "3G"
    "256M"
    #"2.5G"
    "4G"
    "3.5G"
    "3.7G"
)

#######################

for MMAX in "${CGROUP_MMAX[@]}"; do
    
    RESULTS_DIR="result_dp_${MMAX}"
    echo $MMAX
    if [ ! -d "$RESULTS_DIR" ]; then
        mkdir "$RESULTS_DIR"
    fi

    for ((i=0; i<=NUM_REPEATS; i++)); do
        echo "========== Iteration $i/$NUM_REPEATS =========="

        prompt_id=1 # prompt_id 변수 초기화
        
        while read -r line; do # while read 루프 시작: 파일에서 한 줄씩 읽어옴
            if [[ "$line" =~ ^([0-9]+),\"(.*)\"$ ]]; then
                echo "[INFO] Clearing CPU Caches"
                ./clear_cache_x86

                token_count="${BASH_REMATCH[1]}"
                prompt="${BASH_REMATCH[2]}"

                PERF_FILE="${RESULTS_DIR}/perf_${token_count}_$((${i}*${PROMPT_ITEM_SIZE}+${prompt_id})).data"
                OUTPUT_FILE="${RESULTS_DIR}/output_${token_count}_$((${i}*${PROMPT_ITEM_SIZE}+${prompt_id})).txt"
                CSV_FILE="${RESULTS_DIR}/memusage_${token_count}_$((${i}*${PROMPT_ITEM_SIZE}+${prompt_id})).csv"

                # echo "[INFO] Running test with $token_count tokens: \"$prompt\""

                # CSV 파일 헤더 추가
                echo "Time (s),VmRSS (KB),VmHWM (KB),VmSize (KB),VmSwap (KB),RssAnon (KB),RssFile (KB)" > "$CSV_FILE"
                
                # 백그라운드 실행 및 PID 추출
                nohup systemd-run --scope \
		            -p MemoryMax=$MMAX \
		            ./text_generator_main \
                    --tflite_model="${MODEL_PATH}/${MODEL_NAME}.tflite" \
                    --sentencepiece_model="${MODEL_PATH}/tokenizer.model" \
                    --max_decode_steps=128 \
                    --start_token="<bos>" \
                    --stop_token="<eos>" \
                    --num_threads=8 \
                    --prompt="$prompt" \
                    --weight_cache_path="${MODEL_PATH}/${MODEL_NAME}.xnnpack_cache" > "$OUTPUT_FILE" 2>&1 &

                PID=$!
                echo "Profiling PID: $PID..."

                # perf record 실행 (출력과 성능 데이터 저장)
                nohup perf record -e major-faults \
                    -e block:block_rq_insert \
                    -e block:block_rq_issue \
                    -o "$PERF_FILE" \
                    -p $PID > /dev/null 2>&1 &

                peak_mem=0
                start_time=$(date +%s)

                while kill -0 $PID 2>/dev/null; do
                    mem_rss=$(awk '/VmRSS/ {print $2}' /proc/$PID/status)
                    mem_hwm=$(awk '/VmHWM/ {print $2}' /proc/$PID/status)
                    mem_vmsize=$(awk '/VmSize/ {print $2}' /proc/$PID/status)
                    mem_swap=$(awk '/VmSwap/ {print $2}' /proc/$PID/status)
                    rss_anon=$(awk '/RssAnon/ {print $2}' /proc/$PID/status)
                    rss_file=$(awk '/RssFile/ {print $2}' /proc/$PID/status)

                    if [[ "$mem_rss" =~ ^[0-9]+$ ]]; then
                        if (( mem_hwm > peak_mem )); then
                            peak_mem=$mem_hwm
                        fi

                        elapsed_time=$(( $(date +%s) - start_time ))
                        echo "$elapsed_time,$mem_rss,$mem_hwm,$mem_vmsize,$mem_swap,$rss_anon,$rss_file" >> "$CSV_FILE"
                    fi

                    sleep 1
                done 

                echo "[INFO] Results saved: $PERF_FILE, $OUTPUT_FILE, $CSV_FILE"
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



