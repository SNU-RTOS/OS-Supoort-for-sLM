#!/bin/bash

TRACEFS="/sys/kernel/debug/tracing"
TRACE_SAVE_INTERVAL=5  # Save trace every 5 seconds

setup_ftrace() {
    # Clear existing trace
    echo > $TRACEFS/trace

    # Disable tracing temporarily
    echo 0 > $TRACEFS/tracing_on

    # Clear existing events
    echo > $TRACEFS/set_event

    # Enable memory-related events
    echo 1 > $TRACEFS/events/kmem/mm_page_alloc/enable
    echo 1 > $TRACEFS/events/kmem/mm_page_free/enable
    echo 1 > $TRACEFS/events/kmem/rss_stat/enable
    
    # Memory reclaim events
    echo 1 > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_begin/enable
    echo 1 > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_end/enable
    
    # Set the trace buffer size (in KB)
    echo 8192 > $TRACEFS/buffer_size_kb

    # Use function tracer
    echo function > $TRACEFS/current_tracer
    
    # Clear the trace buffer
    echo > $TRACEFS/trace
}

setup_pid_filter() {
    local pid=$1
    
    echo "common_pid==$pid" > $TRACEFS/events/kmem/mm_page_alloc/filter
    echo "common_pid==$pid" > $TRACEFS/events/kmem/mm_page_free/filter
    echo "common_pid==$pid" > $TRACEFS/events/kmem/rss_stat/filter
    echo "common_pid==$pid" > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_begin/filter
    echo "common_pid==$pid" > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_end/filter
}

save_trace_buffer() {
    local output_file=$1
    local elapsed_time=$2
    
    # Save current trace buffer by appending to the output file
    echo -e "\n# Elapsed time: $elapsed_time seconds" >> "$output_file"
    cat $TRACEFS/trace >> "$output_file"
    
    # Clear the buffer for next collection
    echo > $TRACEFS/trace
    
    echo "[INFO] Appended trace data at $elapsed_time seconds"
}

get_pagefault_stats() {
    local pid=$1
    local stat_line=$(cat /proc/$pid/stat 2>/dev/null)
    if [ $? -eq 0 ]; then
        local stats=($stat_line)
        echo "${stats[9]},${stats[11]}"
    else
        echo "0,0"
    fi
}

clear_caches() {
    echo "[INFO] Dropping OS Page Caches.."
    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    echo "[INFO] Clearing CPU Caches"

    if [[ -f "./clear_cache_arm" ]]; then
        ./clear_cache_arm
    else
        echo "[WARNING] CPU cache clearing script not found: ./clear_cache_arm"
    fi

    echo "[INFO] DONE: Clearing Caches"
    echo ""
}

# Llama 3.2 3B INT8 quantized
MODEL_PATH="/home/rtos/workspace/ghpark/export//llama-3.2-3b-it-q8"
MODEL_NAME="llama_q8_ekv1024"

# Gemma 2B FP32
# MODEL_PATH="./models/gemma-2-2b-it"
# MODEL_NAME="gemma2_f32_ekv1024"

echo "[INFO] Start LLM inference"
echo "[INFO] Model: ${MODEL_NAME}"

FILE="./prompt/sample_prompt.txt"

if [ ! -f "$FILE" ]; then
    echo "Error: File '$FILE' does not exist."
    exit 1
fi

NUM_REPEATS=0
PROMPT_ITEM_SIZE=3

CGROUP_MMAX=(
    #"8G"
    #"4G"
    #"2G"
    #"1G"
    #"512M"
    "256M"
)

# Cleanup existing cgroup if it exists
# if [ -d "/sys/fs/cgroup/memory/mygroup" ]; then
#     # Kill any processes still in the cgroup
#     kill -9 $(cat /sys/fs/cgroup/memory/mygroup/tasks 2>/dev/null) 2>/dev/null
#     # Remove the cgroup
#     rmdir /sys/fs/cgroup/memory/mygroup 2>/dev/null
# fi

# mkdir -p /sys/fs/cgroup/memory/mygroup
# echo 1 > /sys/fs/cgroup/memory/mygroup/memory.use_hierarchy
# echo 0 > /sys/fs/cgroup/memory/mygroup/memory.oom_control
CGROUP_PATH="/sys/fs/cgroup/mygroup"

for MMAX in "${CGROUP_MMAX[@]}"; do
    RESULTS_DIR="result_dp_${MMAX}"

    if [ ! -d "$RESULTS_DIR" ]; then
        mkdir "$RESULTS_DIR"
    fi

    # // Set memory limit of the cgroup
    # echo $MMAX > /sys/fs/cgroup/memory/mygroup/memory.limit_in_bytes
    # echo 0 > /sys/fs/cgroup/memory/mygroup/memory.force_empty
    echo "[INFO] Setting cgroup memory limit to ${MMAX}"
    echo $MMAX > "$CGROUP_PATH/memory.max"

    for ((i=0; i<=NUM_REPEATS; i++)); do
        echo "========== Iteration $i/$NUM_REPEATS =========="

        prompt_id=1

        while read -r line; do
            if [[ "$line" =~ ^([0-9]+),\"(.*)\"$ ]]; then
                # clear_caches

                token_count="${BASH_REMATCH[1]}"
                prompt="${BASH_REMATCH[2]}"

                TRACE_FILE="${RESULTS_DIR}/ftrace_${token_count}_$((${i}*${PROMPT_ITEM_SIZE}+${prompt_id})).txt"
                OUTPUT_FILE="${RESULTS_DIR}/output_${token_count}_$((${i}*${PROMPT_ITEM_SIZE}+${prompt_id})).txt"
                CSV_FILE="${RESULTS_DIR}/memusage_${token_count}_$((${i}*${PROMPT_ITEM_SIZE}+${prompt_id})).csv"
                PAGEFAULT_FILE="${RESULTS_DIR}/pagefault_${token_count}_$((${i}*${PROMPT_ITEM_SIZE}+${prompt_id})).csv"

                # Set up ftrace
                # setup_ftrace

                # Create initial trace file with header
                # echo "# Trace started at: $(date)" > "$TRACE_FILE"
                # echo "# Process ID: $PID" >> "$TRACE_FILE"
                # echo "# Token count: $token_count" >> "$TRACE_FILE"
                # echo "# Memory limit: $MMAX" >> "$TRACE_FILE"
                # echo -e "# ==================\n" >> "$TRACE_FILE"

                # CSV file headers
                echo "Time (s),VmRSS (KB),VmHWM (KB),VmSize (KB),VmSwap (KB),RssAnon (KB),RssFile (KB)" > "$CSV_FILE"
                echo "Time (s),MinorFaults,MajorFaults,MinorFaultsDelta,MajorFaultsDelta" > "$PAGEFAULT_FILE"

                # Start tracing
                # echo 1 > $TRACEFS/tracing_on

                # Run the application
                nohup cgexec -g memory:mygroup \
                    taskset -c 0-0 \
                    ./text_generator_main \
                    --tflite_model="${MODEL_PATH}/${MODEL_NAME}.tflite" \
                    --sentencepiece_model="${MODEL_PATH}/tokenizer.model" \
                    --max_decode_steps=3 \
                    --start_token="<bos>" \
                    --stop_token="<eos>" \
                    --num_threads=1 \
                    --prompt="$prompt" \
                    --weight_cache_path="${MODEL_PATH}/${MODEL_NAME}.xnnpack_cache" > "$OUTPUT_FILE" 2>&1 &

                PID=$!
                echo "Tracing PID: $PID..."

                # Set up PID-specific filtering
                # setup_pid_filter $PID
                
                peak_mem=0
                start_time=$(date +%s)
                last_minor_faults=0
                last_major_faults=0
                last_trace_save=$start_time

                # Monitor process and save trace periodically
                while kill -0 $PID 2>/dev/null; do
                    current_time=$(date +%s)
                    elapsed_time=$((current_time - start_time))
                    
                    # Save trace buffer periodically
                    # if ((current_time - last_trace_save >= TRACE_SAVE_INTERVAL)); then
                    #     save_trace_buffer "$TRACE_FILE" $elapsed_time
                    #     last_trace_save=$current_time
                    # fi

                    # Memory statistics
                    mem_rss=$(awk '/VmRSS/ {print $2}' /proc/$PID/status)
                    mem_hwm=$(awk '/VmHWM/ {print $2}' /proc/$PID/status)
                    mem_vmsize=$(awk '/VmSize/ {print $2}' /proc/$PID/status)
                    mem_swap=$(awk '/VmSwap/ {print $2}' /proc/$PID/status)
                    rss_anon=$(awk '/RssAnon/ {print $2}' /proc/$PID/status)
                    rss_file=$(awk '/RssFile/ {print $2}' /proc/$PID/status)

                    # Page fault statistics
                    IFS=',' read -r minor_faults major_faults <<< $(get_pagefault_stats $PID)
                    minor_faults_delta=$((minor_faults - last_minor_faults))
                    major_faults_delta=$((major_faults - last_major_faults))

                    if [[ "$mem_rss" =~ ^[0-9]+$ ]]; then
                        if (( mem_hwm > peak_mem )); then
                            peak_mem=$mem_hwm
                        fi

                        echo "$elapsed_time,$mem_rss,$mem_hwm,$mem_vmsize,$mem_swap,$rss_anon,$rss_file" >> "$CSV_FILE"
                        echo "$elapsed_time,$minor_faults,$major_faults,$minor_faults_delta,$major_faults_delta" >> "$PAGEFAULT_FILE"
                    fi

                    last_minor_faults=$minor_faults
                    last_major_faults=$major_faults

                    sleep 1
                done
                
                # # Save final trace buffer
                # save_trace_buffer "$TRACE_FILE" $elapsed_time

                # # Stop tracing
                # echo 0 > $TRACEFS/tracing_on

                # # Add end marker to trace file
                # echo -e "\n# Trace ended at: $(date)" >> "$TRACE_FILE"
                # echo "# Total runtime: $elapsed_time seconds" >> "$TRACE_FILE"

                echo "[INFO] Results saved: $TRACE_FILE, $OUTPUT_FILE, $CSV_FILE, $PAGEFAULT_FILE"
                echo "==================================="
                echo ""
            fi

            prompt_id=$((prompt_id + 1))
            if (( prompt_id > ${PROMPT_ITEM_SIZE} )); then
                prompt_id=1
            fi

        done < "$FILE"
    done
done

# Cleanup: Disable all events
# echo 0 > $TRACEFS/events/kmem/mm_page_alloc/enable
# echo 0 > $TRACEFS/events/kmem/mm_page_free/enable
# echo 0 > $TRACEFS/events/kmem/rss_stat/enable
# echo 0 > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_begin/enable
# echo 0 > $TRACEFS/events/vmscan/mm_vmscan_direct_reclaim_end/enable
