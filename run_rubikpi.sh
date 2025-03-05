#!/bin/bash

# Enhanced LLM Memory and I/O Profiler Script for ARM
# Comprehensive memory, strace, and I/O profiling for LLM inference workloads

# Exit on any error
# set -e

# Configuration Variables
MODEL_PATH="./models/llama-3.2-3b-it-q8"
MODEL_NAME="llama_q8_ekv1024"
PROMPT_FILE="sample_prompt.txt"
NUM_REPEATS=0
PROMPT_ITEM_SIZE=10
CGROUP_MMAX=("512M" "256M")
LOG_FILE="profiler.log"

# Enhanced logging function with categories
log() {
    local category=$1
    shift
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$category] $*" | tee -a "$LOG_FILE"
}

# Enhanced cleanup function with comprehensive process and resource management
cleanup() {
    log "CLEANUP" "Cleaning up resources..."
    
    # Kill monitoring processes from PID files
    for pidfile in result_dp_*/*/memusage_*_monitor.pid result_dp_*/*/io_stats_*_monitor.pid; do
        if [[ -f "$pidfile" ]]; then
            monitor_pid=$(cat "$pidfile")
            log "CLEANUP" "Killing monitoring process: $monitor_pid"
            kill "$monitor_pid" 2>/dev/null || true
            wait "$monitor_pid" 2>/dev/null || true
            rm "$pidfile"
        fi
    done
    
    # Kill sub-processes from PID files
    for pidfile in result_dp_*/*/memusage_*.csv.pid \
                   result_dp_*/*/strace_*.txt.pid \
                   result_dp_*/*/io_stats_*.csv.pid; do
        if [[ -f "$pidfile" ]]; then
            proc_pid=$(cat "$pidfile")
            log "CLEANUP" "Killing sub-process: $proc_pid"
            kill "$proc_pid" 2>/dev/null || true
            wait "$proc_pid" 2>/dev/null || true
            rm "$pidfile"
        fi
    done
    
    # Kill specific monitoring processes if they exist
    log "CLEANUP" "Killing specific monitoring processes"
    
    # Kill memory monitoring processes
    if [[ -n "$MEMORY_MONITOR_PID" ]]; then
        kill $MEMORY_MONITOR_PID 2>/dev/null || true
        wait $MEMORY_MONITOR_PID 2>/dev/null || true
        unset MEMORY_MONITOR_PID
    fi
    
    # Kill I/O monitoring processes
    if [[ -n "$IO_MONITOR_PID" ]]; then
        kill $IO_MONITOR_PID 2>/dev/null || true
        wait $IO_MONITOR_PID 2>/dev/null || true
        unset IO_MONITOR_PID
    fi
    
    # Kill strace process
    if [[ -n "$STRACE_PID" ]]; then
        kill $STRACE_PID 2>/dev/null || true
        wait $STRACE_PID 2>/dev/null || true
        unset STRACE_PID
    fi
    
    # Kill perf recording process
    if [[ -n "$PERF_PID" ]]; then
        kill $PERF_PID 2>/dev/null || true
        wait $PERF_PID 2>/dev/null || true
        unset PERF_PID
    fi
    
    # Kill any remaining child processes
    log "CLEANUP" "Killing remaining child processes"
    pkill -P $$ || true
    
    # Disable kernel tracing
    log "CLEANUP" "Disabling kernel tracing"
    echo 0 > /sys/kernel/tracing/tracing_on 2>/dev/null || true
    
    # Clear any remaining monitor PIDs
    unset BLKTRACE_PID MEMORY_MONITOR_PID IO_MONITOR_PID STRACE_PID PERF_PID
    
    log "CLEANUP" "Cleanup completed successfully"
}

# Trap EXIT signal to ensure cleanup happens even if script exits unexpectedly
trap cleanup EXIT

# Clear system caches
clear_caches() {
    log "CACHE" "Dropping OS Page Caches.."
    sync && echo 3 | tee /proc/sys/vm/drop_caches > /dev/null

    log "CACHE" "Clearing CPU Caches"
    if [[ -f "./clear_cache_arm" ]]; then
        ./clear_cache_arm
    else
        log "WARNING" "CPU cache clearing script not found: ./clear_cache_arm"
    fi

    log "CACHE" "Cache clearing completed"
}

# Function to get the root disk device
get_root_disk() {
    # Get the base disk device for root partition
    root_part=$(mount | grep ' / ' | cut -d' ' -f1)
    if [[ "$root_part" == "/dev/root" ]]; then
        # If mounted as /dev/root, find the actual device
        root_part=$(findmnt -n -o SOURCE /)
    fi
    
    # Get the base disk device (remove partition number)
    if [[ "$root_part" =~ ^/dev/[a-zA-Z]+[0-9]+$ ]]; then
        # Extract the base device name (e.g., /dev/sda from /dev/sda8)
        root_disk=$(echo "$root_part" | sed 's/[0-9]*$//')
        echo "$root_disk"
    else
        # Fallback to first disk if detection fails
        echo "/dev/sda"
    fi
}

# Setup tracing for memory and I/O events
setup_tracing() {
    local PID=$1
    log "TRACE" "Setting up comprehensive tracing for PID: $PID"

    # Mount tracefs if not mounted
    if ! grep -q tracefs /proc/mounts; then
        mount -t tracefs nodev /sys/kernel/tracing
    fi

    # Clear previous trace
    echo > /sys/kernel/tracing/trace

    # Set up tracing for specific events and PID
    echo "$PID" > /sys/kernel/tracing/set_event_pid 2>/dev/null || true

    # Combined tracing events for both memory and I/O
    local trace_events=(
        # Memory events
        "kmem/mm_page_alloc"
        "kmem/mm_page_free"
        "vmscan/mm_vmscan_direct_reclaim_begin"
        "vmscan/mm_vmscan_direct_reclaim_end"
        "sched/sched_process_fork"
        "sched/sched_process_exec"
        "sched/sched_process_exit"
        # I/O events
        "block/block_rq_issue"
        "block/block_rq_complete"
        "block/block_rq_insert"
        "filemap/mm_filemap_add_to_page_cache"
        "filemap/mm_filemap_delete_from_page_cache"
        "writeback/writeback_dirty_page"
        "jbd2/jbd2_commit_flushing"
        "ext4/ext4_da_write_begin"
        "ext4/ext4_da_write_end"
    )

    # Enable all tracing events
    for event in "${trace_events[@]}"; do
        trace_path="/sys/kernel/tracing/events/${event}/enable"
        if [[ -f "$trace_path" ]]; then
            echo 1 > "$trace_path"
        else
            log "WARNING" "Tracing event ${event} not available"
        fi
    done

    # Turn on tracing
    echo 1 > /sys/kernel/tracing/tracing_on
}

# Enhanced memory monitoring with strace integration
monitor_memory_advanced() {
    local PID=$1
    local CSV_FILE=$2
    local STRACE_FILE=$3
    local MEMORY_MAP_FILE="${CSV_FILE%.csv}_maps.txt"
    local MONITOR_PID_FILE="${CSV_FILE%.csv}_monitor.pid"
    
    # Create a monitoring function that will run in the background
    (
        # Start strace in background
        strace -f -e trace=memory -p "$PID" -o "$STRACE_FILE" 2>/dev/null &
        STRACE_PID=$!
        
        # Monitor memory maps periodically
        {
            while kill -0 "$PID" 2>/dev/null; do
                echo "=== Memory Map at $(date) ===" >> "$MEMORY_MAP_FILE"
                cat "/proc/$PID/maps" >> "$MEMORY_MAP_FILE" 2>/dev/null || true
                echo >> "$MEMORY_MAP_FILE"
                sleep 1
            done
        } &
        
        # Enhanced memory statistics
        {
            echo "Timestamp,VmRSS,VmHWM,VmSize,VmSwap,RssAnon,RssFile,PSS,USS,SharedMem,PrivateMem,SwapPSS,Locked,PageFaults,MajorFaults"
            
            while kill -0 "$PID" 2>/dev/null; do
                timestamp=$(date +%s.%N)
                
                # Get detailed memory stats from status
                mem_stats=$(grep -E "Vm|Rss|Pss|Swap" "/proc/$PID/status" 2>/dev/null)
                vm_rss=$(echo "$mem_stats" | grep VmRSS | awk '{print $2}')
                vm_hwm=$(echo "$mem_stats" | grep VmHWM | awk '{print $2}')
                vm_size=$(echo "$mem_stats" | grep VmSize | awk '{print $2}')
                vm_swap=$(echo "$mem_stats" | grep VmSwap | awk '{print $2}')
                rss_anon=$(echo "$mem_stats" | grep RssAnon | awk '{print $2}')
                rss_file=$(echo "$mem_stats" | grep RssFile | awk '{print $2}')
                
                # Get detailed memory stats from smaps
                if [[ -f "/proc/$PID/smaps" ]]; then
                    pss=$(awk '/Pss:/ {sum += $2} END {print sum}' "/proc/$PID/smaps")
                    uss=$(awk '/Private_Clean:|Private_Dirty:/ {sum += $2} END {print sum}' "/proc/$PID/smaps")
                    shared=$(awk '/Shared_Clean:|Shared_Dirty:/ {sum += $2} END {print sum}' "/proc/$PID/smaps")
                    private=$(awk '/Private_Clean:|Private_Dirty:/ {sum += $2} END {print sum}' "/proc/$PID/smaps")
                    swap_pss=$(awk '/SwapPss:/ {sum += $2} END {print sum}' "/proc/$PID/smaps")
                    locked=$(awk '/Locked:/ {sum += $2} END {print sum}' "/proc/$PID/smaps")
                else
                    pss=0; uss=0; shared=0; private=0; swap_pss=0; locked=0
                fi
                
                # Get page fault statistics
                if [[ -f "/proc/$PID/stat" ]]; then
                    stat_data=( $(cat "/proc/$PID/stat") )
                    page_faults=${stat_data[9]:-0}
                    major_faults=${stat_data[11]:-0}
                else
                    page_faults=0; major_faults=0
                fi
                
                echo "$timestamp,$vm_rss,$vm_hwm,$vm_size,$vm_swap,$rss_anon,$rss_file,$pss,$uss,$shared,$private,$swap_pss,$locked,$page_faults,$major_faults"
                sleep 0.1
                
            done
        } > "$CSV_FILE" 2>/dev/null &
        
        # Store PIDs of background processes to allow cleanup
        echo $STRACE_PID > "${STRACE_FILE}.pid"
        echo $! > "${CSV_FILE}.pid"
    ) &
    
    # Store the main monitoring process PID
    echo $! > "$MONITOR_PID_FILE"
    
    # Return the PID of the monitoring process
    echo $!
}

# Enhanced I/O monitoring function
monitor_io() {
    local PID=$1
    local IO_STATS_FILE=$2
    local ROOT_DISK=$(get_root_disk)
    local DISK_NAME=$(basename "$ROOT_DISK")
    local MONITOR_PID_FILE="${IO_STATS_FILE%.csv}_monitor.pid"
    
    log "Using disk device for I/O monitoring: $ROOT_DISK"
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$IO_STATS_FILE")"
    
    # Create a monitoring function that will run in the background
    (
        {
            echo "Timestamp,Read_IO_Ops,Write_IO_Ops,Read_Bytes,Write_Bytes,IO_Time_ms,IO_Queue_Length"
            
            while true; do
                if ! kill -0 "$PID" 2>/dev/null; then
                    log "Process $PID has terminated, stopping I/O monitoring"
                    break
                fi
                
                timestamp=$(date +%s.%N)
                
                # Process-specific I/O statistics with error handling
                if [[ -f "/proc/$PID/io" ]]; then
                    read_bytes=$(awk '/read_bytes/{print $2}' "/proc/$PID/io" 2>/dev/null || echo 0)
                    write_bytes=$(awk '/write_bytes/{print $2}' "/proc/$PID/io" 2>/dev/null || echo 0)
                    read_ops=$(awk '/syscr/{print $2}' "/proc/$PID/io" 2>/dev/null || echo 0)
                    write_ops=$(awk '/syscw/{print $2}' "/proc/$PID/io" 2>/dev/null || echo 0)
                else
                    read_bytes=0
                    write_bytes=0
                    read_ops=0
                    write_ops=0
                fi
                
                # Disk statistics with error handling
                if [[ -f "/proc/diskstats" ]]; then
                    disk_stats=$(awk -v disk="$DISK_NAME" '$3==disk{print $13","$14}' /proc/diskstats 2>/dev/null)
                    if [[ -z "$disk_stats" ]]; then
                        disk_stats="0,0"
                    fi
                else
                    disk_stats="0,0"
                fi
                
                echo "$timestamp,$read_ops,$write_ops,$read_bytes,$write_bytes,$disk_stats"
                sleep 0.1
                
            done 
        } > "$IO_STATS_FILE" 2>/dev/null &
        
        # Store PID of background process
        echo $! > "${IO_STATS_FILE}.pid"
    ) &
    
    # Store the main monitoring process PID
    echo $! > "$MONITOR_PID_FILE"
    
    # Return the PID of the monitoring process
    echo $!
}

# Enhanced profile_llm_inference function
profile_llm_inference() {
    local MMAX=$1
    local RESULTS_DIR="result_dp_${MMAX}"
    
    mkdir -p "$RESULTS_DIR"
    log "PROFILE" "Starting profiling with memory limit: $MMAX"
    
    for ((iteration=0; iteration<=NUM_REPEATS; iteration++)); do
        log "ITERATION" "========== Iteration $iteration/$NUM_REPEATS =========="
        prompt_id=1
        
        while IFS=',' read -r token_count prompt || [[ -n "$token_count" ]]; do
            prompt=$(echo "$prompt" | tr -d '"')
            timestamp=$(date +%Y%m%d_%H%M%S)
            
            # Prepare output files
            OUTPUT_FILE="${RESULTS_DIR}/output_${token_count}_${prompt_id}.txt"
            CSV_FILE="${RESULTS_DIR}/memusage_${token_count}_${prompt_id}.csv"
            STRACE_FILE="${RESULTS_DIR}/strace_${token_count}_${prompt_id}.txt"
            PERF_FILE="${RESULTS_DIR}/perf_${token_count}_${prompt_id}.data"
            IO_STATS_FILE="${RESULTS_DIR}/io_stats_${token_count}_${prompt_id}.csv"
            
            log "PROMPT" "Processing prompt $prompt_id with token count $token_count"
            
            # Clear caches before each run
            clear_caches
            
            # Try to start perf recording, but continue if it fails
            if perf record --version >/dev/null 2>&1; then
                log "PERF" "Starting perf recording"
                perf record -o "$PERF_FILE" -e cpu-clock,page-faults,major-faults,minor-faults,cache-misses,cache-references -p $ &
                PERF_PID=$!
            else
                log "WARNING" "Perf recording skipped - continuing with other monitoring"
                PERF_PID=""
            fi
            
            # Launch main process with memory limit and monitoring
            nohup systemd-run --scope -p MemoryMax=$MMAX ./text_generator_main \
                --tflite_model="${MODEL_PATH}/${MODEL_NAME}.tflite" \
                --sentencepiece_model="${MODEL_PATH}/tokenizer.model" \
                --max_decode_steps=32 \
                --start_token="<bos>" \
                --stop_token="<eos>" \
                --num_threads=8 \
                --prompt="$prompt" \
                --weight_cache_path="${MODEL_PATH}/${MODEL_NAME}.xnnpack_cache" > "$OUTPUT_FILE" 2>&1 &
            
            PID=$!
            log "PROCESS" "Started main process with PID: $PID"
            
            # Setup tracing
            setup_tracing "$PID"
            
            # Start monitoring
            IO_MONITOR_PID=$(monitor_io "$PID" "$IO_STATS_FILE")
            MEMORY_MONITOR_PID=$(monitor_memory_advanced "$PID" "$CSV_FILE" "$STRACE_FILE")
            
            

            log "Started monitoring processes - Memory: $MEMORY_MONITOR_PID, I/O: $IO_MONITOR_PID"

            # Wait for main process to complete
            while kill -0 $PID 2>/dev/null; do
                sleep 1
            done
            log "Main process completed"

            # Cleanup monitoring processes
            kill $MEMORY_MONITOR_PID 2>/dev/null || true
            kill $IO_MONITOR_PID 2>/dev/null || true
            
            # Process trace data if available
            if [[ -f "/sys/kernel/tracing/trace" ]]; then
                log "Processing trace data..."
                cp "/sys/kernel/tracing/trace" "$TRACE_FILE" 2>/dev/null || true
            fi

            log "Completed processing for prompt $prompt_id"
            log "----------------------------------------"

            # # Wait for completion
            # wait $PID || true
            
            # # Stop monitoring and wait for processes to finish
            # log "MONITOR" "Stopping monitoring processes..."
            
            # if [[ -n "$MEMORY_MONITOR_PID" ]]; then
            #     kill $MEMORY_MONITOR_PID 2>/dev/null || true
            #     wait $MEMORY_MONITOR_PID 2>/dev/null || true
            #     unset MEMORY_MONITOR_PID
            # fi
            
            # if [[ -n "$IO_MONITOR_PID" ]]; then
            #     kill $IO_MONITOR_PID 2>/dev/null || true
            #     wait $IO_MONITOR_PID 2>/dev/null || true
            #     unset IO_MONITOR_PID
            # fi
            
            # if [[ -n "$PERF_PID" ]]; then
            #     kill $PERF_PID 2>/dev/null || true
            #     wait $PERF_PID 2>/dev/null || true
            #     unset PERF_PID
            # fi
            
            # log "MONITOR" "All monitoring processes stopped"
            
            # Process perf results if perf was running
            if [[ -n "$PERF_PID" ]] && [[ -f "$PERF_FILE" ]]; then
                if ! perf report -i "$PERF_FILE" > "${PERF_FILE%.data}_report.txt" 2>/dev/null; then
                    log "WARNING" "Failed to generate perf report - continuing"
                fi
            fi
            
            log "COMPLETE" "Finished processing prompt $prompt_id"
            
            prompt_id=$((prompt_id + 1))
            if (( prompt_id > PROMPT_ITEM_SIZE )); then
                prompt_id=1
            fi
            
        done < "$PROMPT_FILE"
    done
}

# Main execution
main() {
    log "START" "Initializing profiling script"
    
    # Validate prerequisites
    for cmd in strace systemd-run; do
        if ! command -v "$cmd" &> /dev/null; then
            log "ERROR" "Required tool $cmd is not installed"
            exit 1
        fi
    done
    
    # Check for optional tools
    for cmd in blktrace iotop; do
        if ! command -v "$cmd" &> /dev/null; then
            log "WARNING" "Tool $cmd is not installed - some monitoring features will be limited"
        fi
    done
    
    # Check input files and permissions
    if [[ ! -f "$PROMPT_FILE" ]]; then
        log "ERROR" "Prompt file '$PROMPT_FILE' does not exist"
        exit 1
    fi
    
    if [[ ! -x "./text_generator_main" ]]; then
        log "ERROR" "text_generator_main is not executable or missing"
        exit 1
    fi
    
    if [[ $EUID -ne 0 ]]; then
        log "WARNING" "Script not running as root - some features may be limited"
    fi
    
    # Run profiling for each memory configuration
    for MMAX in "${CGROUP_MMAX[@]}"; do
        log "CONFIG" "Starting profiling session with memory limit: $MMAX"
        profile_llm_inference "$MMAX"
    done
    
    log "COMPLETE" "All profiling sessions completed successfully"
}

# Start the script
main # "$@" 2>&1 | tee -a "$LOG_FILE"
