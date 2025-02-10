#!/bin/bash

# python3 memory-analyze.py  /home/geonha/workspace/ws_DS_NPU/results-3/0207_dp/memusage "^memusage_\\d+_\\d+\\.csv$"
# python3 output_analyze_compare.py /home/geonha/workspace/ws_DS_NPU/results-3/0207_dp/output "^output_\\d+_\\d+\\.txt$"
# python3 perf-analyze.py  /home/geonha/workspace/ws_DS_NPU/results-3/0207_dp/ "^perf_\\d+_\\d+\\.data$"


# python3 memory-analyze.py  /home/geonha/workspace/ws_DS_NPU/results-4/0208_dp_cg_2GB   "^memusage_\\d+_\\d+\\.csv$"
# python3 output-analyze.py /home/geonha/workspace/ws_DS_NPU/results-4/0208_dp_cg_2GB   "^output_\\d+_\\d+\\.txt$"
# python3 perf-analyze.py  /home/geonha/workspace/ws_DS_NPU/results-4/0208_dp_cg_2GB   "^perf_\\d+_\\d+\\.data$"





python3 output_analyze_compare_TTFT.py  /home/geonha/workspace/ws_DS_NPU/results-4/0208_dp_cg_2GB  /home/geonha/workspace/ws_DS_NPU/results-3/0207_dp/ "^output_\\d+_\\d+\\.txt$"
python3 output_analyze_compare_decoding.py  /home/geonha/workspace/ws_DS_NPU/results-4/0208_dp_cg_2GB  /home/geonha/workspace/ws_DS_NPU/results-3/0207_dp/ "^output_\\d+_\\d+\\.txt$"
python3 perf_analyze_compare.py  /home/geonha/workspace/ws_DS_NPU/results-4/0208_dp_cg_2GB  /home/geonha/workspace/ws_DS_NPU/results-3/0207_dp/ "^perf_\\d+_\\d+\\.data$"
