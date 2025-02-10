import re
import subprocess
import sys
import os
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
import plotly.offline as py  # plotly.offline 추가

def parse_perf_report_multi_event_counts(report_text):
    """
    perf report --stdio 텍스트 결과에서 여러 이벤트별 횟수를 파싱합니다.
    (생략: 기존 주석 그대로)
    """
    all_event_metrics = {}
    current_event_name = None

    lines = report_text.strip().split('\n')
    for line in lines:
        # 예: "# Samples: 2K of event 'major-faults'"
        samples_match = re.match(r"# Samples:\s*([\d\.]+)(\S*?)\s*of event\s*'([^']+)'", line)
        if samples_match:
            samples_count_str = samples_match.group(1)
            k_suffix = samples_match.group(2)  # K가 있으면 K_suffix는 "K", 없으면 ""
            event_name = samples_match.group(3)
            current_event_name = event_name  # 현재 이벤트 이름 저장

            try:
                samples_count_float = float(samples_count_str)
            except ValueError:
                print(f"Warning: Could not convert samples count to float: {samples_count_str}")
                samples_count_float = None

            if samples_count_float is not None:
                if k_suffix == "K":
                    samples_count_int = int(samples_count_float * 1000)
                else:
                    samples_count_int = int(samples_count_float)
            else:
                samples_count_int = None

            all_event_metrics[event_name] = {
                "samples_count_str": samples_count_str + k_suffix,
                "samples_count_int": samples_count_int,
                "event_count": None  # 아직 파싱되지 않은 상태
            }
            continue  # 다음 줄 처리

        # 예: "# Event count (approx.): 2408"
        event_count_match = re.match(r"# Event count \(approx.\):\s*(\d+)", line)
        if event_count_match and current_event_name:
            try:
                event_count = int(event_count_match.group(1))
            except ValueError:
                print(f"Warning: Could not convert event count to integer: {event_count_match.group(1)}")
                event_count = None

            if current_event_name in all_event_metrics:
                all_event_metrics[current_event_name]["event_count"] = event_count
            else:
                print(f"Warning: Event count line found before Samples line for event: {current_event_name}")
            current_event_name = None  # 초기화
            continue
        elif event_count_match and not current_event_name:
            print(f"Warning: Event count line found before Samples line, ignoring: {line}")
            continue

    return all_event_metrics

def calculate_statistics_grouped_by_context_length(grouped_event_data):
    """
    grouped_event_data: key는 컨텍스트 길이, 값은 해당 그룹의 파일들의 파싱 결과 (리스트)
    """
    grouped_statistics_results = {}

    for context_length, event_counts_list in grouped_event_data.items():
        event_stats = defaultdict(list)
        for event_counts in event_counts_list:
            if not event_counts:
                print(f"Warning: No event counts found in one of the files for context length: {context_length}")
                continue
            for event_name, counts in event_counts.items():
                event_count = counts.get("event_count")
                if event_count is not None:
                    event_stats[event_name].append(event_count)

        statistics_results = {}
        for event_name, counts_list in event_stats.items():
            if not counts_list:
                print(f"Warning: No valid event counts found for event: {event_name} in context length group {context_length}")
                statistics_results[event_name] = {
                    "average_event_count": None,
                    "std_dev_event_count": None,
                    "quartiles_event_count": [None, None, None],
                    "raw_event_counts": []
                }
                continue

            average_count = np.mean(counts_list)
            std_dev_count = np.std(counts_list)
            quartiles_count = np.percentile(counts_list, [25, 50, 75])

            statistics_results[event_name] = {
                "average_event_count": average_count,
                "std_dev_event_count": std_dev_count,
                "quartiles_event_count": quartiles_count.tolist(),
                "raw_event_counts": counts_list
            }
        grouped_statistics_results[context_length] = statistics_results

    return grouped_statistics_results

def visualize_event_counts_separately(grouped_statistics_results, output_dir="."):
    """
    이벤트별로 컨텍스트 길이에 따라 독립적인 그래프를 생성하여 저장합니다.
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    context_lengths = sorted(list(grouped_statistics_results.keys()))
    event_names = set()

    for context_length in context_lengths:
        event_names.update(grouped_statistics_results[context_length].keys())
    event_names = sorted(list(event_names))

    # 색상 팔레트 (Tableau 10)
    tableau_colors = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
    ]

    for event_name in event_names:
        bar_fig = go.Figure()
        box_fig = go.Figure()

        for i, context_length in enumerate(context_lengths):
            stats = grouped_statistics_results[context_length].get(event_name, {})
            avg_count = stats.get('average_event_count')
            raw_counts = stats.get('raw_event_counts')

            # 막대 그래프 (평균 이벤트 카운트)
            if avg_count is not None:
                bar_fig.add_trace(go.Bar(
                    x=[f"CL: {context_length}"],
                    y=[avg_count],
                    name=f"CL: {context_length}",
                    marker=dict(color=tableau_colors[i % len(tableau_colors)]),
                ))

            # 박스 플롯 (이벤트 카운트 분포)
            if raw_counts:
                box_fig.add_trace(go.Box(
                    y=raw_counts,
                    name=f"CL: {context_length}",
                    marker=dict(color=tableau_colors[i % len(tableau_colors)]),
                ))

        # 막대 그래프 업데이트 및 저장
        bar_fig.update_layout(
            title=f"<b>Average Event Count for {event_name}</b>",
            yaxis_title="Average Event Count",
            xaxis_title="Context Length",
            barmode="group",
            template="plotly_white"
        )
        event_name = event_name.replace("block:", "")
        # bar_output_file = os.path.join(output_dir, f"{event_name}_bar.html")
        py.offline.iplot(bar_fig)
        bar_fig.write_image(f"{output_dir}/event_{event_name}_bar.png")

        
        print(f"Generated plots for event: {event_name}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python perf-analyze.py [디렉토리 경로] [파일 정규표현식]")
        print('예시: python perf-analyze.py /home/geonha/workspace/ws_DS_NPU/results-3/0207_dp_mmap/perf "^perf_\\d+_\\d+\\.data$"')
        sys.exit(1)
    
    directory = sys.argv[1]
    file_regex = sys.argv[2]
    
    try:
        pattern = re.compile(file_regex)
    except re.error as e:
        print(f"Error compiling regex: {e}")
        sys.exit(1)

    # 지정한 디렉토리 내의 모든 파일 목록 중 정규표현식에 매칭되는 파일만 선택
    data_files = [os.path.join(directory, f) for f in os.listdir(directory) if pattern.match(f)]
    
    if not data_files:
        print("Error: 전달된 perf.data 파일이 없습니다.")
        sys.exit(1)

    grouped_event_data = defaultdict(list)

    for perf_data_file in data_files:
        # 파일 이름에서 컨텍스트 길이를 추출
        # 파일명 예: perf_256_50.data → 컨텍스트 길이: 256
        basename = os.path.basename(perf_data_file)
        match = re.search(r"perf_(\d+)_(\d+)\.data", basename)
        if match:
            context_length = int(match.group(1))
        else:
            print(f"Warning: 파일 이름에서 컨텍스트 길이를 추출할 수 없음: {perf_data_file}")
            continue

        # perf report 명령 실행
        command = ["perf", "report", "-i", perf_data_file, "--stdio"]
        print(f"perf report 실행: {perf_data_file}", end="\r", flush=True)
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=15)
            report_text = stdout.decode('utf-8')
            # error_text = stderr.decode('utf-8')  # 필요시 확인
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            report_text = stdout.decode('utf-8')
            print(f"perf report 타임아웃 (파일: {perf_data_file})")
        except FileNotFoundError:
            print(f"Error: perf 명령어를 찾을 수 없음 (파일: {perf_data_file}).")
            report_text = ""
        except Exception as e:
            print(f"perf report 실행 중 예외 (파일: {perf_data_file}): {e}")
            report_text = ""

        if report_text:
            parsed_event_counts = parse_perf_report_multi_event_counts(report_text)
            grouped_event_data[context_length].append(parsed_event_counts)
        else:
            print(f"Warning: {perf_data_file} 파일에서 perf report 결과가 없음.")

    if grouped_event_data:
        grouped_statistics_results = calculate_statistics_grouped_by_context_length(grouped_event_data)
        if grouped_statistics_results:
            print("\n컨텍스트 길이별 이벤트 통계 수치:")
            for context_length, event_stats in sorted(grouped_statistics_results.items()):
                print(f"\n--- 컨텍스트 길이: {context_length} ---")
                for event_name, stats in event_stats.items():
                    avg = stats.get('average_event_count')
                    std = stats.get('std_dev_event_count')
                    quartiles = stats.get('quartiles_event_count')
                    if avg is not None:
                        print(f"  이벤트: {event_name}")
                        print(f"    평균: {avg:.2f}")
                        print(f"    표준편차: {std:.2f}")
                        print(f"    사분위수 (25%, 50%, 75%): {quartiles}")
                    else:
                        print(f"  이벤트: {event_name} - 데이터 없음")
            visualize_event_counts_separately(
                grouped_statistics_results, 
                output_dir="plot"
            )
        else:
            print("통계 수치를 계산할 이벤트 데이터가 없습니다.")
    else:
        print("perf report 실행 결과가 있는 파일이 없습니다.")