import re
import subprocess
import sys
import os
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
import plotly.offline as py


def parse_perf_report_multi_event_counts(report_text):
    """
    perf report --stdio 텍스트 결과에서 여러 이벤트별 횟수를 파싱합니다.
    예시:
      # Samples: 2K of event 'major-faults'
      # Event count (approx.): 2408
    """
    all_event_metrics = {}
    current_event_name = None

    lines = report_text.strip().split("\n")
    for line in lines:
        # 예: "# Samples: 2K of event 'major-faults'"
        samples_match = re.match(
            r"# Samples:\s*([\d\.]+)(\S*?)\s*of event\s*'([^']+)'", line
        )
        if samples_match:
            samples_count_str = samples_match.group(1)
            suffix = samples_match.group(2)  # 'K' 등 임의 문자열
            event_name = samples_match.group(3)
            current_event_name = event_name  # 현재 이벤트 이름 저장

            try:
                samples_count_float = float(samples_count_str)
            except ValueError:
                print(
                    f"Warning: Could not convert samples count to float: {samples_count_str}"
                )
                samples_count_float = None

            if samples_count_float is not None:
                if suffix == "K":
                    samples_count_int = int(samples_count_float * 1000)
                else:
                    samples_count_int = int(samples_count_float)
            else:
                samples_count_int = None

            all_event_metrics[event_name] = {
                "samples_count_str": samples_count_str + suffix,
                "samples_count_int": samples_count_int,
                "event_count": None,  # 아직 파싱되지 않은 상태
            }
            continue  # 다음 줄 처리

        # 예: "# Event count (approx.): 2408"
        event_count_match = re.match(r"# Event count \(approx.\):\s*(\d+)", line)
        if event_count_match and current_event_name:
            try:
                event_count = int(event_count_match.group(1))
            except ValueError:
                print(
                    f"Warning: Could not convert event count to integer: {event_count_match.group(1)}"
                )
                event_count = None

            if current_event_name in all_event_metrics:
                all_event_metrics[current_event_name]["event_count"] = event_count
            else:
                print(
                    f"Warning: Event count line found before Samples line for event: {current_event_name}"
                )
            current_event_name = None  # 초기화
            continue
        elif event_count_match and not current_event_name:
            # "# Event count (approx.):"가 나왔는데, 아직 event_name이 없는 경우
            print(
                f"Warning: Event count line found before Samples line, ignoring: {line}"
            )
            continue

    return all_event_metrics


def calculate_statistics_grouped_by_context_length(grouped_event_data):
    """
    grouped_event_data[context_length] = {
        'on': [ {event_name: {...}, ...}, {event_name: {...}, ...}, ... ],
        'off': [ {...}, {...}, ... ]
    }
    """
    grouped_statistics_results = {}

    for context_length, data_dict in grouped_event_data.items():
        # data_dict: {'on': [...], 'off': [...]}
        stats_dict = {}
        # on/off 각각에 대해 이벤트 통계 계산
        for cg_status, event_counts_list in data_dict.items():
            event_stats = defaultdict(list)
            for event_counts in event_counts_list:
                if not event_counts:
                    continue
                for event_name, counts in event_counts.items():
                    event_count = counts.get("event_count")
                    if event_count is not None:
                        event_stats[event_name].append(event_count)

            # 각 이벤트에 대해 평균, 표준편차, 사분위수 계산
            status_results = {}
            for event_name, counts_list in event_stats.items():
                if not counts_list:
                    status_results[event_name] = None
                    continue
                average_count = np.mean(counts_list)
                std_dev_count = np.std(counts_list)
                quartiles_count = np.percentile(counts_list, [25, 50, 75]).tolist()
                status_results[event_name] = {
                    "average_event_count": average_count,
                    "std_dev_event_count": std_dev_count,
                    "quartiles_event_count": quartiles_count,
                    "raw_event_counts": counts_list,
                }
            stats_dict[cg_status] = status_results
        grouped_statistics_results[context_length] = stats_dict

    return grouped_statistics_results


def visualize_event_counts_side_by_side(grouped_statistics_results, output_dir="."):
    """
    이벤트별로 컨텍스트 길이에 따라 cgroup on/off의 평균 event count를 2개의 막대로 표시
    → barmode='group' 형태
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 컨텍스트 길이 목록
    # context_lengths = sorted(grouped_statistics_results.keys())
    context_lengths = [8, 128, 512]
    # 모든 이벤트 이름 모으기
    all_events = set()
    for cl in context_lengths:
        for cg_status in ("on", "off"):
            stats_dict = grouped_statistics_results[cl].get(cg_status, {})
            if stats_dict:
                all_events.update(stats_dict.keys())
    all_events = sorted(list(all_events))

    # 색상 팔레트
    # cgroup on / off 를 구분하기 위한 색상 2개
    color_cg_on = "#4E79A7"  # (파란색 계열)
    color_cg_off = "#F28E2B"  # (주황색 계열)

    all_events = ["major-faults"]
    for event_name in all_events:
        fig = go.Figure()
        x_labels = []  # context length list
        on_values = []
        off_values = []

        for cl in context_lengths:

            # cg_on
            cg_on_stats = grouped_statistics_results[cl].get("on", {})
            cg_on_event = cg_on_stats.get(event_name)
            if cg_on_event is not None:
                on_values.append(cg_on_event["average_event_count"])
            else:
                on_values.append(None)

            # cg_off
            cg_off_stats = grouped_statistics_results[cl].get("off", {})
            cg_off_event = cg_off_stats.get(event_name)
            if cg_off_event is not None:
                off_values.append(cg_off_event["average_event_count"])
            else:
                off_values.append(None)

            x_labels.append(f"{cl}")

        # 막대 그래프 추가
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=on_values,
                name="RAM 2GB",
                marker=dict(color=color_cg_on, line=dict(color="black", width=1)),
            )
        )
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=off_values,
                name="RAM 4GB",
                marker=dict(color=color_cg_off, line=dict(color="black", width=1)),
            )
        )

        fig.update_layout(
            title=f"<b>Average {event_name.capitalize()} Count by Input Token Length </b>",
            xaxis_title="Input Token Length [token]",
            yaxis_title=f"Average {event_name.capitalize()} Count [count]",
            barmode="group",  # 나란히 막대 그래프
            font=dict(family="Noto Sans, sans-serif", size=28, color="black"),
            plot_bgcolor="white",
            xaxis=dict(
                showgrid=True,
                gridcolor="lightgray",
                gridwidth=0.5,
                type="category",
                zeroline=True,
                zerolinecolor="lightgray",
            ),
            yaxis=dict(
                showgrid=True,
                title_standoff=40,
                gridcolor="lightgray",
                gridwidth=0.5,
                zeroline=True,
                zerolinecolor="lightgray",
                rangemode="tozero",
                # dtick=50,
            ),
            bargroupgap=0.1,  # 그룹 간 간격 (기본값 0.2~0.3),
            bargap=0.2,  # 막대 사이의 간격 (기본값 0.1~0.2),
            showlegend=True,
        )

        # 화면 출력 + 이미지 저장
        py.offline.iplot(fig)
        filename = f"{output_dir}/event_{event_name}_compare_on_off.png"
        fig.write_image(filename)
        print(f"✅ Saved: {filename}")


if __name__ == "__main__":
    # 인자: cgroup_on 디렉토리, cgroup_off 디렉토리, 파일 정규표현식
    if len(sys.argv) < 4:
        print(
            "Usage: python perf-analyze.py [cgroup_on_dir] [cgroup_off_dir] [파일정규표현식]"
        )
        sys.exit(1)

    cg_on_dir = sys.argv[1]
    cg_off_dir = sys.argv[2]
    file_regex = sys.argv[3]

    # 파일 목록 구성
    # (1) cgroup on
    try:
        pattern = re.compile(file_regex)
    except re.error as e:
        print(f"Error compiling regex: {e}")
        sys.exit(1)

    def find_files(directory):
        if not os.path.isdir(directory):
            return []
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if pattern.match(f)
        ]

    on_files = find_files(cg_on_dir)
    off_files = find_files(cg_off_dir)

    if not on_files and not off_files:
        print("No matching perf.data files found in both directories.")
        sys.exit(1)

    # grouped_event_data[context_length]['on' or 'off'] = [ { event -> counts }, ... ]
    grouped_event_data = defaultdict(lambda: {"on": [], "off": []})

    def process_perf_files(files, cg_status):
        for perf_data_file in files:
            # 파일 이름에서 컨텍스트 길이를 추출
            basename = os.path.basename(perf_data_file)
            match = re.search(r"perf_(\d+)_(\d+)\.data", basename)
            if match:
                context_length = int(match.group(1))
            else:
                print(f"Warning: cannot parse context length from filename: {basename}")
                continue

            # perf report 실행
            command = ["perf", "report", "-i", perf_data_file, "--stdio"]
            print(f"Processing {perf_data_file}...", end="\r", flush=True)
            try:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate(timeout=15)
                report_text = stdout.decode("utf-8")
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                report_text = stdout.decode("utf-8")
                print(f"perf report timed out: {perf_data_file}")
            except FileNotFoundError:
                print("Error: perf command not found.")
                report_text = ""
            except Exception as e:
                print(f"Error while running perf report: {e}")
                report_text = ""

            if report_text:
                parsed_event_counts = parse_perf_report_multi_event_counts(report_text)
                grouped_event_data[context_length][cg_status].append(
                    parsed_event_counts
                )

    # on/off 디렉토리 파일 처리
    process_perf_files(on_files, "on")
    process_perf_files(off_files, "off")

    # 통계 계산
    grouped_statistics_results = calculate_statistics_grouped_by_context_length(
        grouped_event_data
    )
    if not grouped_statistics_results:
        print("No statistic results to display.")
        sys.exit(1)

    # 통계 출력(옵션)
    print("\n[Stats by Context Length / cgroup on/off]")
    for cl in sorted(grouped_statistics_results.keys()):
        print(f"\n-- Context Length: {cl}")
        for cg_status in ("on", "off"):
            event_dict = grouped_statistics_results[cl].get(cg_status, {})
            print(f"  [CG: {cg_status}]")
            for event_name, stats in event_dict.items():
                if stats is not None:
                    avg = stats["average_event_count"]
                    std = stats["std_dev_event_count"]
                    quartiles = stats["quartiles_event_count"]
                    print(
                        f"    Event: {event_name} | Avg: {avg:.2f}, Std: {std:.2f}, Q25/50/75: {quartiles}"
                    )
                else:
                    print(f"    Event: {event_name} - No data")

    # 그래프 생성
    visualize_event_counts_side_by_side(grouped_statistics_results, output_dir="plot")
