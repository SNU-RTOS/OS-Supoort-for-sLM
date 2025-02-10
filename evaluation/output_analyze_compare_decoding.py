import os
import re
import sys
import numpy as np
import plotly.graph_objects as go
import plotly.offline as py
from collections import defaultdict

def parse_inference_output(file_path):
    """
    추론 결과 파일을 파싱하여 필요한 원본 측정치를 추출한 후,
    Avg Decoding Latency (sec/tokens) = [(Total Decoding) - (First Token)] / (Tokens - 1) / 1000
    을 계산하여 반환합니다.
    """
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            output_string = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None

    output_string = output_string.replace("\r", "")

    patterns = {
        "First Token Decoding Latency": r"\[METRICS\]\s*Time To First Token\s*:\s*([\d,]+(?:\.\d+)?)\s*ms",
        "Total Decoding Latency": r"\[METRICS\]\s*Total Decoding Latency\s*:\s*([\d,]+(?:\.\d+)?)\s*ms",
        "Total Number of Generated Tokens": r"\[METRICS\]\s*Total Number of Generated Tokens\s*:\s*(\d+)\s*tokens",
    }

    raw_metrics = {}
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, output_string)
        if match:
            try:
                value = float(match.group(1).replace(",", ""))
                raw_metrics[metric_name] = value
            except ValueError:
                print(f"Warning: Could not convert value for {metric_name} in {file_path} to number.")
                raw_metrics[metric_name] = None
        else:
            print(f"Warning: Could not find metric: {metric_name} in {file_path}")
            raw_metrics[metric_name] = None

    first_token = raw_metrics.get("First Token Decoding Latency")
    total_decoding = raw_metrics.get("Total Decoding Latency")
    total_tokens = raw_metrics.get("Total Number of Generated Tokens")

    if first_token is not None and total_decoding is not None and total_tokens is not None and total_tokens > 1:
        # ms → sec 변경
        ms_per_token = (total_decoding - first_token) / (total_tokens - 1)
        sec_per_token = ms_per_token / 1000.0
        return sec_per_token
    else:
        return None


def process_files_in_directory(directory_on, directory_no_on, filename_pattern):
    """
    두 디렉토리(cg_on, cg_no_on)에서 파일을 찾아,
    파일명에서 context length를 파싱한 뒤,
    각 context length 별로 on/off 데이터를 저장.
    결과 형태:
      all_metrics = {
         context_length1: {"on": [sec_val, ...], "no_on": [sec_val, ...]},
         context_length2: {...},
         ...
      }
    """
    all_metrics = defaultdict(lambda: {"on": [], "no_on": []})

    # directory_on → on / directory_no_on → no_on
    for cg_status, directory in zip(["on", "no_on"], [directory_on, directory_no_on]):
        if not os.path.isdir(directory):
            print(f"Error: Directory not found: {directory}")
            continue

        try:
            pattern = re.compile(filename_pattern)
        except re.error as e:
            print(f"Error compiling filename pattern: {e}")
            continue

        matched_files = [f for f in os.listdir(directory) if pattern.match(f)]
        if not matched_files:
            print(f"No matching files found in directory: {directory}")
            continue

        for filename in matched_files:
            file_path = os.path.join(directory, filename)
            # 예: output_256_12.txt → context_length=256
            match_cl = re.search(r"output_(\d+)_\d+\.txt", filename)
            if not match_cl:
                print(f"Warning: Could not parse context length from filename: {filename}")
                continue
            context_length = int(match_cl.group(1))

            # parse inference result
            metric_value = parse_inference_output(file_path)
            if metric_value is not None:
                all_metrics[context_length][cg_status].append(metric_value)

    return all_metrics


def visualize_metrics(grouped_data, output_dir="plot"):
    """
    컨텍스트 길이별로 on/off 평균 Latency(sec/tokens)를 막대그래프(barmode='group')로 보여줍니다.
    x축: context length
    y축: 평균 sec/tokens
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 각 context length에 대해 on/off 평균 계산
    context_length = sorted(grouped_data.keys())
    context_length = [8, 128, 512]
    if not context_length:
        print("No grouped data to visualize.")
        return

    x_values = []
    on_avgs = []
    off_avgs = []

    for cl in context_length:
        cg_on_list = grouped_data[cl]["on"]
        cg_off_list = grouped_data[cl]["no_on"]

        on_avg = np.mean(cg_on_list) if cg_on_list else 0
        off_avg = np.mean(cg_off_list) if cg_off_list else 0

        x_values.append(str(cl))
        on_avgs.append(on_avg)
        off_avgs.append(off_avg)

        print(f"[INFO] Context={cl}: cgroup_on Avg={on_avg:.6f} sec/token, cgroup_off Avg={off_avg:.6f} sec/token")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x_values,
        y=on_avgs,
        name="RAM 2GB",
        marker=dict(color="#4E79A7", line=dict(color="black", width=1))
    ))

    fig.add_trace(go.Bar(
        x=x_values,
        y=off_avgs,
        name="RAM 4GB",
        marker=dict(color="#F28E2B", line=dict(color="black", width=1))
    ))

    fig.update_layout(
        title=f"<b>Avg Inference Latency by Input Token Length [sec/token]</b>",
        xaxis_title="Input Token Length [token]",
        yaxis_title="Avg Inference Latency [sec/token]",
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
            rangemode='tozero',
        ),
        bargroupgap=0.1,
        bargap=0.2,
        showlegend=True,
        barmode="group"
    )

    # 화면에 표시 + 이미지 저장
    py.offline.iplot(fig)
    output_png = os.path.join(output_dir, "avg_decoding_latency_by_context.png")
    fig.write_image(output_png)
    print(f"✅ Plot saved as {output_png}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py [directory_cg_on] [directory_cg_no_on] [filename_regex]")
        print('예시: python script.py /path/to/cg_on /path/to/cg_no_on "output_\\d+_\\d+\\.txt"')
        sys.exit(1)

    directory_on = sys.argv[1]
    directory_no_on = sys.argv[2]
    filename_pattern = sys.argv[3]

    all_data = process_files_in_directory(directory_on, directory_no_on, filename_pattern)
    if not all_data:
        print("No data to display.")
        sys.exit(1)

    visualize_metrics(all_data, output_dir="plot")
