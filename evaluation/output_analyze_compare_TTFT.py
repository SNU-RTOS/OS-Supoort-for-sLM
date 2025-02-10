import os
import re
import sys
import numpy as np
import plotly.graph_objects as go
import plotly.offline as py
import plotly.colors as colors
from collections import defaultdict

def parse_inference_output(file_path):
    """
    추론 결과 파일을 파싱하여 필요한 원본 측정치를 추출한 후,
    Time To First Token Latency (ms) = Prefill + First Token
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
        "Total Prefill Latency": r"\[INFO\]\s*Prefill Stage took\s*([\d,]+(?:\.\d+)?)\s*ms",
        "First Token Decoding Latency": r"\[METRICS\]\s*Time To First Token\s*:\s*([\d,]+(?:\.\d+)?)\s*ms",
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
    
    prefill = raw_metrics.get("Total Prefill Latency", None)
    first_token = raw_metrics.get("First Token Decoding Latency", None)
    
    if prefill is not None and first_token is not None:
        return (prefill + first_token) / 1000
    else:
        return None

def process_files_in_directory(directory_on, directory_no_on, filename_pattern):
    """
    정규표현식 패턴과 매칭되는 파일을 검색하고, Time To First Token Latency를 계산.
    """
    all_metrics = defaultdict(dict)
    
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
            metric_value = parse_inference_output(file_path)
            if metric_value is not None:
                match = re.search(r"output_(\d+)_(\d+)\.txt", filename)
                if match:
                    context_length = int(match.group(1))
                    all_metrics[context_length][cg_status] = metric_value
    
    return all_metrics

def visualize_metrics(grouped_data, dir="plot"):
    """
    Time To First Token Latency를 cg on과 cg no_on 비교하여 막대그래프를 그림.
    """
    if not grouped_data:
        print("No grouped data to visualize.")
        return
    
    context_labels = sorted(grouped_data.keys())
    context_labels = [8,128,512]
    
    cg_on_values = [grouped_data[cl].get("on", None) for cl in context_labels]
    cg_no_on_values = [grouped_data[cl].get("no_on", None) for cl in context_labels]
    
    print(f"[INFO] Time To First Token [ms] with cg_on : {cg_on_values}")
    print(f"[INFO] Time To First Token [ms] with cg_off : {cg_no_on_values}")

    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=context_labels,
        y=cg_on_values,
        name="RAM 2GB",
        marker=dict(color="#4E79A7", line=dict(color="black", width=1))
    ))
    
    fig.add_trace(go.Bar(
        x=context_labels,
        y=cg_no_on_values,
        name="RAM 4GB",
        marker=dict(color="#F28E2B", line=dict(color="black", width=1))
    ))
    
    
    fig.update_layout(
        title=f"<b>TTFT by Input Token Length</b>",
        xaxis_title="Input Token Length [token]",
        yaxis_title=f"Time To First Token Latency [sec]",
        barmode="group",
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
            dtick=5,
        ),
        bargroupgap=0.1,  # 그룹 간 간격 (기본값 0.2~0.3),
        bargap=0.2,  # 막대 사이의 간격 (기본값 0.1~0.2),
        showlegend=True, 
    )
    
    py.offline.iplot(fig)
    fig.write_image(f"{dir}/time_to_first_token_latency_comparison.png")
    print("✅ Plot saved as time_to_first_token_latency_comparison.png")

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
    
    visualize_metrics(all_data,dir="plot")
