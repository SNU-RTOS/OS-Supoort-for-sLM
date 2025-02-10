# import pandas as pd
# import plotly.graph_objects as go
# import os
# import re
# import sys
# from collections import defaultdict

# def load_files(directory, regex_pattern):
#     """
#     디렉토리 내에서 정규표현식에 맞는 파일들을 읽어오는 함수.

#     Args:
#     - directory (str): 탐색할 디렉토리 경로
#     - regex_pattern (str): 파일명을 필터링할 정규표현식

#     Returns:
#     - List[pd.DataFrame]: 로드된 데이터프레임의 리스트
#     """

#     all_data = []
#     regex = re.compile(regex_pattern)
#     matched_files = [f for f in os.listdir(directory) if regex.match(f)]

#     if not matched_files:
#         print(f"No matching files found in directory: {directory}")
#         return None

#     for file in matched_files:
#         if regex.match(file):
#             filepath = os.path.join(directory, file)
#             print(f"Loading file: {filepath}")
#             df = pd.read_csv(filepath)
#             all_data.append(df)

#     return all_data


# def group_metrics_by_context_length(all_data):
#     """
#     파일 이름에서 컨텍스트 길이를 추출하여, 해당 길이별로 메트릭 데이터를 그룹핑합니다.

#     파일 이름 형식: memusage_<컨텍스트길이>_<데이터번호>.txt
#     """
#     grouped_data = defaultdict(list)
#     for file_path, metrics in all_data.items():
#         basename = os.path.basename(file_path)
#         match = re.search(r"memusage_(\d+)_\d+\.txt", basename)
#         if match:
#             context_length = int(match.group(1))
#             grouped_data[context_length].append(metrics)
#         else:
#             print(f"Warning: 파일 이름에서 컨텍스트 길이를 추출할 수 없음: {basename}")
#     return grouped_data

# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python script.py [directory] [filename_regex]")
#         print('예시: python script.py /path/to/output "^memusage_\\d_\\d+\\.csv$"')
#         sys.exit(1)

#     # 📂 사용자 입력
#     directory = sys.argv[1]
#     regex_pattern = sys.argv[2]

#     # 파일 읽기
#     all_data = load_files(directory, regex_pattern)

#     grouped_data = group_metrics_by_context_length(all_data)

#     # 📊 데이터 평균 계산
#     if not all_data:
#         print("No files matched the pattern. Please check the directory and regex pattern.")
#         exit()

#     average_data = pd.concat(all_data).groupby("Time (s)").mean().reset_index()

#     # 메모리 지표
#     memory_metrics = ["VmRSS (KB)", "VmSize (KB)", "VmSwap (KB)", "RssAnon (KB)", "RssFile (KB)"]


import pandas as pd
import plotly.graph_objects as go
import os
import re
import sys
from collections import defaultdict


def load_files(directory, regex_pattern):
    """
    디렉토리 내에서 정규표현식에 맞는 파일들을 읽어오는 함수.

    Args:
    - directory (str): 탐색할 디렉토리 경로
    - regex_pattern (str): 파일명을 필터링할 정규표현식

    Returns:
    - dict: {input context length: List[pd.DataFrame]}
    """
    grouped_data = defaultdict(list)
    regex = re.compile(regex_pattern)
    files = [f for f in os.listdir(directory) if regex.match(f)]

    if not files:
        print(f"No matching files found in directory: {directory}")
        return None
    for file in files:
        match = regex.match(file)
        if match:
            input_context_length = int(file.split("_")[1])  # 첫 번째 숫자 추출
            filepath = os.path.join(directory, file)
            df = pd.read_csv(filepath)
            grouped_data[input_context_length].append(df)
    
    # sort by context length
    grouped_data = dict(sorted(grouped_data.items())) 
    
    return grouped_data


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py [directory] [filename_regex]")
        print('예시: python script.py /path/to/output "memusage_\\d_\\d+\\.csv$"')
        sys.exit(1)

    directory = sys.argv[1]
    regex_pattern = sys.argv[2]
    grouped_data = load_files(directory, regex_pattern)

    if not grouped_data:
        print(
            "No files matched the pattern. Please check the directory and regex pattern."
        )
        exit()

    memory_metrics = [
        "VmRSS (KB)",
        "VmSize (KB)",
        "VmSwap (KB)",
        "RssAnon (KB)",
        "RssFile (KB)",
    ]
    tableau_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]

    for context_length, dataframes in grouped_data.items():
        # 🚀 Plotly 그래프 생성
        fig = go.Figure()

        # 모든 데이터 백그라운드에 투명하게 표시
        for idx, df in enumerate(dataframes):
            for i, metric in enumerate(memory_metrics):
                fig.add_trace(
                    go.Scatter(
                        x=df["Time (s)"],
                        y=df[metric] / 1024,
                        mode="lines",
                        line=dict(color=tableau_colors[i], width=0.5),
                        name=f"{metric} (File {idx+1})",
                        showlegend=False,
                        opacity=0.2,
                    )
                )

        # 평균 데이터는 진하게 표시
        average_data = pd.concat(dataframes).groupby("Time (s)").mean().reset_index()

        for i, metric in enumerate(memory_metrics):
            fig.add_trace(
                go.Scatter(
                    x=average_data["Time (s)"],
                    y=average_data[metric] / 1024,
                    mode="lines+markers",
                    line=dict(color=tableau_colors[i], width=2),
                    name=f"Avg {metric}",
                    marker=dict(size=5),
                )
            )

        # 그래프 설정
        fig.update_layout(
            title=f"Memory Usage Over Time (Context Length {context_length})",
            xaxis_title="Time (s)",
            yaxis_title="Memory (MB)",
            legend_title="Memory Type",
            template="plotly_white",
            xaxis=dict(showgrid=True, gridcolor="lightgray", dtick=5),
            yaxis=dict(showgrid=True, gridcolor="lightgray", dtick=500),
        )

        # 📌 그래프 출력
        fig.show()
        # 🖼️ PNG 파일로 저장
        output_png_file = f"memory_usage_context_{context_length}.png"
        fig.write_image(output_png_file, width=1200, height=600)
        print(f"✅ Plot saved as {output_png_file}")


#     # 전문적인 색상 팔레트 (Tableau 10)
#     tableau_colors = [
#         "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
#         "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
#     ]
