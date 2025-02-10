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
    추론 결과 파일을 파싱하여
    1) Total Prefill Latency
    2) First Token Decoding Latency
    3) Total Decoding Latency
    4) Total Number of Generated Tokens
    를 얻은 뒤,

    파생 메트릭:
    A) Time To First Token Latency (ms) = Prefill + First Token
    B) Avg Decoding Latency (ms/tokens) = (Total Decoding Latency - First Token Decoding Latency) / (Total Number of Generated Tokens - 1)
    를 계산하여 딕셔너리로 반환합니다.
    """
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            output_string = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None

    # Windows 개행 제거
    output_string = output_string.replace("\r", "")

    # 4가지 원본 측정치 추출
    patterns = {
        "Total Prefill Latency": r"\[INFO\]\s*Prefill Stage took\s*([\d,]+(?:\.\d+)?)\s*ms",
        "First Token Decoding Latency": r"\[METRICS\]\s*Time To First Token\s*:\s*([\d,]+(?:\.\d+)?)\s*ms",
        "Total Decoding Latency": r"\[METRICS\]\s*Total Decoding Latency\s*:\s*([\d,]+(?:\.\d+)?)\s*ms",
        "Total Number of Generated Tokens": r"\[METRICS\]\s*Total Number of Generated Tokens\s*:\s*(\d+)\s*tokens",
        "Total Inference Latency": r"\[METRICS\]\s*Total Inference Latency\s*:\s*([\d,]+(?:\.\d+)?)\s*ms",
        "Total Sampling Latency": r"\[METRICS\]\s*Total Sampling Latency\s*:\s*([\d,]+(?:\.\d+)?)\s*ms",
    }

    # 임시로 결과를 저장할 딕셔너리
    raw_metrics = {}

    for metric_name, pattern in patterns.items():
        match = re.search(pattern, output_string)
        if match:
            try:
                # 쉼표 제거 후 float 변환
                value = float(match.group(1).replace(",", ""))
                raw_metrics[metric_name] = value
            except ValueError:
                print(
                    f"Warning: Could not convert value for {metric_name} in {file_path} to number."
                )
                raw_metrics[metric_name] = None
        else:
            print(f"Warning: Could not find metric: {metric_name} in {file_path}")
            print(f"  ➝ File path: {file_path}")
            raw_metrics[metric_name] = None

    # 파생 메트릭 계산
    derived_metrics = {}

    # 1) Time To First Token Latency (ms) = Prefill + First Token
    prefill = raw_metrics.get("Total Prefill Latency", None)
    first_token = raw_metrics.get("First Token Decoding Latency", None)
    if prefill is not None and first_token is not None:
        derived_metrics["Time To First Token Latency (ms)"] = prefill + first_token
    else:
        derived_metrics["Time To First Token Latency (ms)"] = None

    # 2) Avg Decoding Latency (ms/tokens)
    total_decoding = raw_metrics.get("Total Decoding Latency", None)
    if first_token is not None and total_decoding is not None:
        total_tokens = raw_metrics.get("Total Number of Generated Tokens", None)
        if total_tokens is not None and total_tokens > 1:
            derived_metrics["Avg Decoding Latency (ms/tokens)"] = (
                total_decoding - first_token
            ) / (total_tokens - 1)
        else:
            # 토큰이 1개 이하이면 계산 불가
            derived_metrics["Avg Decoding Latency (ms/tokens)"] = None
    else:
        derived_metrics["Avg Decoding Latency (ms/tokens)"] = None

    # 4) Avg Inference Latency (ms/tokens) = Total Inference Latency / Total Tokens
    total_inference = raw_metrics.get("Total Inference Latency", None)
    if total_inference is not None and total_tokens is not None and total_tokens > 0:
        derived_metrics["Avg inference Latency (ms/tokens)"] = (
            total_inference / total_tokens
        )
    else:
        derived_metrics["Avg inference Latency (ms/tokens)"] = None

    # 4) Avg Sampling Latency (ms/tokens) = Total Sampling Latency / Total Tokens
    total_sampling = raw_metrics.get("Total Sampling Latency", None)
    if total_sampling is not None and total_tokens is not None and total_tokens > 0:
        derived_metrics["Avg Sampling Latency (ms/tokens)"] = (
            total_sampling / total_tokens
        )
    else:
        derived_metrics["Avg Sampling Latency (ms/tokens)"] = None

    return derived_metrics


def process_files_in_directory(directory, filename_pattern):
    """
    주어진 디렉토리에서 정규표현식 패턴과 매칭되는 파일을 찾아서 처리합니다.

    Args:
        directory (str): 검색할 디렉토리 경로
        filename_pattern (str): 파일 이름 정규표현식 패턴 (예: r"output_\d+_\d+\.txt")

    Returns:
        dict: {파일경로: 추출한 메트릭 딕셔너리}
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return None

    try:
        pattern = re.compile(filename_pattern)
    except re.error as e:
        print(f"Error compiling filename pattern: {e}")
        return None

    all_metrics = {}
    matched_files = [f for f in os.listdir(directory) if pattern.match(f)]

    if not matched_files:
        print(f"No matching files found in directory: {directory}")
        return None

    for filename in matched_files:
        file_path = os.path.join(directory, filename)
        metrics = parse_inference_output(file_path)
        if metrics is not None:
            all_metrics[file_path] = metrics

    return all_metrics


def group_metrics_by_context_length(all_data):
    """
    파일 이름에서 컨텍스트 길이를 추출하여, 해당 길이별로 메트릭 데이터를 그룹핑합니다.

    파일 이름 형식: output_<컨텍스트길이>_<데이터번호>.txt
    """
    grouped_data = defaultdict(list)
    for file_path, metrics in all_data.items():
        basename = os.path.basename(file_path)
        match = re.search(r"output_(\d+)_\d+\.txt", basename)
        if match:
            context_length = int(match.group(1))
            grouped_data[context_length].append(metrics)
        else:
            print(f"Warning: 파일 이름에서 컨텍스트 길이를 추출할 수 없음: {basename}")
    return grouped_data


def calculate_statistics_grouped_by_context_length(grouped_data):
    """
    각 컨텍스트 길이 그룹 내의 메트릭에 대해 통계량(평균, 표준편차, 중앙값, 사분위수 등)을 계산합니다.

    Args:
        grouped_data (dict): {컨텍스트길이: [메트릭 딕셔너리, ...]}

    Returns:
        dict: {컨텍스트길이: {메트릭명: 통계량 딕셔너리, ...}, ...}
    """
    grouped_stats = {}
    for context_length, metrics_list in grouped_data.items():
        group_stats = {}
        if not metrics_list:
            continue
        # 첫 번째 파일의 키를 기준으로 메트릭 종류 결정
        metric_keys = metrics_list[0].keys()
        for metric in metric_keys:
            values = [m[metric] for m in metrics_list if m[metric] is not None]
            if values:
                np_values = np.array(values)
                group_stats[metric] = {
                    "mean": np.mean(np_values),
                    "std": np.std(np_values),
                    "median": np.median(np_values),
                    "q25": np.percentile(np_values, 25),
                    "q75": np.percentile(np_values, 75),
                    "min": np.min(np_values),
                    "max": np.max(np_values),
                    "count": len(np_values),
                }
            else:
                group_stats[metric] = None
        grouped_stats[context_length] = group_stats
    return grouped_stats


def visualize_metrics(grouped_data):
    """
    두 메트릭(Time To First Token Latency, Avg Decoding Latency)에 대해 각각
    박스플롯을 생성하고, x축에 Context Length를 두는 방식으로 시각화합니다.

    - y축과 x축에 격자선(grid) 추가
    """
    if not grouped_data:
        print("No grouped data to visualize.")
        return

    # 사용된 모든 메트릭 추출
    metrics = None
    for cl, metrics_list in grouped_data.items():
        if metrics_list:
            metrics = list(metrics_list[0].keys())
            break
    if not metrics:
        print("No metric keys found in grouped data.")
        return

    # 색상 팔레트 (Tableau 10)
    tableau_colors = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]

    color_palate = {
        "bars": ["#F9C74F", "#90BE6D", "#43AA8B", "#4D908E", "#577590"],
        # "lines": ["#F4A261", "#E76F51", "#2A9D8F", "#264653", "#8AB17D"],
    }

    # 컨텍스트 길이별 데이터를 추가
    sorted_contexts = sorted(grouped_data.keys())

    ##############################################
    metric = metrics[0]
    fig = go.Figure()
    for idx, cl in enumerate(sorted_contexts):
        values = [m[metric] / 1000 for m in grouped_data[cl] if m[metric] is not None]

        avg_value = np.mean(values) if values else 0
        fig.add_trace(
            go.Bar(
                x=[f"{cl}"],  # X축: 컨텍스트 길이
                y=[avg_value],  # Y축: 평균값
                name=f"CL:{cl}",
                marker=dict(
                    line=dict(color="black", width=1),
                    color=color_palate["bars"][2],
                ),  # 테두리
                # marker_color=tableau_colors[idx % len(tableau_colors)],  # 색상
                # 단일 색상
                # marker_color="darkgray",
                #      aliceblue, antiquewhite, aqua, aquamarine, azure,
                # beige, bisque, black, blanchedalmond, blue,
                # blueviolet, brown, burlywood, cadetblue,
                # chartreuse, chocolate, coral, cornflowerblue,
                # cornsilk, crimson, cyan, darkblue, darkcyan,
                # darkgoldenrod, darkgray, darkgrey, darkgreen,
                # darkkhaki, darkmagenta, darkolivegreen, darkorange,
                # darkorchid, darkred, darksalmon, darkseagreen,
                # darkslateblue, darkslategray, darkslategrey,
                # darkturquoise, darkviolet, deeppink, deepskyblue,
                # dimgray, dimgrey, dodgerblue, firebrick,
                # floralwhite, forestgreen, fuchsia, gainsboro,
                # ghostwhite, gold, goldenrod, gray, grey, green,
                # greenyellow, honeydew, hotpink, indianred, indigo,
                # ivory, khaki, lavender, lavenderblush, lawngreen,
                # lemonchiffon, lightblue, lightcoral, lightcyan,
                # lightgoldenrodyellow, lightgray, lightgrey,
                # lightgreen, lightpink, lightsalmon, lightseagreen,
                # lightskyblue, lightslategray, lightslategrey,
                # lightsteelblue, lightyellow, lime, limegreen,
                # linen, magenta, maroon, mediumaquamarine,
                # mediumblue, mediumorchid, mediumpurple,
                # mediumseagreen, mediumslateblue, mediumspringgreen,
                # mediumturquoise, mediumvioletred, midnightblue,
                # mintcream, mistyrose, moccasin, navajowhite, navy,
                # oldlace, olive, olivedrab, orange, orangered,
                # orchid, palegoldenrod, palegreen, paleturquoise,
                # palevioletred, papayawhip, peachpuff, peru, pink,
                # plum, powderblue, purple, red, rosybrown,
                # royalblue, rebeccapurple, saddlebrown, salmon,
                # sandybrown, seagreen, seashell, sienna, silver,
                # skyblue, slateblue, slategray, slategrey, snow,
                # springgreen, steelblue, tan, teal, thistle, tomato,
                # turquoise, violet, wheat, white, whitesmoke,
                # yellow, yellowgreen
            )
        )
    fig.update_layout(
        # title=f"<b>{metric} by Context Length</b>",
        xaxis_title="Input Token Length ",
        yaxis_title=f"Time To First Token Latency [sec]",
        barmode="group",
        plot_bgcolor="white",
        font=dict(family="Noto Sans", size=18, color="black"),
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
            type="category",  # X축 눈금을 문자열로 표시
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
            dtick=2,  # Y축 눈금을 10 단위로 설정 (더 촘촘하게)
        ),
        bargroupgap=0.5,  # 그룹 간 간격 (기본값 0.2~0.3),
        bargap=0.2,  # 막대 사이의 간격 (기본값 0.1~0.2),
        showlegend=False,  # 범례 제거
    )
    print(f"Showing plot for metric: {metric}")
    py.offline.iplot(fig)
    fig.write_image(f"1.png")
    

    ##############################################
    # metric = metrics[1]

    # fig = go.Figure()

    # for idx, cl in enumerate(sorted_contexts):
    #     values = [m[metric] for m in grouped_data[cl] if m[metric] is not None]
    #     if values:
    #         fig.add_trace(
    #             go.Box(
    #                 y=values,
    #                 name=cl,
    #                 marker=dict(
    #                     line=dict(
    #                         color="black",
    #                         width=1,
    #                     ),
    #                     color=color_palate["bars"][idx % len(color_palate["bars"])],
    #                 ),
    #                 marker_color=tableau_colors[idx % len(tableau_colors)],
    #                 boxmean=True,
    #                 boxpoints=False,
    #                 showlegend=True,
    #             )
    #         )

    # fig.update_layout(
    #     xaxis_title="Input Token Length ",
    #     yaxis_title=f"Avg Decoding Latency (sec)",
    #     barmode="group",
    #     plot_bgcolor="white",
    #     font=dict(family="Noto Sans, sans-serif", size=18, color="black"),
    #     xaxis=dict(
    #         showgrid=True,
    #         gridcolor="lightgray",
    #         gridwidth=0.5,
    #         type="category",  # X축 눈금을 문자열로 표시
    #     ),
    #     yaxis=dict(
    #         showgrid=True,
    #         gridcolor="lightgray",
    #         gridwidth=0.5,
    #         dtick=2,  # Y축 눈금을 10 단위로 설정 (더 촘촘하게)
    #     ),
    #     bargroupgap=0.5,  # 그룹 간 간격 (기본값 0.2~0.3),
    #     bargap=0.2,  # 막대 사이의 간격 (기본값 0.1~0.2),
    #     showlegend=False,  # 범례 제거
    # )

    # py.offline.iplot(fig)
    # fig.write_image(f"2.png")
    

    ################
    metrics_to_plot = [metrics[1], metrics[2], metrics[3]]

    # Input Context 길이별로 데이터를 저장할 딕셔너리
    grouped_bar_data = {metric: [] for metric in metrics_to_plot}
    context_labels = sorted(grouped_data.keys())  # 정렬된 Context 길이 리스트

    # 각 Input Context 길이별 데이터 추출
    for metric in metrics_to_plot:
        for cl in context_labels:
            values = [m[metric] for m in grouped_data[cl] if m[metric] is not None]
            avg_value = np.mean(values) if values else 0  # 평균값 계산
            grouped_bar_data[metric].append(avg_value)

    # 막대그래프 생성
    fig = go.Figure()

    for idx, metric in enumerate(metrics_to_plot):
        fig.add_trace(
            go.Bar(
                # x=[str(cl) for cl in context_labels],  # X축: Input Context 길이를 문자열로 변환
                x=context_labels,  # X축: Input Context 길이
                y=grouped_bar_data[metric],  # Y축: 평균 Latency 값
                name=metric,  # 범례 이름
                marker=dict(
                        line=dict(
                            color="black",
                            width=1,
                        ),
                        color=color_palate["bars"][idx % len(color_palate["bars"])],
                    ),
            )
        )

    # 그래프 설정
    fig.update_layout(
        # title='<b>Latency Comparison by Input Context Length</b>',
        xaxis_title="Input Context Length (Tokens)",
        yaxis_title="Latency (ms/tokens)",
        font=dict(family="Noto Sans, sans-serif", size=18, color="black"),
        barmode="group",  # 그룹별 막대그래프
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            type="category",  # X축 눈금을 문자열로 표시
            gridwidth=0.5,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
        ),
        bargroupgap=0.1,  # 그룹 내 막대 사이 간격 (기본값 0.2~0.3)
        bargap=0.2,  # 그룹 간 막대 사이의 간격 (기본값 0.1~0.2)
        showlegend=True,  # 범례 활성화
    )

    # 그래프 출력
    py.offline.iplot(fig)
    fig.write_image(f"3.png")
    

    ################


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py [directory] [filename_regex]")
        print('예시: python script.py /path/to/output "output_\\d+_\\d+\\.txt"')
        sys.exit(1)

    directory = sys.argv[1]
    filename_pattern = sys.argv[2]

    # 1. 파일 패턴에 맞는 모든 파일에서 메트릭 추출
    all_data = process_files_in_directory(directory, filename_pattern)
    if not all_data:
        print("No data to display.")
        sys.exit(1)

    # 2. 파일 이름에서 컨텍스트 길이를 추출하여 그룹핑
    grouped_data = group_metrics_by_context_length(all_data)
    if not grouped_data:
        print("No grouped data found.")
        sys.exit(1)

    # 3. 각 그룹별 통계량 계산
    grouped_stats = calculate_statistics_grouped_by_context_length(grouped_data)
    if grouped_stats:
        print("\n컨텍스트 길이별 통계 수치:")
        for cl, stats in sorted(grouped_stats.items()):
            print(f"\n--- 컨텍스트 길이: {cl} ---")
            for metric, stat in stats.items():
                if stat is not None:
                    print(f"{metric}:")
                    print(f"  - 평균     : {stat['mean']:.2f}")
                    print(f"  - 표준편차 : {stat['std']:.2f}")
                    print(f"  - 개수     : {stat['count']}")
                else:
                    print(f"{metric}: 데이터 없음")
    else:
        print("통계 수치를 계산할 데이터가 없습니다.")

    # 4. 시각화
    visualize_metrics(grouped_data)


###############33


# def visualize_metrics_as_bars(grouped_data):
#     """
#     막대그래프(Bar Chart)로 시각화
#     - X축: 메트릭 이름 (Time To First Token Latency, Avg Decoding Latency)
#     - Y축: 값 (ms)
#     - 컨텍스트 길이별로 색상 차별화
#     - 범례 제거
#     """
#     if not grouped_data:
#         print("No grouped data to visualize.")
#         return

#     # 사용된 모든 메트릭 추출
#     metrics = None
#     for cl, metrics_list in grouped_data.items():
#         if metrics_list:
#             metrics = list(
#                 metrics_list[0].keys()
#             )  # ['Time To First Token Latency (ms)', 'Avg Decoding Latency (ms/tokens)']
#             break
#     if not metrics:
#         print("No metric keys found in grouped data.")
#         return

#     fig = go.Figure()
#     # 색상 팔레트 (Tableau 10)
#     tableau_colors = [
#         "#4E79A7",
#         "#F28E2B",
#         "#E15759",
#         "#76B7B2",
#         "#59A14F",
#         "#EDC948",
#         "#B07AA1",
#         "#FF9DA7",
#         "#9C755F",
#         "#BAB0AC",
#     ]
#     # 컨텍스트 길이별 색상 설정
#     cl_colors = tableau_colors
#     sorted_contexts = sorted(grouped_data.keys())

#     for cl_idx, cl in enumerate(sorted_contexts):
#         metric_means = []
#         for metric in metrics:
#             values = [m[metric] for m in grouped_data[cl] if m[metric] is not None]
#             metric_means.append(np.mean(values) if values else 0)

#         fig.add_trace(
#             go.Bar(
#                 x=metrics,
#                 y=metric_means,
#                 name=f"CL: {cl}",
#                 marker_color=cl_colors[cl_idx % len(cl_colors)],
#                 showlegend=False,  # 범례 제거
#             )
#         )

#     fig.update_layout(
#         title="<b>Inference Metrics (Grouped Bar Chart by Context Length)</b>",
#         xaxis_title="Metric Name",
#         yaxis_title="Metric Value (ms)",
#         barmode="group",
#         plot_bgcolor="white",
#         xaxis=dict(
#             showgrid=True,
#             gridcolor="lightgray",
#             gridwidth=0.5,
#         ),
#         yaxis=dict(
#             showgrid=True,
#             gridcolor="lightgray",
#             gridwidth=0.5,
#             dtick=10,  # Y축 눈금을 10 단위로 설정 (더 촘촘하게)
#         ),
#     )
#     py.offline.iplot(fig)


# def visualize_grouped_metrics_in_one_figure(grouped_data):
#     """
#     x축에 메트릭 이름(2개: Time To First Token Latency, Avg Decoding Latency)을 두고,
#     컨텍스트 길이에 따라 색상을 달리 표시하여 하나의 박스플롯에 모으는 함수입니다.

#     - y축과 x축에 격자선(grid) 추가
#     """
#     if not grouped_data:
#         print("No grouped data to visualize.")
#         return

#     # 사용된 모든 메트릭 추출
#     metrics = None
#     for cl, metrics_list in grouped_data.items():
#         if metrics_list:
#             metrics = list(metrics_list[0].keys())
#             break
#     if not metrics:
#         print("No metric keys found in grouped data.")
#         return

#     fig = go.Figure()
#     # 컨텍스트 길이에 따라 색상 지정
#     cl_colors = colors.qualitative.Set3

#     sorted_contexts = sorted(grouped_data.keys())
#     for cl_idx, cl in enumerate(sorted_contexts):
#         for metric in metrics:
#             values = [m[metric] for m in grouped_data[cl] if m[metric] is not None]
#             if values:
#                 fig.add_trace(
#                     go.Box(
#                         y=values,
#                         x=[metric] * len(values),  # x축에 메트릭 이름 고정
#                         name=f"CL: {cl}",
#                         marker_color=cl_colors[cl_idx % len(cl_colors)],
#                         boxmean=True,
#                         boxpoints=False,
#                         legendgroup=f"CL: {cl}",
#                         showlegend=(
#                             metric == metrics[0]
#                         ),  # 첫 메트릭에서만 범례 보이도록
#                     )
#                 )

#     fig.update_layout(
#         title="<b>All Metrics in One Figure (Grouped by Context Length)</b>",
#         yaxis_title="Metric Value (ms or tokens)",
#         xaxis_title="Metric Name",
#         boxmode="group",
#         showlegend=True,
#         plot_bgcolor="white",
#         xaxis=dict(
#             tickangle=0,
#             tickmode="array",
#             tickvals=metrics,
#             ticktext=metrics,
#             categoryorder="array",
#             showgrid=True,  # x축 눈금선 추가
#             gridcolor="lightgray",
#             gridwidth=0.5,
#         ),
#         yaxis=dict(
#             showgrid=True, gridcolor="lightgray", gridwidth=0.5  # y축 눈금선 추가
#         ),
#     )

#     py.offline.iplot(fig)
