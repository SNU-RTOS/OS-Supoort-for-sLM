import os
import re
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PhaseMetrics:
    def __init__(self):
        self.duration = 0
        self.minor_faults = 0
        self.major_faults = 0

class LogMetrics:
    def __init__(self):
        self.phases: Dict[str, PhaseMetrics] = {}
        self.decode_times: List[float] = []
        self.prefill_time = 0
        self.time_to_first_token = 0
        self.is_valid = True  # Flag to track if all required metrics were found
        self.decoding_time_breakdown: Dict[str, float] = {
            'avg_latency': 0.0,
            'user_time': 0.0,
            'system_time': 0.0
        }
        
    def calculate_total_major_faults(self) -> int:
        return sum(phase.major_faults for phase in self.phases.values())
    
    def calculate_average_decode_time(self) -> float:
        return np.mean(self.decode_times) if self.decode_times else 0

def safe_float_conversion(value: str, default: float = 0.0) -> float:
    """Safely convert string to float with error handling."""
    try:
        return float(value.replace(',', ''))
    except (ValueError, AttributeError):
        logger.warning(f"Could not convert '{value}' to float, using default: {default}")
        return default

def parse_phase_info(content: str) -> Dict[str, PhaseMetrics]:
    phases = {}
    
    # First, parse decode step-level faults
    decode_step_pattern = r"Decode step (\d+)\s*\n\s*- Major page faults: (\d+)\s*\n\s*- Minor page faults: (\d+)"
    decode_step_matches = list(re.finditer(decode_step_pattern, content))
    
    # Calculate averages from step-level data
    if decode_step_matches:
        major_faults = [int(match.group(2)) for match in decode_step_matches]
        minor_faults = [int(match.group(3)) for match in decode_step_matches]
        
        # Create Decode phase metrics from step-level data
        decode_metrics = PhaseMetrics()
        decode_metrics.major_faults = np.mean(major_faults)
        decode_metrics.minor_faults = np.mean(minor_faults)
        phases['Decode'] = decode_metrics
        
        logger.info(f"Parsed Decode step faults: Avg Major={decode_metrics.major_faults:.2f}, Avg Minor={decode_metrics.minor_faults:.2f}")
    else:
        # Fallback to previous parsing method if step-level data not found
        logger.warning("No step-level decode faults found, falling back to summary parsing")
        decode_phase_pattern = r"Phase: Decode\s*\nNumber of measurements: (\d+)\s*\nAverage duration: ([\d.]+) ms\s*\nTotal minor page faults: (\d+)\s*\nTotal major page faults: (\d+)"
        decode_match = re.search(decode_phase_pattern, content)
        
        if decode_match:
            decode_metrics = PhaseMetrics()
            decode_metrics.duration = float(decode_match.group(2))
            decode_metrics.minor_faults = int(decode_match.group(3))
            decode_metrics.major_faults = int(decode_match.group(4))
            phases['Decode'] = decode_metrics
    
    # Parse other phases with the updated format
    phase_pattern = r"Phase: ([^\n]+)\s*\nDuration: (\d+) ms\s*\nMinor page faults: (\d+)\s*\nMajor page faults: (\d+)"
    matches = re.finditer(phase_pattern, content)
    
    for match in matches:
        try:
            phase_name = match.group(1).strip()
            if phase_name != 'Decode':  # Skip Decode as it's already handled
                metrics = PhaseMetrics()
                metrics.duration = int(match.group(2))
                metrics.minor_faults = int(match.group(3))
                metrics.major_faults = int(match.group(4))
                phases[phase_name] = metrics
                logger.debug(f"Parsed phase {phase_name}: duration={metrics.duration}, minor={metrics.minor_faults}, major={metrics.major_faults}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing phase info for {phase_name}: {e}")
            continue
    
    return phases

def parse_decode_times(content: str) -> List[float]:
    """Parse decode times from the correct format, extracting CPU time"""
    decode_pattern = r"Decode \d+ took\s*\n-\s*([\d.]+)\s*\[sec\] CPU time"
    times = []
    matches = list(re.finditer(decode_pattern, content))
    
    if not matches:
        logger.warning("No decode time entries found")
        return times
    
    i = 0
    for match in matches:
        try:
            cpu_time = float(match.group(1))
            if(i == 0):
                i += 1
                continue
            times.append(cpu_time)
            logger.debug(f"Parsed decode time: {cpu_time} seconds")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing decode time: {e}")
            continue
    
    if times:
        logger.info(f"Successfully parsed {len(times)} decode times")
    
    return times

def parse_decoding_time_breakdown(content: str) -> Dict[str, float]:
    """
    Parse decoding time breakdown from log content.
    
    Returns a dictionary with the following keys:
    - 'avg_latency': Average Decoding Latency
    - 'user_time': User time
    - 'system_time': System time
    """
    breakdown = {
        'avg_latency': 0.0,
        'user_time': 0.0,
        'system_time': 0.0
    }
    
    # Parse average decoding latency
    avg_latency_pattern = r"\[METRICS\]\s*Average Decoding Latency\s*:\s*([\d.]+)\s*\[sec\]"
    avg_latency_match = re.search(avg_latency_pattern, content)
    if avg_latency_match:
        breakdown['avg_latency'] = safe_float_conversion(avg_latency_match.group(1))
    
    # Parse time breakdown
    time_pattern = r"- ([\d.]+)\s*\[sec\] (User|System) time"
    time_matches = list(re.finditer(time_pattern, content))
    
    for match in time_matches:
        time_value = safe_float_conversion(match.group(1))
        time_type = match.group(2).lower()
        
        if time_type == 'user':
            breakdown['user_time'] = time_value
        elif time_type == 'system':
            breakdown['system_time'] = time_value
    
    return breakdown

def parse_prefill_and_ttft(content: str) -> Tuple[float, float]:
    """Parse prefill time and time to first token with more flexible patterns"""
    prefill_pattern = r"\[INFO\]\s*Prefill Stage took\s*([\d,]+(?:\.\d+)?)\s*ms"
    ttft_pattern = r"\[METRICS\]\s*Time To First Token\s*:\s*([\d,]+(?:\.\d+)?)\s*ms"
    
    prefill_match = re.search(prefill_pattern, content)
    ttft_match = re.search(ttft_pattern, content)
    
    prefill_time = safe_float_conversion(prefill_match.group(1)) if prefill_match else 0
    ttft = safe_float_conversion(ttft_match.group(1)) if ttft_match else 0
    
    if prefill_match and ttft_match:
        logger.info(f"Parsed prefill={prefill_time}ms, ttft={ttft}ms")
    else:
        logger.warning("Failed to parse prefill or TTFT values")
    
    return prefill_time, ttft

def parse_log_file(file_path: str) -> Optional[LogMetrics]:
    metrics = LogMetrics()
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Parse all components
        metrics.phases = parse_phase_info(content)
        metrics.decode_times = parse_decode_times(content)
        metrics.prefill_time, metrics.time_to_first_token = parse_prefill_and_ttft(content)
        
        # Parse decoding time breakdown
        metrics.decoding_time_breakdown = parse_decoding_time_breakdown(content)
        
        # Validation
        if 'Decode' not in metrics.phases:
            logger.warning(f"No Decode phase information found in {file_path}")
            metrics.is_valid = False
        
        return metrics
    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {e}")
        return None

def process_directory(base_dir: str, ram_dirs: Dict[str, str], input_tokens: List[int]) -> Dict:
    results = defaultdict(lambda: defaultdict(list))
    valid_data_found = False
    
    if not os.path.exists(base_dir):
        logger.error(f"Base directory {base_dir} does not exist")
        return results
    
    for ram_size, dir_name in ram_dirs.items():
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            logger.warning(f"Directory not found for RAM size {ram_size}: {dir_path}")
            continue
            
        for token in input_tokens:
            pattern = f"output_{token}_\\d+\\.txt"
            matching_files = [f for f in os.listdir(dir_path) if re.match(pattern, f)]
            
            if not matching_files:
                logger.warning(f"No matching files found for token {token} in {ram_size}")
                continue
                
            valid_files = 0
            for file_name in matching_files:
                metrics = parse_log_file(os.path.join(dir_path, file_name))
                if metrics and metrics.is_valid:
                    results[ram_size][token].append(metrics)
                    valid_files += 1
                    valid_data_found = True
            
            if valid_files > 0:
                logger.info(f"Processed {valid_files} valid files for {ram_size}, token {token}")
    
    if not valid_data_found:
        logger.error("No valid data found in any of the directories")
    
    return results

def create_decode_major_faults_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    # Create a separate plot for each input token length
    figs = []
    has_data = False

    # For each input token length
    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False
        
        # Collect data for this token length
        y_values = []
        for ram_size in ram_sizes:
            if token in results[ram_size] and results[ram_size][token]:
                metrics_list = [m for m in results[ram_size][token] if m.is_valid and 'Decode' in m.phases]
                if metrics_list:
                    # Use the average of the Decode phase major faults across all runs
                    avg_faults = np.mean([m.phases['Decode'].major_faults for m in metrics_list[1:]])
                    y_values.append(avg_faults)
                    token_has_data = True
                    has_data = True
                else:
                    y_values.append(None)
            else:
                y_values.append(None)
        
        if token_has_data:
            fig.add_trace(go.Bar(
                x=ram_sizes,
                y=y_values,
                text=[f"{v:.1f}" if v is not None else "N/A" for v in y_values],
                textposition='outside',
                marker_color='#E15759'
            ))

            # Update layout for this specific plot
            fig.update_layout(
                title=f"Major Page Faults During Decode Phase (Input Tokens: {token})",
                xaxis_title="RAM Size",
                yaxis_title="Average Major Page Faults",
                template='plotly_white',
                showlegend=False,
                height=600,
                width=1000,
                # Add grid lines for better readability
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinecolor='LightGray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='Gray'
                )
            )
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for decode major faults plot")
        return None

    return figs


def create_total_major_faults_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    # Collect all unique phases
    all_phases = set()
    for ram_size in ram_sizes:
        for token in input_tokens:
            if token in results[ram_size]:
                for metrics in results[ram_size][token]:
                    if metrics.is_valid:
                        all_phases.update(metrics.phases.keys())
    
    if not all_phases:
        logger.warning("No valid phases found for total major faults plot")
        return None

    # Color scheme for phases
    phase_colors = {
        'Decode': '#1f77b4',
        'Prefill': '#ff7f0e',
        'Prepare_Runners': '#2ca02c',
        'Build_KVCache': '#d62728',
        'Load_SentencePiece': '#9467bd',
        'Prepare_Prompt': '#8c564b',
        'Build_Interpreter': '#e377c2',
        'Model_Loading': '#7f7f7f'
    }

    # Create a separate plot for each input token length
    figs = []
    has_data = False

    # For each input token length
    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False

        # For each phase
        for phase in sorted(all_phases):
            y_values = []
            # For each RAM size
            for ram_size in ram_sizes:
                if token in results[ram_size] and results[ram_size][token]:
                    metrics_list = [m for m in results[ram_size][token] if m.is_valid and phase in m.phases]
                    if metrics_list:
                        avg_faults = np.mean([m.phases[phase].major_faults for m in metrics_list])
                        y_values.append(avg_faults)
                        token_has_data = True
                        has_data = True
                    else:
                        y_values.append(0)
                else:
                    y_values.append(0)
            
            if any(v != 0 for v in y_values):
                fig.add_trace(
                    go.Bar(
                        name=phase,
                        x=ram_sizes,
                        y=y_values,
                        marker_color=phase_colors.get(phase, '#000000')
                    )
                )

        if token_has_data:
            # Update layout for this specific plot
            fig.update_layout(
                title=f"Total Major Page Faults by Phase (Input Tokens: {token})",
                xaxis_title="RAM Size",
                yaxis_title="Average Major Page Faults",
                barmode='stack',
                template='plotly_white',
                showlegend=True,
                legend_title="Phases",
                height=600,
                width=1000
            )
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for total major faults plot")
        return None

    return figs


def create_ttft_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    # Create a separate plot for each input token length
    figs = []
    has_data = False

    # For each input token length
    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False
        
        # Collect prefill and first decode data for this token length
        prefill_values = []
        first_decode_values = []
        
        for ram_size in ram_sizes:
            if token in results[ram_size] and results[ram_size][token]:
                metrics_list = [m for m in results[ram_size][token] if m.is_valid]
                if metrics_list:
                    # Calculate average prefill time (convert from ms to sec)
                    avg_prefill = np.mean([m.prefill_time for m in metrics_list]) / 1000
                    
                    # Get first decode time from decode_times list
                    avg_first_decode = np.mean([m.time_to_first_token if m.time_to_first_token else 0 for m in metrics_list]) / 1000
                    
                    prefill_values.append(avg_prefill)
                    first_decode_values.append(avg_first_decode)
                    token_has_data = True
                    has_data = True
                else:
                    prefill_values.append(None)
                    first_decode_values.append(None)
            else:
                prefill_values.append(None)
                first_decode_values.append(None)
        
        if token_has_data:
            # Add prefill bars
            fig.add_trace(go.Bar(
                name='Prefill',
                x=ram_sizes,
                y=prefill_values,
                marker_color='#ff7f0e',
                text=[f"{v:.2f}" if v is not None else "N/A" for v in prefill_values],
                textposition='auto'
            ))
            
            # Add first decode bars
            fig.add_trace(go.Bar(
                name='First Decode',
                x=ram_sizes,
                y=first_decode_values,
                marker_color='#1f77b4',
                text=[f"{v:.2f}" if v is not None else "N/A" for v in first_decode_values],
                textposition='auto'
            ))

            # Update layout for this specific plot
            fig.update_layout(
                title=f"Time to First Token Breakdown (Input Tokens: {token})",
                xaxis_title="RAM Size",
                yaxis_title="Time (seconds)",
                barmode='stack',
                template='plotly_white',
                showlegend=True,
                # legend_title="Components",
                height=600,
                width=1000,
                # Add grid lines for better readability
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinecolor='lightgray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinewidth=0.1,
                    zerolinecolor='LightGray'
                )
            )
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for TTFT plot")
        return None

    return figs

def create_decode_latency_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    # Create a separate plot for each input token length
    figs = []
    has_data = False

    # For each input token length
    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False
        
        # Collect data for this token length
        y_values = []
        for ram_size in ram_sizes:
            if token in results[ram_size] and results[ram_size][token]:
                metrics_list = [m for m in results[ram_size][token] if m.is_valid and m.decode_times]
                if metrics_list:
                    # Calculate average decode time
                    avg_decode = np.mean([m.calculate_average_decode_time() for m in metrics_list]) / 4
                    # avg_decode = np.mean([m.decoding_time_breakdown['avg_latency'] for m in metrics_list[1:]])
                    y_values.append(avg_decode)
                    token_has_data = True
                    has_data = True
                else:
                    y_values.append(None)
            else:
                y_values.append(None)
        
        if token_has_data:
            fig.add_trace(go.Bar(
                x=ram_sizes,
                y=y_values,
                text=[f"{v:.3f}" if v is not None else "N/A" for v in y_values],
                textposition='outside',
                marker_color='#1f77b4'
            ))

            # Update layout for this specific plot
            fig.update_layout(
                title=f"Average Decoding Latency (Input Tokens: {token})",
                xaxis_title="RAM Size",
                yaxis_title="Decode Time (seconds)",
                template='plotly_white',
                showlegend=False,
                height=600,
                width=1000,
                # Add grid lines for better readability
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    # zeroline=True,
                    # zerolinecolor='lightgray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinewidth=0.1,
                    zerolinecolor='Gray'
                )
            )
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for decode latency plot")
        return None

    return figs

def create_combined_decode_metrics_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    # Create a separate plot for each input token length
    figs = []
    has_data = False

    # For each input token length
    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False
        
        # Collect latency data for this token length
        latency_values = []
        fault_values = []
        
        for ram_size in ram_sizes:
            if token in results[ram_size] and results[ram_size][token]:
                metrics_list = [m for m in results[ram_size][token] if m.is_valid and m.decode_times]
                if metrics_list:
                    # Calculate average decode time
                    avg_decode = np.mean([m.calculate_average_decode_time() for m in metrics_list]) / 4
                    # avg_decode = np.mean([m.decoding_time_breakdown['avg_latency'] for m in metrics_list[1:]])
                    latency_values.append(avg_decode)
                    
                    # Calculate average major faults during decode
                    avg_faults = np.mean([m.phases['Decode'].major_faults for m in metrics_list[1:] if 'Decode' in m.phases])
                    fault_values.append(avg_faults)
                    
                    token_has_data = True
                    has_data = True
                else:
                    latency_values.append(None)
                    fault_values.append(None)
            else:
                latency_values.append(None)
                fault_values.append(None)
        
        if token_has_data:
            # Calculate y-axis ranges
            latency_max = max(v for v in latency_values if v is not None)
            faults_max = max(v for v in fault_values if v is not None)
            
            # Add decode latency bars - position slightly to the left
            x_positions = list(range(len(ram_sizes)))
            fig.add_trace(go.Bar(
                name='Decode Latency',
                x=[x - 0.2 for x in x_positions],  # Shift left
                y=latency_values,
                text=[f"{v:.3f}" if v is not None else "N/A" for v in latency_values],
                textposition='outside',
                marker_color='#1f77b4',
                width=0.35,  # Make bars thinner
                yaxis='y'
            ))
            
            # Add major faults bars - position slightly to the right
            fig.add_trace(go.Bar(
                name='Major Page Faults',
                x=[x + 0.2 for x in x_positions],  # Shift right
                y=fault_values,
                text=[f"{v:.1f}" if v is not None else "N/A" for v in fault_values],
                textposition='outside',
                marker_color='#E15759',
                width=0.35,  # Make bars thinner
                yaxis='y2'
            ))

            # Update layout for this specific plot
            fig.update_layout(
                title=f"Decode Latency and Major Page Faults (Input Tokens: {token})",
                plot_bgcolor="white",
                xaxis_title="RAM Size",
                yaxis=dict(
                    title=dict(
                        text="Decode Time (seconds)",
                        font=dict(color='#1f77b4')
                    ),
                    tickfont=dict(color='#1f77b4'),
                    gridcolor='LightGray',
                    # zeroline=True,
                    # zerolinecolor="lightgray",
                    range=[0, latency_max * 1.2]  # Add 20% padding
                ),
                yaxis2=dict(
                    title=dict(
                        text="Average Major Page Faults",
                        font=dict(color='#E15759')
                    ),
                    tickfont=dict(color='#E15759'),
                    anchor="x",
                    overlaying="y",
                    side="right",
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinecolor="lightgray",
                    zerolinewidth=0.1,
                    range=[0, faults_max * 1.2]  # Add 20% padding
                ),
                showlegend=True,
                # legend=dict(
                #     x=-0.15,  # Position legend to the left of the plot
                #     y=1.0,
                #     xanchor='right',
                #     yanchor='bottom',
                #     bgcolor='rgba(255, 255, 255, 0.8)'  # Semi-transparent background
                # ),
                height=600,
                width=1200,  # Increased width to accommodate legend
                margin=dict(l=150),  # Added left margin for legend
                # Add grid lines for better readability
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    ticktext=ram_sizes,
                    tickvals=x_positions,
                    # zeroline=True,
                    # zerolinecolor="lightgray",
                )
            )
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for combined decode metrics plot")
        return None

    return figs

def create_decode_minor_faults_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    figs = []
    has_data = False

    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False
        
        y_values = []
        for ram_size in ram_sizes:
            if token in results[ram_size] and results[ram_size][token]:
                metrics_list = [m for m in results[ram_size][token] if m.is_valid and 'Decode' in m.phases]
                if metrics_list:
                    # Use the total minor faults instead of average
                    avg_faults = np.mean([m.phases['Decode'].minor_faults for m in metrics_list])
                    y_values.append(avg_faults)
                    token_has_data = True
                    has_data = True
                else:
                    y_values.append(None)
            else:
                y_values.append(None)
        
        if token_has_data:
            fig.add_trace(go.Bar(
                x=ram_sizes,
                y=y_values,
                text=[f"{v:,.0f}" if v is not None else "N/A" for v in y_values],  # Format with commas
                textposition='outside',
                marker_color='#2ca02c'
            ))

            fig.update_layout(
                title=f"Minor Page Faults During Decode Phase (Input Tokens: {token})",
                xaxis_title="RAM Size",
                yaxis_title="Number of Minor Page Faults",
                template='plotly_white',
                showlegend=False,
                height=600,
                width=1000,
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='Gray',
                    # Format y-axis labels with commas
                    tickformat=",d"
                )
            )
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for decode minor faults plot")
        return None

    return figs

def create_combined_decode_latency_minor_faults_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    figs = []
    has_data = False

    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False
        
        latency_values = []
        fault_values = []
        
        for ram_size in ram_sizes:
            if token in results[ram_size] and results[ram_size][token]:
                metrics_list = [m for m in results[ram_size][token] if m.is_valid and m.decode_times]
                if metrics_list:
                    # Use CPU time for latency
                    # 나중에 rusage 사용이 명확해지면 사용, thread 여러 개일 때 사용 불가가
                    avg_decode = np.mean([m.calculate_average_decode_time() for m in metrics_list]) / 4
                    # avg_decode = np.mean([m.decoding_time_breakdown['avg_latency'] for m in metrics_list[1:]])
                    latency_values.append(avg_decode)
                    
                    # Use total minor faults
                    avg_faults = np.mean([m.phases['Decode'].minor_faults for m in metrics_list if 'Decode' in m.phases])
                    fault_values.append(avg_faults)
                    
                    token_has_data = True
                    has_data = True
                else:
                    latency_values.append(None)
                    fault_values.append(None)
            else:
                latency_values.append(None)
                fault_values.append(None)
        
        if token_has_data:
            x_positions = list(range(len(ram_sizes)))
            
            # Add decode latency bars
            fig.add_trace(go.Bar(
                name='Decode Latency',
                x=[x - 0.2 for x in x_positions],
                y=latency_values,
                text=[f"{v:.3f}" if v is not None else "N/A" for v in latency_values],
                textposition='outside',
                marker_color='#1f77b4',
                width=0.35,
                yaxis='y'
            ))
            
            # Add minor faults bars
            fig.add_trace(go.Bar(
                name='Minor Page Faults',
                x=[x + 0.2 for x in x_positions],
                y=fault_values,
                text=[f"{v:,.0f}" if v is not None else "N/A" for v in fault_values],  # Format with commas
                textposition='outside',
                marker_color='#2ca02c',
                width=0.35,
                yaxis='y2'
            ))

            fig.update_layout(
                title=f"Decode Latency and Minor Page Faults (Input Tokens: {token})",
                plot_bgcolor="white",
                xaxis_title="RAM Size",
                yaxis=dict(
                    title=dict(
                        text="Decode Latency (seconds)",
                        font=dict(color='#1f77b4')
                    ),
                    tickfont=dict(color='#1f77b4'),
                    gridcolor='LightGray'
                ),
                yaxis2=dict(
                    title=dict(
                        text="Number of Minor Page Faults",
                        font=dict(color='#2ca02c')
                    ),
                    tickfont=dict(color='#2ca02c'),
                    anchor="x",
                    overlaying="y",
                    side="right",
                    gridcolor='LightGray',
                    tickformat=",d"  # Format y-axis labels with commas
                ),
                showlegend=True,
                height=600,
                width=1200,
                margin=dict(l=150),
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    ticktext=ram_sizes,
                    tickvals=x_positions
                )
            )
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for combined decode metrics plot")
        return None

    return figs

def create_decoding_time_breakdown_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    """
    Create a stacked bar plot showing breakdown of decoding time 
    across different components for different RAM sizes.
    """
    figs = []
    has_data = False

    # Color scheme for time components
    time_colors = {
        'user_time': '#EDC948',      # Orange
        'system_time': '#BAB0AC',    # Green
        'other_time': '#d62728'      # Red
    }

    # For each input token length
    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False
        
        # Prepare data for time components
        time_values = {
            'avg_latency': [],
            'user_time': [],
            'system_time': [],
            'other_time': []
        }
        
        for ram_size in ram_sizes:
            if token in results[ram_size] and results[ram_size][token]:
                metrics_list = [m for m in results[ram_size][token] if m.is_valid]
                if metrics_list:
                    # Calculate averages for latency, user time, and system time
                    avg_latency = np.mean([m.decoding_time_breakdown['avg_latency'] for m in metrics_list])
                    logger.info(f"AVG {avg_latency}")
                    avg_user_time = np.mean([m.decoding_time_breakdown['user_time'] for m in metrics_list]) / 4
                    avg_system_time = np.mean([m.decoding_time_breakdown['system_time'] for m in metrics_list]) / 4
                    
                    # Calculate other time
                    avg_cpu_time = avg_user_time + avg_system_time
                    avg_other_time = max(0, avg_latency - avg_cpu_time)
                    # avg_latency = 
                    
                    # Store values
                    time_values['avg_latency'].append(avg_latency)
                    time_values['user_time'].append(avg_user_time)
                    time_values['system_time'].append(avg_system_time)
                    time_values['other_time'].append(avg_other_time)
                    
                    token_has_data = True
                    has_data = True
                else:
                    for key in time_values:
                        time_values[key].append(None)
            else:
                for key in time_values:
                    time_values[key].append(None)
        
        if token_has_data:
            # Add traces for user time and system time
            fig.add_trace(go.Bar(
                name='User Time',
                x=ram_sizes,
                y=time_values['user_time'],
                marker_color=time_colors['user_time'],
                text=[f"{v:.3f}" if v is not None else "N/A" for v in time_values['user_time']],
                textposition='inside'
            ))
            
            fig.add_trace(go.Bar(
                name='System Time',
                x=ram_sizes,
                y=time_values['system_time'],
                marker_color=time_colors['system_time'],
                text=[f"{v:.3f}" if v is not None else "N/A" for v in time_values['system_time']],
                textposition='inside'
            ))
            
            # Add other time trace
            # fig.add_trace(go.Bar(
            #     name='Other Time',
            #     x=ram_sizes,
            #     y=time_values['other_time'],
            #     marker_color=time_colors['other_time'],
            #     text=[f"{v:.3f}" if v is not None else "N/A" for v in time_values['other_time']],
            #     textposition='inside'
            # ))

            # Update layout
            fig.update_layout(
                title=f"Decoding Time Breakdown (Input Tokens: {token})",
                xaxis_title="RAM Size",
                yaxis_title="Time (seconds)",
                barmode='stack',
                template='plotly_white',
                showlegend=True,
                height=600,
                width=1000,
                # Add grid lines for better readability
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinecolor='lightgray'
                )
            )
            
            # Add total time annotation
            fig.add_trace(go.Scatter(
                x=ram_sizes,
                y=time_values['avg_latency'],
                mode='text',
                # text=[f"Total: {v:.3f}" for v in time_values['avg_latency']],
                textposition='top center',
                showlegend=False
            ))
            
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for decoding time breakdown plot")
        return None

    return figs

def create_prefill_major_faults_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    # Create a separate plot for each input token length
    figs = []
    has_data = False

    # For each input token length
    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False
        
        # Collect data for this token length
        y_values = []
        for ram_size in ram_sizes:
            if token in results[ram_size] and results[ram_size][token]:
                metrics_list = [m for m in results[ram_size][token] if m.is_valid and 'Prefill' in m.phases]
                if metrics_list:
                    # Calculate average major faults for prefill phase
                    avg_faults = np.mean([m.phases['Prefill'].major_faults for m in metrics_list])
                    y_values.append(avg_faults)
                    token_has_data = True
                    has_data = True
                else:
                    y_values.append(None)
            else:
                y_values.append(None)
        
        if token_has_data:
            fig.add_trace(go.Bar(
                x=ram_sizes,
                y=y_values,
                text=[f"{v:.1f}" if v is not None else "N/A" for v in y_values],
                textposition='outside',
                marker_color='#E15759'  # Orange color to match prefill phase
            ))

            # Update layout for this specific plot
            fig.update_layout(
                title=f"Major Page Faults During Prefill Phase (Input Tokens: {token})",
                xaxis_title="RAM Size",
                yaxis_title="Average Major Page Faults",
                template='plotly_white',
                showlegend=False,
                height=600,
                width=1000,
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinecolor='LightGray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='Gray'
                )
            )
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for prefill major faults plot")
        return None

    return figs

def create_prefill_minor_faults_plot(results: Dict, ram_sizes: List[str], input_tokens: List[int]):
    # Create a separate plot for each input token length
    figs = []
    has_data = False

    # For each input token length
    for token in input_tokens:
        fig = go.Figure()
        token_has_data = False
        
        # Collect data for this token length
        y_values = []
        for ram_size in ram_sizes:
            if token in results[ram_size] and results[ram_size][token]:
                metrics_list = [m for m in results[ram_size][token] if m.is_valid and 'Prefill' in m.phases]
                if metrics_list:
                    # Calculate average minor faults for prefill phase
                    avg_faults = np.mean([m.phases['Prefill'].minor_faults for m in metrics_list])
                    y_values.append(avg_faults)
                    token_has_data = True
                    has_data = True
                else:
                    y_values.append(None)
            else:
                y_values.append(None)
        
        if token_has_data:
            fig.add_trace(go.Bar(
                x=ram_sizes,
                y=y_values,
                text=[f"{v:,.0f}" if v is not None else "N/A" for v in y_values],  # Format with commas
                textposition='outside',
                marker_color='#2ca02c'  # Orange color to match prefill phase
            ))

            # Update layout for this specific plot
            fig.update_layout(
                title=f"Minor Page Faults During Prefill Phase (Input Tokens: {token})",
                xaxis_title="RAM Size",
                yaxis_title="Number of Minor Page Faults",
                template='plotly_white',
                showlegend=False,
                height=600,
                width=1000,
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='Gray',
                    tickformat=",d"  # Format y-axis labels with commas
                )
            )
            figs.append((token, fig))

    if not has_data:
        logger.warning("No valid data for prefill minor faults plot")
        return None

    return figs

def parse_ram_dir(dir_name: str) -> Optional[str]:
    """Parse RAM size from directory name in the format result_dp_[]G or result_dp_[]M"""
    match = re.match(r'result_dp_(\d+)(G|M)', dir_name)
    if match:
        size = int(match.group(1))
        unit = match.group(2)
        if unit == 'M':
            return f"{size}M"
        else:  # unit == 'G'
            return f"{size}G"
    return None

def get_available_ram_sizes(base_dir: str) -> List[str]:
    """Get list of available RAM sizes from directory names"""
    try:
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        ram_sizes = []
        for dir_name in dirs:
            ram_size = parse_ram_dir(dir_name)
            if ram_size:
                ram_sizes.append(ram_size)
        return sorted(ram_sizes, key=lambda x: (x[-1], int(x[:-1])))  # Sort by unit (M then G) and then by size
    except Exception as e:
        logger.error(f"Error getting RAM sizes: {e}")
        return []

def get_ram_dir_name(ram_size: str) -> str:
    """Convert RAM size (e.g., '4G' or '512M') to directory name format"""
    return f"result_dp_{ram_size}"

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze inference logs for different RAM sizes.')
    parser.add_argument('--base-dir', type=str, required=True,
                      help='Base directory containing RAM size subdirectories')
    parser.add_argument('--ram-sizes', type=str, nargs='+',
                      help='List of RAM sizes to analyze (e.g., 4G 8G 16G 512M)')
    parser.add_argument('--input-tokens', type=int, nargs='+', default=[8, 128, 512],
                      help='List of input token lengths to analyze')
    
    args = parser.parse_args()
    
    # Validate base directory
    if not os.path.isdir(args.base_dir):
        logger.error(f"Base directory does not exist: {args.base_dir}")
        return

    # Get available RAM sizes if not specified
    if args.ram_sizes is None:
        available_ram_sizes = get_available_ram_sizes(args.base_dir)
        if not available_ram_sizes:
            logger.error("No valid RAM size directories found")
            return
        logger.info(f"Found RAM sizes: {', '.join(available_ram_sizes)}")
        ram_sizes = available_ram_sizes
    else:
        ram_sizes = args.ram_sizes
        
    # Convert RAM sizes to directory names
    ram_dirs = {size: get_ram_dir_name(size) for size in ram_sizes}
    
    logger.info("Starting log analysis...")
    logger.info(f"Analyzing RAM sizes: {', '.join(ram_sizes)}")
    logger.info(f"Input tokens: {args.input_tokens}")
    
    # Process all log files
    results = process_directory(args.base_dir, ram_dirs, args.input_tokens)
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Create and display plots
    plot_functions = {
        'decode_major_faults': create_decode_major_faults_plot,
        'decode_minor_faults': create_decode_minor_faults_plot, 
        'ttft_breakdown': create_ttft_plot,
        'decode_latency': create_decode_latency_plot,
        'combined_decode_metrics': create_combined_decode_metrics_plot,
        'combined_decode_latency_minor_faults': create_combined_decode_latency_minor_faults_plot,
        'decoding_time_breakdown': create_decoding_time_breakdown_plot,
        'create_prefill_major_faults_plot': create_prefill_major_faults_plot,
        'create_prefill_minor_faults_plot': create_prefill_minor_faults_plot
    }
    
    plots_created = False
    for name, func in plot_functions.items():
        figs = func(results, ram_sizes, args.input_tokens)
        if figs is not None:
            for token, fig in figs:
                # Save as PNG
                png_path = os.path.join('plots', f'{name}_{token}_tokens.png')
                fig.write_image(png_path)
                logger.info(f"Created plot: {png_path}")
                
                # Show the plot in browser
                fig.show()
            plots_created = True
    
    if plots_created:
        logger.info("Analysis complete. Plots have been saved to the 'plots' directory.")
    else:
        logger.error("No plots were created due to lack of valid data.")

if __name__ == "__main__":
    main()