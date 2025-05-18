# RDS Anomaly Detection Model Documentation

## Overview

This document describes the RDS monitoring and anomaly detection system designed to identify significant performance issues in AWS RDS database instances. The model uses statistical approaches to detect meaningful deviations from baseline performance across multiple database metrics.

## Detection Methodology

The system employs a multi-layered approach to anomaly detection, combining several techniques to accurately identify the onset, duration, and severity of anomalies.

### Core Principles

1. **Baseline Establishment**: Uses first 60% of historical data by default, automatically adjusting if major shifts are detected
2. **Threshold Determination**: Combines percentile analysis with standard deviation multipliers
3. **Anomaly Qualification**: Requires both absolute threshold violation and relative percentage change
4. **Persistence Verification**: Confirms anomalies persist for minimum duration to filter out transient spikes
5. **Onset Identification**: Precisely identifies when anomalies began, not just when they were detected

## Key Detection Features

### 1. Smart Baseline Calculation

The system dynamically determines the appropriate baseline period to exclude anomalous data:

```python
# Look for large shifts in the data (significant jumps in rolling mean)
rolling_mean = df['value'].rolling(window=6).mean()  # 30-min rolling average
pct_changes = rolling_mean.pct_change().abs() * 100
significant_jumps = pct_changes[pct_changes > 200].index

# Adjust baseline calculation point if significant jumps found
if len(significant_jumps) > 0:
    first_jump = significant_jumps[0]
    jump_idx = df.index.get_loc(first_jump)
    baseline_pct = max(0.4, min(0.9, jump_idx / len(df)))
```

This allows the model to:
- Detect when data patterns fundamentally change
- Exclude anomalies from the baseline calculation
- Adapt to each metric's specific characteristics

The baseline is calculated as the **median** of the baseline period, making it robust against outliers while still representing typical performance.

### 2. Anomaly Onset Detection

When an anomaly is detected, the system looks backward to identify exactly when it started:

```python
# Try to find the exact moment the metric started to rise
lookback = min(10, i)  # Look back up to 10 points
start_idx = i
                    
for j in range(i-1, i-lookback-1, -1):
    if j >= 0:
        prev_value = values[j]
        prev_pct_change = ((prev_value - baseline) / baseline) * 100
        
        # Find inflection point
        if prev_value < threshold * 0.8 or prev_pct_change < min_percentage_change * 0.5:
            start_idx = j + 1  # Anomaly starts right after this point
            break
```

This technique:
- Identifies the precise moment metrics began to deviate
- Captures the full duration of the anomaly
- Reports accurate start times, not just detection times

### 3. Ongoing Anomaly Detection

The system specifically checks for anomalies still in progress at execution time:

```python
# Check if there's an ongoing anomaly at the end of the data
if current_anomaly is not None:
    duration = current_anomaly['latest_timestamp'] - current_anomaly['start_timestamp']
    duration_minutes = duration.total_seconds() / 60
    
    # Process the anomaly
    if duration_minutes >= min_duration_minutes:
        # Calculate metrics and report the anomaly
        current_anomaly['duration_minutes'] = duration_minutes
        current_anomaly['average_value'] = avg_value
        current_anomaly['peak_ratio'] = peak_ratio
        anomalies.append(current_anomaly)
```

This ensures that:
- Anomalies in progress are properly reported
- Duration is accurately calculated
- Critical issues are flagged even if they haven't resolved

### 4. Major Shift Detection

A specialized algorithm detects sustained, dramatic shifts in metric patterns:

```python
# Special handling for ongoing major shifts
if len(df) > 10:
    # Check if there's a major shift near the end of the data
    recent_data = df.iloc[-int(len(df)*0.2):]  # Last 20% of data
    recent_values = recent_data['value'].values
    
    if len(recent_values) > 5:
        recent_mean = np.mean(recent_values)
        recent_vs_baseline = (recent_mean - baseline) / baseline * 100
        
        # If recent data shows sustained, dramatic shift from baseline
        if recent_vs_baseline > 500:
            # Look backwards to find when this shift started
            threshold_for_shift = baseline * 3  # 3x baseline
            
            # Find first point where values consistently exceed threshold
            shift_start_idx = None
            consecutive_high = 0
            min_consecutive_for_shift = 3
            
            # Backward search algorithm implementation
            # ...
```

This specialized detection:
- Specifically targets sustained performance regime changes
- Works backward to find the exact starting point
- Is particularly effective for metrics that suddenly change to a new "normal"

### 5. Anomaly Merging

The system intelligently merges related anomalies separated by brief gaps:

```python
# Try to merge anomalies that are close in time
merged_anomalies = []
current_merged = all_potential_anomalies[0]

for i in range(1, len(all_potential_anomalies)):
    next_anomaly = all_potential_anomalies[i]
    time_gap = (next_anomaly['start_timestamp'] - current_merged['latest_timestamp']).total_seconds() / 60
    
    # If the gap is less than 30 minutes, merge them
    if time_gap < 30:
        # Extend the current merged anomaly
        # ...
```

This prevents multiple alerts for what is essentially the same issue by:
- Combining anomalies separated by less than 30 minutes
- Treating them as a single longer incident
- Providing a more accurate view of the anomaly's true duration

## Parameters and Thresholds

The model uses several configurable parameters with sensible defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_percentage_change` | 100% | Minimum percentage deviation from baseline to qualify as an anomaly |
| `min_consecutive` | 3 | Minimum consecutive data points required to confirm an anomaly |
| `min_duration_minutes` | 15 | Minimum duration required for an anomaly to be reported |
| Baseline period | 60% | Percentage of data used for baseline calculation |
| Threshold multiplier | 4 | Number of standard deviations above baseline for threshold |
| Percentile threshold | 95th | Percentile of baseline data used for threshold calculation |
| Major shift threshold | 500% | Percentage change required to trigger major shift detection |
| Merge window | 30 min | Maximum gap between anomalies for merging |

## Alert Qualification Process

For an anomaly to generate an alert, it must pass through several qualification stages:

1. **Initial Detection**:
   - Value exceeds the statistical threshold (min of 95th percentile or baseline + 4 std)
   - Percentage change exceeds minimum (default 100% above baseline)

2. **Persistence Verification**:
   - Anomaly persists for at least N consecutive data points (default 3)
   - Continues for minimum duration (default 15 minutes)

3. **Significance Assessment**:
   - Average deviation during anomaly period is calculated
   - Peak-to-baseline ratio is determined
   - Full anomaly duration is measured

This multi-stage qualification process ensures that alerts represent significant operational issues while filtering out normal fluctuations and brief spikes.

## Monitored Metrics

The system monitors the following RDS performance metrics by default:

- `ReadLatency`
- `WriteLatency`
- `UpdateLatency`
- `DeleteLatency`
- `DMLLatency`
- `SelectLatency`

Additional metrics can be added to the `self.metrics` list in the `RDSMonitor` class initialization.

## Running the Monitoring System

The monitoring system is designed to be run periodically (e.g., every 5 minutes via cron) with the following command:

```bash
python rds_monitor.py --instance your-rds-instance-name
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--instance` | (Required) | RDS instance identifier |
| `--region` | us-east-1 | AWS region |
| `--days` | 14 | Number of days of historical data to retrieve |
| `--min-change` | 100 | Minimum percentage change threshold |
| `--min-consecutive` | 3 | Minimum consecutive data points |
| `--min-duration` | 15 | Minimum duration in minutes |
| `--debug` | False | Enable debug logging |

## Example Alert Output

When anomalies are detected, they are logged and output with detailed information:

```
=== SIGNIFICANT ANOMALY SUMMARY ===
Metric: DeleteLatency
Detection Method: major_shift
Anomaly started at: 2025-05-16 19:05:00+00:00
Latest data point: 2025-05-18 03:45:00+00:00
Duration: 1962.0 minutes
Peak value: 7.299533
Baseline value: 0.168966
Average during anomaly: 6.854271
Percentage change: 4220.11%
Peak-to-baseline ratio: 43.20x
---
```

This output provides:
- The affected metric
- Detection method used
- Precise start time
- Current status
- Duration of the issue
- Peak and average values
- Percentage change from baseline
- Severity ratio

## Real-time Detection Capabilities

When run every 5 minutes, the system can detect anomalies as they emerge:

1. **Immediate flagging** of potential anomalies as soon as threshold is crossed
2. **Confirmation** after minimum consecutive points (default: 3 points = 15 minutes at 5-minute intervals)
3. **Precise identification** of when the anomaly actually began
4. **Ongoing tracking** of evolving anomalies

The typical detection delay for significant anomalies is 15-20 minutes from onset (to confirm persistence), with the actual start time correctly identified in the report.

## Operational Considerations

For critical production environments, consider these operational adjustments:

1. **Adjust thresholds** based on metric type and importance
2. **Increase frequency** for critical systems (e.g., run every 3 minutes)
3. **Lower duration threshold** for faster alerting (e.g., 10 minutes instead of 15)
4. **Implement state persistence** between runs for better continuity
5. **Add notification integration** with monitoring systems or chat platforms

## Further Development

Potential enhancements for the system include:

1. Implementing machine learning for more adaptive baseline calculation
2. Adding correlation analysis across multiple metrics
3. Developing anomaly classification by pattern type
4. Creating metric-specific threshold profiles
5. Integrating with deployment and change management systems
