import boto3
import pandas as pd
import numpy as np
import logging
import datetime
import argparse
import sys
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Configure logging
logging.basicConfig(
    filename='rds_monitor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)

class RDSMonitor:
    def __init__(self, instance_name, region='us-east-1'):
        """Initialize the RDS monitor with instance name and region."""
        self.instance_name = instance_name
        self.region = region
        self.metrics = [
            'ReadLatency',
            'WriteLatency',
            'UpdateLatency',
            'DeleteLatency',
            'DMLLatency'
        ]
        
        # Initialize boto3 clients
        self.cloudwatch = boto3.client('cloudwatch', region_name=self.region)
        logger.info(f"Initialized monitoring for RDS instance: {self.instance_name} in {self.region}")
    
    def get_metric_data(self, metric_name, days=14):
        """
        Retrieve CloudWatch metric data for the specified period using GetMetricData API.
        
        This method uses different granularity for different time periods:
        - Recent data (2 days): 5-minute intervals
        - Older data (12 days): 1-hour intervals
        
        Args:
            metric_name: The name of the CloudWatch metric to retrieve
            days: Number of days of historical data to retrieve (default: 14 days)
            
        Returns:
            A pandas DataFrame with timestamps and metric values
        """
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(days=days)
        recent_start_time = end_time - datetime.timedelta(days=2)  # Last 2 days
        
        logger.info(f"Retrieving {metric_name} data from {start_time} to {end_time}")
        
        # Prepare metric queries - different periods for different time ranges
        metric_query = [
            # Recent data: 5-minute intervals (300 seconds) for the last 2 days
            {
                'Id': 'recent',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/RDS',
                        'MetricName': metric_name,
                        'Dimensions': [
                            {
                                'Name': 'DBInstanceIdentifier',
                                'Value': self.instance_name
                            }
                        ]
                    },
                    'Period': 300,  # 5 minutes
                    'Stat': 'Average'
                },
                'ReturnData': True
            },
            # Historical data: 1-hour intervals (3600 seconds) for days 3-14
            {
                'Id': 'historical',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/RDS',
                        'MetricName': metric_name,
                        'Dimensions': [
                            {
                                'Name': 'DBInstanceIdentifier',
                                'Value': self.instance_name
                            }
                        ]
                    },
                    'Period': 3600,  # 1 hour
                    'Stat': 'Average'
                },
                'ReturnData': True
            }
        ]
        
        # Get recent data (last 2 days, 5-min intervals)
        recent_response = self.cloudwatch.get_metric_data(
            MetricDataQueries=[metric_query[0]],
            StartTime=recent_start_time,
            EndTime=end_time
        )
        
        # Get historical data (days 3-14, 1-hour intervals)
        historical_response = self.cloudwatch.get_metric_data(
            MetricDataQueries=[metric_query[1]],
            StartTime=start_time,
            EndTime=recent_start_time
        )
        
        # Process recent data
        recent_timestamps = recent_response['MetricDataResults'][0]['Timestamps']
        recent_values = recent_response['MetricDataResults'][0]['Values']
        
        # Process historical data
        historical_timestamps = historical_response['MetricDataResults'][0]['Timestamps']
        historical_values = historical_response['MetricDataResults'][0]['Values']
        
        # Create DataFrames and combine
        if recent_timestamps and recent_values:
            recent_df = pd.DataFrame({
                'timestamp': recent_timestamps,
                'value': recent_values
            })
        else:
            recent_df = pd.DataFrame(columns=['timestamp', 'value'])
            logger.warning(f"No recent data points found for {metric_name}")
        
        if historical_timestamps and historical_values:
            historical_df = pd.DataFrame({
                'timestamp': historical_timestamps,
                'value': historical_values
            })
        else:
            historical_df = pd.DataFrame(columns=['timestamp', 'value'])
            logger.warning(f"No historical data points found for {metric_name}")
        
        # Combine and sort data
        df = pd.concat([historical_df, recent_df]).sort_values('timestamp')
        
        if len(df) == 0:
            logger.warning(f"No data points found for {metric_name}")
            return None
        
        logger.info(f"Retrieved {len(df)} data points for {metric_name} " 
                    f"({len(recent_df)} recent, {len(historical_df)} historical)")
        
        return df
    
    def detect_anomalies_arima(self, metric_name, confidence=0.99, detection_window=24):
        """
        Detect anomalies in the metric data using ARIMA modeling.
        
        Args:
            metric_name: The name of the CloudWatch metric to analyze
            confidence: Confidence level for anomaly detection (default: 0.99)
            detection_window: Number of recent data points to check for anomalies (default: 24, 
                             which is 2 hours with 5-minute intervals)
            
        Returns:
            List of anomalies with timestamps and values
        """
        # Get metric data
        df = self.get_metric_data(metric_name)
        if df is None or len(df) < 30:  # Need sufficient data for ARIMA
            logger.warning(f"Insufficient data for {metric_name} ARIMA analysis")
            return []
        
        # Set timestamp as index for time series analysis
        df = df.set_index('timestamp')
        
        # Split data: training set and detection window
        if len(df) <= detection_window:
            detection_window = max(1, len(df) // 4)  # Fall back to 25% of data if not enough
        
        train_data = df.iloc[:-detection_window]
        test_data = df.iloc[-detection_window:]
        
        # Check if the time series is stationary
        result = adfuller(train_data['value'].dropna())
        d = 0
        if result[1] > 0.05:  # p-value > 0.05 means non-stationary
            d = 1  # Set differencing parameter
            
        anomalies = []
        
        try:
            # Fit ARIMA model (p=1, d=determined, q=1) on training data
            model = ARIMA(train_data['value'], order=(1, d, 1))
            model_fit = model.fit()
            
            # Forecast for the test window
            forecast = model_fit.forecast(steps=detection_window)
            
            # Calculate the residuals and their standard deviation from training data
            residuals = model_fit.resid
            resid_std = np.std(residuals)
            
            # Calculate confidence interval
            z_score = abs(np.log(1 / (1 - confidence)))
            margin = z_score * resid_std
            
            # Compare actual values with forecasts
            anomaly_detected = False
            consecutive_anomalies = 0
            min_consecutive = 2  # Require at least 2 consecutive anomalies to reduce false positives
            
            for i, (idx, actual) in enumerate(test_data['value'].items()):
                # Get the predicted value for this time point
                predicted = forecast[i]
                lower_bound = predicted - margin
                upper_bound = predicted + margin
                
                # Check if the actual value is outside the confidence interval
                is_point_anomaly = (actual < lower_bound) or (actual > upper_bound)
                
                # Handle anomaly tracking
                if is_point_anomaly:
                    consecutive_anomalies += 1
                    
                    # Only record as anomaly if we have consecutive anomalies
                    if consecutive_anomalies >= min_consecutive:
                        if not anomaly_detected:
                            # First time we're detecting this anomaly period
                            anomaly_detected = True
                            # Record the start of the anomaly (min_consecutive points back)
                            anomaly_start_idx = max(0, i - consecutive_anomalies + 1)
                            anomaly_start_time = test_data.index[anomaly_start_idx]
                            
                            anomaly = {
                                'metric': metric_name,
                                'start_timestamp': anomaly_start_time,
                                'latest_timestamp': idx,
                                'actual_value': actual,
                                'expected_value': predicted,
                                'lower_bound': lower_bound,
                                'upper_bound': upper_bound,
                                'deviation': abs(actual - predicted),
                                'percentage_change': abs((actual - predicted) / predicted) * 100 if predicted != 0 else float('inf')
                            }
                            anomalies.append(anomaly)
                            logger.info(f"ANOMALY DETECTED: {metric_name} starting at {anomaly_start_time}")
                            logger.info(f"  Current: {actual}, Expected: {predicted}, Range: [{lower_bound:.6f}, {upper_bound:.6f}]")
                        else:
                            # Update the existing anomaly with the latest timestamp
                            anomalies[-1]['latest_timestamp'] = idx
                            
                            # Update if this is a more severe deviation
                            if abs(actual - predicted) > anomalies[-1]['deviation']:
                                anomalies[-1]['actual_value'] = actual
                                anomalies[-1]['expected_value'] = predicted
                                anomalies[-1]['lower_bound'] = lower_bound
                                anomalies[-1]['upper_bound'] = upper_bound
                                anomalies[-1]['deviation'] = abs(actual - predicted)
                                
                                percentage_change = abs((actual - predicted) / predicted) * 100 if predicted != 0 else float('inf')
                                anomalies[-1]['percentage_change'] = percentage_change
                else:
                    # Reset streak if we see a normal value
                    consecutive_anomalies = 0
                    anomaly_detected = False
                
            if not anomalies:
                logger.info(f"No anomalies detected for {metric_name}")
                
            return anomalies
                
        except Exception as e:
            logger.error(f"Error in ARIMA analysis for {metric_name}: {e}")
            return []
    
    def monitor_all_metrics(self):
        """
        Monitor all RDS metrics and detect anomalies.
        
        Returns:
            List of all detected anomalies
        """
        all_anomalies = []
        
        for metric in self.metrics:
            logger.info(f"Analyzing {metric}...")
            anomalies = self.detect_anomalies_arima(metric)
            all_anomalies.extend(anomalies)
        
        if all_anomalies:
            logger.info(f"Total anomalies detected: {len(all_anomalies)}")
        else:
            logger.info("No anomalies detected across all metrics")
            
        return all_anomalies


def main():
    """Main function to run the RDS monitoring."""
    parser = argparse.ArgumentParser(description='Monitor RDS instance metrics for anomalies')
    parser.add_argument('--instance', required=True, help='RDS instance identifier')
    parser.add_argument('--region', default='us-east-1', help='AWS region (default: us-east-1)')
    parser.add_argument('--window', type=int, default=24, 
                        help='Number of recent data points to check for anomalies (default: 24)')
    parser.add_argument('--days', type=int, default=14,
                        help='Number of days of historical data to retrieve (default: 14)')
    
    args = parser.parse_args()
    
    try:
        monitor = RDSMonitor(args.instance, args.region)
        anomalies = monitor.monitor_all_metrics()
        
        if anomalies:
            print("\n=== ANOMALY SUMMARY ===")
            for anomaly in anomalies:
                print(f"Metric: {anomaly['metric']}")
                print(f"Anomaly started at: {anomaly['start_timestamp']}")
                print(f"Latest data point: {anomaly['latest_timestamp']}")
                print(f"Duration: {anomaly['latest_timestamp'] - anomaly['start_timestamp']}")
                print(f"Actual value: {anomaly['actual_value']:.6f}")
                print(f"Expected range: [{anomaly['lower_bound']:.6f}, {anomaly['upper_bound']:.6f}]")
                print(f"Deviation: {anomaly['deviation']:.6f}")
                print(f"Percentage change: {anomaly['percentage_change']:.2f}%")
                print("---")
        else:
            print("No anomalies detected.")
    
    except Exception as e:
        logger.error(f"Error in monitoring script: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
