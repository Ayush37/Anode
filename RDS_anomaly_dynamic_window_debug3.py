import boto3
import pandas as pd
import numpy as np
import logging
import datetime
import argparse
import sys
import json
from botocore.exceptions import ClientError
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
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)

class RDSMonitor:
    def __init__(self, instance_name, region='us-east-1', debug=False):
        """Initialize the RDS monitor with instance name and region."""
        self.instance_name = instance_name
        self.region = region
        self.debug = debug
        self.metrics = [
            'ReadLatency',
            'WriteLatency',
            'UpdateLatency',
            'DeleteLatency',
            'DMLLatency'
        ]
        
        # Initialize boto3 clients
        try:
            self.cloudwatch = boto3.client('cloudwatch', region_name=self.region)
            # Validate AWS credentials by making a simple API call
            self.cloudwatch.list_metrics(Namespace='AWS/RDS', Dimensions=[
                {'Name': 'DBInstanceIdentifier', 'Value': self.instance_name}
            ])
            logger.info(f"Successfully connected to CloudWatch in {self.region}")
            logger.info(f"Initialized monitoring for RDS instance: {self.instance_name}")
        except ClientError as e:
            logger.error(f"AWS API Error: {e}")
            logger.error("Please check your AWS credentials and instance name")
            raise
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise
    
    def list_available_metrics(self):
        """List all available metrics for the given RDS instance."""
        try:
            logger.info(f"Listing available metrics for {self.instance_name}...")
            response = self.cloudwatch.list_metrics(
                Namespace='AWS/RDS',
                Dimensions=[
                    {'Name': 'DBInstanceIdentifier', 'Value': self.instance_name}
                ]
            )
            
            if self.debug:
                logger.info(f"List metrics response: {json.dumps(response, default=str)}")
            
            available_metrics = [metric['MetricName'] for metric in response['Metrics']]
            logger.info(f"Available metrics: {available_metrics}")
            
            # Check if our target metrics are available
            for metric in self.metrics:
                if metric in available_metrics:
                    logger.info(f"✓ {metric} is available")
                else:
                    logger.warning(f"✗ {metric} is NOT available")
                    
            return available_metrics
        except Exception as e:
            logger.error(f"Error listing metrics: {e}")
            return []
    
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
        
        if self.debug:
            # Print detailed information about the request
            logger.debug(f"DB Instance: {self.instance_name}")
            logger.debug(f"Region: {self.region}")
            logger.debug(f"Full date range: {start_time} to {end_time}")
            logger.debug(f"Recent date range: {recent_start_time} to {end_time}")
        
        # Prepare metric queries - different periods for different time ranges
        recent_query = {
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
        }
        
        historical_query = {
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
        
        # First, try a simple query to see if any data exists for this metric
        try:
            test_response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/RDS',
                MetricName=metric_name,
                Dimensions=[
                    {
                        'Name': 'DBInstanceIdentifier',
                        'Value': self.instance_name
                    }
                ],
                StartTime=end_time - datetime.timedelta(hours=1),
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )
            
            if self.debug:
                logger.debug(f"Test query response: {json.dumps(test_response, default=str)}")
                
            if not test_response['Datapoints']:
                logger.warning(f"No recent data available for {metric_name} in the last hour")
                logger.warning("This may indicate the metric is not collected for this instance type")
        except ClientError as e:
            logger.error(f"AWS API Error in test query: {e}")
        
        # Get recent data (last 2 days, 5-min intervals)
        recent_df = pd.DataFrame(columns=['timestamp', 'value'])
        try:
            recent_response = self.cloudwatch.get_metric_data(
                MetricDataQueries=[recent_query],
                StartTime=recent_start_time,
                EndTime=end_time
            )
            
            if self.debug:
                logger.debug(f"Recent data response: {json.dumps(recent_response, default=str)}")
            
            recent_timestamps = recent_response['MetricDataResults'][0]['Timestamps']
            recent_values = recent_response['MetricDataResults'][0]['Values']
            
            if recent_timestamps and recent_values:
                recent_df = pd.DataFrame({
                    'timestamp': recent_timestamps,
                    'value': recent_values
                })
                logger.info(f"Retrieved {len(recent_df)} recent data points for {metric_name}")
            else:
                logger.warning(f"No recent data points found for {metric_name}")
                
            # Check if we have partial data
            if len(recent_timestamps) > 0 and len(recent_timestamps) < 576:  # 2 days at 5-min intervals
                logger.warning(f"Retrieved partial data for {metric_name}: {len(recent_timestamps)} points " +
                              f"(expected ~576 for 2 days at 5-min intervals)")
                
        except ClientError as e:
            logger.error(f"AWS API Error retrieving recent data: {e}")
        except Exception as e:
            logger.error(f"Error processing recent data: {e}")
        
        # Get historical data (days 3-14, 1-hour intervals)
        historical_df = pd.DataFrame(columns=['timestamp', 'value'])
        try:
            historical_response = self.cloudwatch.get_metric_data(
                MetricDataQueries=[historical_query],
                StartTime=start_time,
                EndTime=recent_start_time
            )
            
            if self.debug:
                logger.debug(f"Historical data response: {json.dumps(historical_response, default=str)}")
            
            historical_timestamps = historical_response['MetricDataResults'][0]['Timestamps']
            historical_values = historical_response['MetricDataResults'][0]['Values']
            
            if historical_timestamps and historical_values:
                historical_df = pd.DataFrame({
                    'timestamp': historical_timestamps,
                    'value': historical_values
                })
                logger.info(f"Retrieved {len(historical_df)} historical data points for {metric_name}")
            else:
                logger.warning(f"No historical data points found for {metric_name}")
                
            # Check if we have partial data
            if len(historical_timestamps) > 0 and len(historical_timestamps) < 288:  # 12 days at 1-hour intervals
                logger.warning(f"Retrieved partial historical data for {metric_name}: {len(historical_timestamps)} points " +
                              f"(expected ~288 for 12 days at 1-hour intervals)")
                
        except ClientError as e:
            logger.error(f"AWS API Error retrieving historical data: {e}")
        except Exception as e:
            logger.error(f"Error processing historical data: {e}")
        
        # Combine and sort data
        df = pd.concat([historical_df, recent_df]).sort_values('timestamp')
        
        if len(df) == 0:
            logger.warning(f"No data points found for {metric_name}")
            
            # Try checking if the instance exists and has metrics
            try:
                rds_client = boto3.client('rds', region_name=self.region)
                instances = rds_client.describe_db_instances(DBInstanceIdentifier=self.instance_name)
                
                if instances['DBInstances']:
                    instance = instances['DBInstances'][0]
                    logger.info(f"RDS instance exists: {instance['DBInstanceIdentifier']}")
                    logger.info(f"Instance status: {instance['DBInstanceStatus']}")
                    logger.info(f"Instance class: {instance['DBInstanceClass']}")
                    logger.info(f"Engine: {instance['Engine']} {instance['EngineVersion']}")
                else:
                    logger.error(f"RDS instance {self.instance_name} not found")
            except ClientError as e:
                logger.error(f"Error checking RDS instance: {e}")
            except Exception as e:
                logger.error(f"Error in RDS instance lookup: {e}")
                
            return None
        
        logger.info(f"Total: Retrieved {len(df)} data points for {metric_name}")
        
        # Print sample data if in debug mode
        if self.debug and not df.empty:
            logger.debug(f"Sample data for {metric_name}:")
            logger.debug(f"First 3 rows: {df.head(3).to_dict('records')}")
            logger.debug(f"Last 3 rows: {df.tail(3).to_dict('records')}")
            
        return df
    
    def detect_anomalies_arima(self, metric_name, confidence=0.99, detection_window=24):
        """
        Detect anomalies in the metric data using ARIMA modeling.
        
        Args:
            metric_name: The name of the CloudWatch metric to analyze
            confidence: Confidence level for anomaly detection (default: 0.99)
            detection_window: Number of recent data points to check for anomalies (default: 24)
            
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
        
        # Quick data quality check
        logger.info(f"Data statistics for {metric_name}:")
        logger.info(f"  Min: {df['value'].min():.6f}")
        logger.info(f"  Max: {df['value'].max():.6f}")
        logger.info(f"  Mean: {df['value'].mean():.6f}")
        logger.info(f"  Std Dev: {df['value'].std():.6f}")
        
        # Check if we have enough data for detection window
        if len(df) <= detection_window:
            detection_window = max(1, len(df) // 4)  # Fall back to 25% of data if not enough
            logger.warning(f"Adjusted detection window to {detection_window} due to limited data")
        
        train_data = df.iloc[:-detection_window]
        test_data = df.iloc[-detection_window:]
        
        logger.info(f"Training data points: {len(train_data)}")
        logger.info(f"Test data points: {len(test_data)}")
        
        # Check if training data is too small
        if len(train_data) < 10:
            logger.warning(f"Training data too small for {metric_name} ARIMA analysis")
            return []
        
        # Check if the time series is stationary
        try:
            result = adfuller(train_data['value'].dropna())
            logger.info(f"ADF Statistic: {result[0]:.6f}")
            logger.info(f"p-value: {result[1]:.6f}")
            
            d = 0
            if result[1] > 0.05:  # p-value > 0.05 means non-stationary
                d = 1  # Set differencing parameter
                logger.info(f"Series is non-stationary (p > 0.05), setting d=1 for ARIMA")
            else:
                logger.info(f"Series is stationary (p <= 0.05), setting d=0 for ARIMA")
        except Exception as e:
            logger.warning(f"Error in stationarity test: {e}")
            d = 1  # Default to 1st order differencing in case of error
            
        anomalies = []
        
        try:
            # Fit ARIMA model (p=1, d=determined, q=1) on training data
            logger.info(f"Fitting ARIMA(1,{d},1) model on {len(train_data)} data points")
            model = ARIMA(train_data['value'], order=(1, d, 1))
            model_fit = model.fit()
            
            if self.debug:
                logger.debug(f"ARIMA model summary:\n{model_fit.summary()}")
            
            # Forecast for the test window
            logger.info(f"Forecasting {len(test_data)} steps ahead")
            forecast_result = model_fit.forecast(steps=len(test_data))
            
            # Fix: Convert forecast to list if it's a pandas Series
            if isinstance(forecast_result, pd.Series):
                forecast_values = forecast_result.values
            else:
                forecast_values = np.array(forecast_result)
                
            logger.info(f"Generated {len(forecast_values)} forecast values")
            
            # Calculate the residuals and their standard deviation from training data
            residuals = model_fit.resid
            resid_std = np.std(residuals)
            logger.info(f"Residual standard deviation: {resid_std:.6f}")
            
            # Calculate confidence interval
            z_score = abs(np.log(1 / (1 - confidence)))
            margin = z_score * resid_std
            logger.info(f"Using {confidence*100}% confidence interval (margin: ±{margin:.6f})")
            
            # Compare actual values with forecasts
            anomaly_detected = False
            consecutive_anomalies = 0
            min_consecutive = 2  # Require at least 2 consecutive anomalies to reduce false positives
            
            logger.info(f"Checking {len(test_data)} points for anomalies")
            
            # Print first few actual vs forecast values in debug mode
            if self.debug and len(test_data) > 0 and len(forecast_values) > 0:
                sample_size = min(5, len(test_data), len(forecast_values))
                logger.debug("Sample actual vs forecast values:")
                for i in range(sample_size):
                    idx = test_data.index[i]
                    actual = test_data['value'].iloc[i]
                    predicted = forecast_values[i]
                    logger.debug(f"  Point {i}: {idx} - Actual: {actual:.6f}, Forecast: {predicted:.6f}, " +
                                f"Range: [{predicted-margin:.6f}, {predicted+margin:.6f}]")
            
            # Verify we have enough forecast values
            if len(forecast_values) < len(test_data):
                logger.warning(f"Forecast length ({len(forecast_values)}) is less than test data length ({len(test_data)})")
                # Adjust test_data to match forecast_values length
                test_data = test_data.iloc[:len(forecast_values)]
                
            # Convert test_data to list for easier iteration
            test_indices = test_data.index.tolist()
            test_values = test_data['value'].tolist()
            
            # Iterate through test data and forecasts
            for i in range(len(test_values)):
                idx = test_indices[i]
                actual = test_values[i]
                
                # Safe access to forecast values
                if i < len(forecast_values):
                    predicted = forecast_values[i]
                else:
                    logger.warning(f"Missing forecast for index {i}, skipping")
                    continue
                    
                lower_bound = predicted - margin
                upper_bound = predicted + margin
                
                # Check if the actual value is outside the confidence interval
                is_point_anomaly = (actual < lower_bound) or (actual > upper_bound)
                
                # Handle anomaly tracking
                if is_point_anomaly:
                    consecutive_anomalies += 1
                    logger.info(f"Potential anomaly at {idx}: Actual={actual:.6f}, " +
                               f"Predicted={predicted:.6f}, Range=[{lower_bound:.6f}, {upper_bound:.6f}]")
                    
                    # Only record as anomaly if we have consecutive anomalies
                    if consecutive_anomalies >= min_consecutive:
                        if not anomaly_detected:
                            # First time we're detecting this anomaly period
                            anomaly_detected = True
                            # Record the start of the anomaly (min_consecutive points back)
                            anomaly_start_idx = max(0, i - consecutive_anomalies + 1)
                            if anomaly_start_idx < len(test_indices):
                                anomaly_start_time = test_indices[anomaly_start_idx]
                            else:
                                anomaly_start_time = idx  # Fallback
                            
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
                            logger.info(f"  Current: {actual:.6f}, Expected: {predicted:.6f}, " +
                                       f"Range: [{lower_bound:.6f}, {upper_bound:.6f}]")
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
                    if consecutive_anomalies > 0:
                        logger.info(f"Normal value at {idx}: Anomaly streak ended after {consecutive_anomalies} points")
                    consecutive_anomalies = 0
                    anomaly_detected = False
                
            if not anomalies:
                logger.info(f"No anomalies detected for {metric_name}")
                
            return anomalies
                
        except Exception as e:
            logger.error(f"Error in ARIMA analysis for {metric_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def monitor_all_metrics(self):
        """
        Monitor all RDS metrics and detect anomalies.
        
        Returns:
            List of all detected anomalies
        """
        # First, check what metrics are available
        available_metrics = self.list_available_metrics()
        
        all_anomalies = []
        
        for metric in self.metrics:
            # Skip metrics that definitely aren't available
            if available_metrics and metric not in available_metrics:
                logger.warning(f"Skipping {metric} as it's not available for this instance")
                continue
                
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
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger('').setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    try:
        # Print AWS configuration information
        logger.info("Checking AWS Configuration...")
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials:
            logger.info(f"AWS credentials found: {credentials.access_key[:4]}...{credentials.access_key[-4:]}")
            logger.info(f"Using region: {args.region}")
        else:
            logger.warning("No AWS credentials found. Check your AWS configuration.")
        
        # Initialize and run the monitor
        monitor = RDSMonitor(args.instance, args.region, debug=args.debug)
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
        import traceback
        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
