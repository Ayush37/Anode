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
            'DMLLatency',
            'SelectLatency'
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
    
    def detect_significant_anomalies(self, metric_name, min_percentage_change=100, min_consecutive=3, min_duration_minutes=15):
        """
        Detect significant anomalies in the metric data using statistical methods.
        
        Focus on major anomalies that:
        1. Exceed a high percentage change from baseline
        2. Persist for multiple consecutive data points
        3. Last for a minimum duration
        
        Args:
            metric_name: The name of the CloudWatch metric to analyze
            min_percentage_change: Minimum percentage change to consider as anomaly (default: 100%)
            min_consecutive: Minimum consecutive points needed to confirm anomaly (default: 3)
            min_duration_minutes: Minimum duration for an anomaly to be reported (default: 15 minutes)
            
        Returns:
            List of significant anomalies with timestamps and values
        """
        # Get metric data
        df = self.get_metric_data(metric_name)
        if df is None or len(df) < 30:  # Need sufficient data
            logger.warning(f"Insufficient data for {metric_name} anomaly analysis")
            return []
        
        # Set timestamp as index for time series analysis
        df = df.set_index('timestamp')
        
        # Quick data quality check
        min_val = df['value'].min()
        max_val = df['value'].max()
        mean_val = df['value'].mean()
        std_val = df['value'].std()
        
        logger.info(f"Data statistics for {metric_name}:")
        logger.info(f"  Min: {min_val:.6f}")
        logger.info(f"  Max: {max_val:.6f}")
        logger.info(f"  Mean: {mean_val:.6f}")
        logger.info(f"  Std Dev: {std_val:.6f}")

        # Calculate baseline using median of first 75% of data
        # Median is more robust to outliers than mean
        baseline_data = df.iloc[:int(len(df) * 0.75)]
        baseline = np.median(baseline_data['value'])
        logger.info(f"  Baseline (median): {baseline:.6f}")
        
        # Calculate a robust threshold using percentiles and std dev
        percentile_95 = np.percentile(baseline_data['value'], 95)
        std_threshold = baseline + 4 * std_val  # 4 standard deviations (more conservative)
        
        # Use the lower of the two as our threshold to be more conservative
        threshold = min(percentile_95, std_threshold)
        logger.info(f"  Anomaly threshold: {threshold:.6f}")
        
        # Identify potential anomalies - values that exceed both absolute and relative thresholds
        anomalies = []
        current_anomaly = None
        consecutive_count = 0
        
        for idx, row in df.iterrows():
            value = row['value']
            percentage_change = ((value - baseline) / baseline) * 100 if baseline > 0 else float('inf')
            
            # Check both absolute threshold and percentage change
            is_anomaly = value > threshold and percentage_change > min_percentage_change
            
            if is_anomaly:
                consecutive_count += 1
                
                if consecutive_count >= min_consecutive:
                    if current_anomaly is None:
                        # Start of a new anomaly
                        start_idx = idx - datetime.timedelta(minutes=(consecutive_count - 1) * 5)  # Assuming 5-min intervals
                        current_anomaly = {
                            'metric': metric_name,
                            'start_timestamp': start_idx,
                            'latest_timestamp': idx,
                            'actual_value': value,
                            'baseline': baseline,
                            'threshold': threshold,
                            'deviation': value - baseline,
                            'percentage_change': percentage_change,
                            'consecutive_points': consecutive_count,
                            'values': [value]
                        }
                    else:
                        # Update existing anomaly
                        current_anomaly['latest_timestamp'] = idx
                        current_anomaly['consecutive_points'] = consecutive_count
                        current_anomaly['values'].append(value)
                        
                        # Update peak value if this is higher
                        if value > current_anomaly['actual_value']:
                            current_anomaly['actual_value'] = value
                            current_anomaly['deviation'] = value - baseline
                            current_anomaly['percentage_change'] = percentage_change
            else:
                # Reset consecutive counter
                if current_anomaly is not None:
                    # Check if the anomaly meets minimum duration requirements
                    duration = current_anomaly['latest_timestamp'] - current_anomaly['start_timestamp']
                    duration_minutes = duration.total_seconds() / 60
                    
                    if duration_minutes >= min_duration_minutes:
                        # Calculate average value during anomaly
                        avg_value = sum(current_anomaly['values']) / len(current_anomaly['values'])
                        current_anomaly['average_value'] = avg_value
                        
                        # Calculate peak-to-baseline ratio
                        current_anomaly['peak_ratio'] = current_anomaly['actual_value'] / baseline if baseline > 0 else float('inf')
                        
                        # Add duration information
                        current_anomaly['duration_minutes'] = duration_minutes
                        
                        # Add to anomalies list
                        anomalies.append(current_anomaly)
                        logger.info(f"SIGNIFICANT ANOMALY DETECTED: {metric_name}")
                        logger.info(f"  Start: {current_anomaly['start_timestamp']}")
                        logger.info(f"  End: {current_anomaly['latest_timestamp']}")
                        logger.info(f"  Duration: {duration_minutes:.1f} minutes")
                        logger.info(f"  Peak: {current_anomaly['actual_value']:.6f} (baseline: {baseline:.6f})")
                        logger.info(f"  Change: {current_anomaly['percentage_change']:.2f}%")
                    else:
                        logger.debug(f"Anomaly too brief ({duration_minutes:.1f} min) - ignoring")
                
                # Reset tracking
                consecutive_count = 0
                current_anomaly = None
        
        # Check if there's an ongoing anomaly at the end of the data
        if current_anomaly is not None:
            duration = current_anomaly['latest_timestamp'] - current_anomaly['start_timestamp']
            duration_minutes = duration.total_seconds() / 60
            
            if duration_minutes >= min_duration_minutes:
                # Calculate average value during anomaly
                avg_value = sum(current_anomaly['values']) / len(current_anomaly['values'])
                current_anomaly['average_value'] = avg_value
                
                # Calculate peak-to-baseline ratio
                current_anomaly['peak_ratio'] = current_anomaly['actual_value'] / baseline if baseline > 0 else float('inf')
                
                # Add duration information
                current_anomaly['duration_minutes'] = duration_minutes
                
                # Add to anomalies list
                anomalies.append(current_anomaly)
                logger.info(f"ONGOING SIGNIFICANT ANOMALY DETECTED: {metric_name}")
                logger.info(f"  Start: {current_anomaly['start_timestamp']}")
                logger.info(f"  End: {current_anomaly['latest_timestamp']} (ongoing)")
                logger.info(f"  Duration so far: {duration_minutes:.1f} minutes")
                logger.info(f"  Peak: {current_anomaly['actual_value']:.6f} (baseline: {baseline:.6f})")
                logger.info(f"  Change: {current_anomaly['percentage_change']:.2f}%")
        
        # Sort anomalies by start time
        anomalies.sort(key=lambda x: x['start_timestamp'])
        
        logger.info(f"Found {len(anomalies)} significant anomalies for {metric_name}")
        return anomalies
    
    def monitor_all_metrics(self, min_percentage_change=100, min_consecutive=3, min_duration_minutes=15):
        """
        Monitor all RDS metrics and detect significant anomalies.
        
        Args:
            min_percentage_change: Minimum percentage change to consider as anomaly (default: 100%)
            min_consecutive: Minimum consecutive points needed to confirm anomaly (default: 3)
            min_duration_minutes: Minimum duration for an anomaly to be reported (default: 15 minutes)
        
        Returns:
            List of all detected significant anomalies
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
            anomalies = self.detect_significant_anomalies(
                metric_name=metric,
                min_percentage_change=min_percentage_change,
                min_consecutive=min_consecutive,
                min_duration_minutes=min_duration_minutes
            )
            all_anomalies.extend(anomalies)
        
        if all_anomalies:
            logger.info(f"Total significant anomalies detected: {len(all_anomalies)}")
        else:
            logger.info("No significant anomalies detected across all metrics")
            
        return all_anomalies


def main():
    """Main function to run the RDS monitoring."""
    parser = argparse.ArgumentParser(description='Monitor RDS instance metrics for anomalies')
    parser.add_argument('--instance', required=True, help='RDS instance identifier')
    parser.add_argument('--region', default='us-east-1', help='AWS region (default: us-east-1)')
    parser.add_argument('--days', type=int, default=14,
                        help='Number of days of historical data to retrieve (default: 14)')
    parser.add_argument('--min-change', type=float, default=100,
                        help='Minimum percentage change to be considered an anomaly (default: 100%)')
    parser.add_argument('--min-consecutive', type=int, default=3,
                        help='Minimum consecutive data points for anomaly (default: 3)')
    parser.add_argument('--min-duration', type=int, default=15,
                        help='Minimum duration in minutes for reported anomalies (default: 15)')
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
        anomalies = monitor.monitor_all_metrics(
            min_percentage_change=args.min_change,
            min_consecutive=args.min_consecutive,
            min_duration_minutes=args.min_duration
        )
        
        if anomalies:
            print("\n=== SIGNIFICANT ANOMALY SUMMARY ===")
            for anomaly in anomalies:
                print(f"Metric: {anomaly['metric']}")
                print(f"Anomaly started at: {anomaly['start_timestamp']}")
                print(f"Latest data point: {anomaly['latest_timestamp']}")
                print(f"Duration: {anomaly['duration_minutes']:.1f} minutes")
                print(f"Peak value: {anomaly['actual_value']:.6f}")
                print(f"Baseline value: {anomaly['baseline']:.6f}")
                print(f"Average during anomaly: {anomaly.get('average_value', anomaly['actual_value']):.6f}")
                print(f"Percentage change: {anomaly['percentage_change']:.2f}%")
                print(f"Peak-to-baseline ratio: {anomaly.get('peak_ratio', 0):.2f}x")
                print("---")
        else:
            print("No significant anomalies detected.")
    
    except Exception as e:
        logger.error(f"Error in monitoring script: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
