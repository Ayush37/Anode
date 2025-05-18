import boto3
import pandas as pd
import numpy as np
import logging
import datetime
import argparse
import sys
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from dateutil.relativedelta import relativedelta

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
    
    def get_metric_data(self, metric_name, period=300, days=14):
        """
        Retrieve CloudWatch metric data for the specified period.
        
        Args:
            metric_name: The name of the CloudWatch metric to retrieve
            period: The period in seconds for data points (default: 300 seconds = 5 minutes)
            days: Number of days of historical data to retrieve (default: 14 days)
            
        Returns:
            A pandas DataFrame with timestamps and metric values
        """
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(days=days)
        
        logger.info(f"Retrieving {metric_name} data from {start_time} to {end_time}")
        
        response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/RDS',
            MetricName=metric_name,
            Dimensions=[
                {
                    'Name': 'DBInstanceIdentifier',
                    'Value': self.instance_name
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            Statistics=['Average']
        )
        
        datapoints = response['Datapoints']
        
        if not datapoints:
            logger.warning(f"No data points found for {metric_name}")
            return None
            
        # Convert to DataFrame and sort by timestamp
        df = pd.DataFrame(datapoints)
        df = df.rename(columns={'Average': 'value', 'Timestamp': 'timestamp'})
        df = df.sort_values('timestamp')
        
        logger.info(f"Retrieved {len(df)} data points for {metric_name}")
        return df[['timestamp', 'value']]
    
    def detect_anomalies_arima(self, metric_name, confidence=0.99):
        """
        Detect anomalies in the metric data using ARIMA modeling.
        
        Args:
            metric_name: The name of the CloudWatch metric to analyze
            confidence: Confidence level for anomaly detection (default: 0.99)
            
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
        
        # Check if the time series is stationary
        result = adfuller(df['value'].dropna())
        d = 0
        if result[1] > 0.05:  # p-value > 0.05 means non-stationary
            d = 1  # Set differencing parameter
            
        try:
            # Fit ARIMA model (p=1, d=determined, q=1)
            model = ARIMA(df['value'], order=(1, d, 1))
            model_fit = model.fit()
            
            # Get the last observation
            last_obs = df['value'].iloc[-1]
            last_time = df.index[-1]
            
            # Get the predicted value for the last observation
            predict = model_fit.predict(start=len(df)-1, end=len(df)-1)
            forecast = predict.iloc[0]
            
            # Calculate the residuals and their standard deviation
            residuals = model_fit.resid
            resid_std = np.std(residuals)
            
            # Calculate confidence interval
            z_score = abs(np.log(1 / (1 - confidence)))
            margin = z_score * resid_std
            lower_bound = forecast - margin
            upper_bound = forecast + margin
            
            # Check if the actual value is outside the confidence interval
            is_anomaly = (last_obs < lower_bound) or (last_obs > upper_bound)
            
            if is_anomaly:
                anomaly = {
                    'metric': metric_name,
                    'timestamp': last_time,
                    'actual_value': last_obs,
                    'expected_value': forecast,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                logger.info(f"ANOMALY DETECTED: {metric_name} at {last_time}")
                logger.info(f"  Actual: {last_obs}, Expected: {forecast}, Range: [{lower_bound:.6f}, {upper_bound:.6f}]")
                return [anomaly]
            else:
                logger.info(f"No anomalies detected for {metric_name}")
                return []
                
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
    
    args = parser.parse_args()
    
    try:
        monitor = RDSMonitor(args.instance, args.region)
        anomalies = monitor.monitor_all_metrics()
        
        if anomalies:
            print("\n=== ANOMALY SUMMARY ===")
            for anomaly in anomalies:
                print(f"Metric: {anomaly['metric']}")
                print(f"Timestamp: {anomaly['timestamp']}")
                print(f"Actual value: {anomaly['actual_value']}")
                print(f"Expected range: [{anomaly['lower_bound']:.6f}, {anomaly['upper_bound']:.6f}]")
                print("---")
    
    except Exception as e:
        logger.error(f"Error in monitoring script: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
