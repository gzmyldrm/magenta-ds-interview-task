"""
Testing Assets for ML Pipeline

Provides utilities to test sensors and trigger events for development/testing.
"""

import logging
import sys
from datetime import datetime

import polars as pl
from dagster import asset, get_dagster_logger
from dagster_duckdb import DuckDBResource

log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

group_name = "testing"


@asset(
    group_name=group_name,
    deps=["feature_statistics_data"],
    description="Inject extreme feature statistics to trigger drift detection sensor"
)
def inject_drift_test_data(feature_statistics_data, database: DuckDBResource):
    """
    Injects extreme feature statistics to test drift detection sensor.
    
    Uses the existing feature_statistics_data asset as input and creates
    statistical outliers that should trigger the z-test in the drift sensor.
    
    Args:
        feature_statistics_data: Current feature statistics from feature engineering
        
    Returns:
        polars.DataFrame: Injected extreme feature statistics
    """
    logger.info("Starting drift test data injection using existing feature statistics")
    
    # Use the feature_statistics_data asset as input
    if feature_statistics_data is None or len(feature_statistics_data) == 0:
        logger.error("No feature statistics data available from upstream asset")
        raise ValueError("feature_statistics_data asset is empty. Run feature engineering first.")
    
    logger.info(f"Using {len(feature_statistics_data)} existing feature statistics as baseline")
    
        
    # Convert input polars DataFrame to list for processing
    stats_data = feature_statistics_data.to_dicts()
    
    # Create extreme values based on existing statistics
    current_timestamp = datetime.now()
    extreme_multipliers = [10.0, -8.0, 15.0, -5.0, 12.0, -3.0, 20.0]
    
    extreme_stats_list = []
    injected_features = []
    z_scores = []
    
    for i, stat_row in enumerate(stats_data[:7]):  # Limit to first 7 features
        feature_name = stat_row["feature_name"]
        current_mean = stat_row["mean_value"] 
        current_std = stat_row["std_value"]
        
        if current_mean is not None and current_std is not None and current_std > 0:
            # Create extreme values that will definitely trigger drift
            multiplier = extreme_multipliers[i % len(extreme_multipliers)]
            
            # For negative multipliers, shift by a large amount instead of multiply
            if multiplier < 0:
                extreme_mean = current_mean + (abs(multiplier) * current_std * 5)  # Large shift
            else:
                extreme_mean = current_mean * multiplier  # Large multiplication
            
            extreme_std = max(current_std * 1.2, 0.1)  # Slightly increase std
            
            # Calculate expected z-score
            pooled_std = ((current_std**2 + extreme_std**2) / 2) ** 0.5
            expected_z_score = abs(extreme_mean - current_mean) / pooled_std if pooled_std > 0 else 999
            
            # Create record for new statistics
            extreme_stats_list.append({
                "feature_name": feature_name,
                "mean_value": extreme_mean,
                "std_value": extreme_std, 
                "computation_timestamp": current_timestamp
            })
            
            injected_features.append({
                "feature_name": feature_name,
                "original_mean": current_mean,
                "extreme_mean": extreme_mean,
                "z_score": expected_z_score,
                "multiplier": multiplier
            })
            z_scores.append(expected_z_score)
            
            logger.info(f"Prepared {feature_name}: {current_mean:.2f} → {extreme_mean:.2f} (z-score: {expected_z_score:.2f})")
    
    # Create polars DataFrame for the extreme statistics
    extreme_stats_df = pl.DataFrame(extreme_stats_list)
    
    # Insert into database
    with database.get_connection() as conn:
        conn.execute("INSERT INTO feature_statistics SELECT * FROM extreme_stats_df")
        
        # Verify insertion
        verification_count = conn.execute("""
            SELECT COUNT(*) 
            FROM feature_statistics 
            WHERE computation_timestamp = ?
        """, [current_timestamp]).fetchone()[0]
    
    logger.info(f"Successfully injected {verification_count} extreme feature statistics")
    logger.info(f"Injection timestamp: {current_timestamp}")
    logger.info(f"Average z-score: {sum(z_scores)/len(z_scores):.2f}")
    logger.info(f"All z-scores > 2.0: {all(z > 2.0 for z in z_scores)}")
    
    logger.warning("Drift detection sensor should trigger within the next hour!")
    logger.warning("Monitor the sensor logs in Dagster UI for drift detection")
    
    # Return the extreme statistics DataFrame as asset output
    return extreme_stats_df
