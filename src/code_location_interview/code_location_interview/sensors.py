"""
Dagster Sensors for ML Pipeline
"""

import dagster as dg
from datetime import datetime, timedelta
import math
from .jobs import training_pipeline_job
from dagster_duckdb import DuckDBResource

# Drift Detection Sensor
@dg.sensor(
    job=training_pipeline_job,
    minimum_interval_seconds=360,  # Check every hour
    name="drift_detection_sensor",
    description="Monitors feature statistics for drift and triggers retraining"
)
def drift_detection_sensor(context: dg.SensorEvaluationContext, database: DuckDBResource):
    """
    Simple drift detection using z-test on feature statistics.
    Compares current stats with baseline (7 days ago) and triggers training if drift detected.
    """
    
    with database.get_connection() as conn:
        # Get current feature statistics (most recent)
        current_stats = conn.execute("""
            SELECT feature_name, mean_value, std_value, computation_timestamp
            FROM feature_statistics 
            WHERE computation_timestamp = (SELECT MAX(computation_timestamp) FROM feature_statistics)
        """).fetchall()
        
        if not current_stats:
            context.log.info("No current feature statistics found")
            return
        
        # Get baseline statistics (7 days ago)
        # Get baseline statistics (previous 5 minutes)
        current_time = datetime.now()
        baseline_start = current_time - timedelta(hours=6)
        baseline_stats = conn.execute("""
            SELECT feature_name, mean_value, std_value, computation_timestamp
            FROM feature_statistics 
            WHERE computation_timestamp >= ? 
              AND computation_timestamp < ?
            ORDER BY computation_timestamp DESC
            LIMIT 1
        """, [baseline_start, current_time - timedelta(hours=1)]).fetchall()
        
        if not baseline_stats:
            context.log.info("No baseline statistics found for drift comparison")
            return
        
        # Convert to dictionaries for easy lookup
        current_dict = {row[0]: (row[1], row[2]) for row in current_stats}
        baseline_dict = {row[0]: (row[1], row[2]) for row in baseline_stats}
        
        drift_detected = False
        drift_features = []
        
        # Simple z-test for drift detection
        z_threshold = 2.0  # Corresponds to ~95% confidence
        
        for feature_name, (current_mean, current_std) in current_dict.items():
            if feature_name in baseline_dict:
                baseline_mean, baseline_std = baseline_dict[feature_name]
                
                # Skip if either std is None or 0 (avoid division by zero)
                if not all([current_std, baseline_std, current_std > 0, baseline_std > 0]):
                    continue
                
                # Calculate z-score for mean difference
                pooled_std = math.sqrt((current_std**2 + baseline_std**2) / 2)
                z_score = abs(current_mean - baseline_mean) / pooled_std
                
                if z_score > z_threshold:
                    drift_detected = True
                    drift_features.append(f"{feature_name} (z={z_score:.2f})")
        
        if drift_detected:
            context.log.warning(f"Drift detected in features: {drift_features}")
            
            # Get cursor to avoid duplicate triggers
            cursor = context.cursor or "0"
            current_timestamp = current_stats[0][3].isoformat()
            
            # Only trigger if we haven't seen this timestamp before
            if current_timestamp != cursor:
                context.log.info("Triggering training pipeline due to drift detection")
                
                return dg.SensorResult(
                    run_requests=[
                        dg.RunRequest(
                            run_key=f"drift_training_{current_timestamp}",
                            tags={
                                "trigger_reason": "drift_detected",
                                "drift_features": ",".join(drift_features)
                            }
                        )
                    ],
                    cursor=current_timestamp
                )
            else:
                context.log.info("Drift detected but training already triggered for this timestamp")
        else:
            context.log.info("No significant drift detected")