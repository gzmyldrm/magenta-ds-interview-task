import logging
import sys
from pathlib import Path
import datetime

import polars as pl
from dagster import asset, get_dagster_logger
from dagster_duckdb import DuckDBResource

# Import ETL pipeline
sys.path.append(str(Path(__file__).parent.parent.parent))
from .lib.etl import ContractFeatureETL

log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

group_name = "feature_engineering"


@asset(
    group_name=group_name,
    deps=["raw_core_data", "raw_usage_info", "raw_customer_interactions"]
)
def processed_features(raw_core_data, raw_usage_info, raw_customer_interactions, database: DuckDBResource):
    """
    Process raw data through ETL pipeline to create ready features.
    
    Uses raw data assets directly, processes through ETL pipeline,
    and stores processed features back to database.
    """
    logger.info("Starting feature engineering for processed_features")
    
    logger.info("Using polars data from ingestion assets")
    core_data_pl = raw_core_data
    usage_info_pl = raw_usage_info
    customer_interactions_pl = raw_customer_interactions
    
    logger.info(f"Processing data - Core: {len(core_data_pl)}, Usage: {len(usage_info_pl)}, Interactions: {len(customer_interactions_pl)}")
    
    with database.get_connection() as conn:
        
        # Initialize ETL pipeline
        logger.info("Initializing ETL pipeline")
        etl_pipeline = ContractFeatureETL()
        
        # Process data through ETL pipeline
        logger.info("Processing features through ETL pipeline")
        try:
            processed_df = etl_pipeline.run_transform(
                core_data=core_data_pl,
                usage_info=usage_info_pl,
                customer_interactions=customer_interactions_pl
            )
            
            logger.info(f"ETL processing completed. Features shape: {processed_df.shape}")

            # Add created_at timestamp column with current time
            processed_df = processed_df.with_columns(
                pl.lit(datetime.datetime.now()).alias("created_at")
            )
            
            # Store processed features in database directly with polars
            logger.info("Storing processed features in database")
            
            # Insert new processed features (append to existing data)
            # Use INSERT OR REPLACE to handle duplicates based on rating_account_id
            # Add created_at timestamp to track when features were processed
            conn.execute("""
                INSERT OR REPLACE INTO processed_features 
                SELECT *
                FROM processed_df
            """)
            
            # Verify insertion
            count = conn.execute("SELECT COUNT(*) FROM processed_features").fetchone()[0]
            new_count = len(processed_df)
            logger.info(f"Successfully added {new_count} processed feature records. Total records: {count}")
            
            logger.info("Feature engineering completed successfully")
            
            # Return the processed features as asset for downstream consumption
            return processed_df
            
        except Exception as e:
            logger.error(f"Error in ETL processing: {str(e)}")
            raise


@asset(
    group_name=group_name,
    deps=["processed_features"]
)
def feature_statistics_data(processed_features, database: DuckDBResource):
    """
    Calculate and store feature statistics for drift detection and monitoring.
    
    Uses processed features asset directly and computes mean/std statistics
    for numerical features, storing results in feature_statistics table.
    """
    logger.info("Starting feature statistics calculation")
    
    # Use processed features asset directly (already in polars format)
    logger.info("Using processed features asset")
    processed_features_df = processed_features
    
    with database.get_connection() as conn:
        
        logger.info(f"Loaded {len(processed_features_df)} processed feature records")
        
        # Calculate feature statistics using polars
        logger.info("Computing feature statistics")
        
        # Use polars selectors to get numerical columns, excluding ID columns
        import polars.selectors as cs
        numerical_cols = processed_features_df.select(
            (cs.numeric() | cs.boolean()) & ~cs.ends_with("_id")
        ).columns
        
        logger.info(f"Computing statistics for {len(numerical_cols)} numerical features")
        
        # Calculate statistics using polars aggregation
        import datetime
        stats_list = []
        for col in numerical_cols:
            col_stats = processed_features_df.select([
                pl.lit(col).alias("feature_name"),
                pl.col(col).mean().alias("mean_value"),
                pl.col(col).std().alias("std_value"),
                pl.lit(datetime.datetime.now()).alias("computation_timestamp"),
            ])
            stats_list.append(col_stats)
        
        # Combine all statistics into a single dataframe
        if stats_list:
            stats_df = pl.concat(stats_list)
        else:
            # Create empty dataframe with correct schema if no numerical columns
            stats_df = pl.DataFrame({
                "feature_name": [],
                "mean_value": [],
                "std_value": [],
                "computation_timestamp": []
            })
        
        # Store feature statistics in database
        logger.info("Storing feature statistics in database")
        
        # Replace statistics for today with new calculations
        conn.execute("DELETE FROM feature_statistics WHERE computation_timestamp = CURRENT_TIMESTAMP")
        conn.execute("INSERT INTO feature_statistics SELECT * FROM stats_df")
        
    # Return the statistics DataFrame for downstream consumption
    return stats_df