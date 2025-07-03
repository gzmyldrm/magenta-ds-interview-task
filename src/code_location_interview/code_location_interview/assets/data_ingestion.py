import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from dagster import AssetOut, asset, get_dagster_logger, multi_asset
from dagster_duckdb import DuckDBResource

# Import ETL and database utilities
sys.path.append(str(Path(__file__).parent.parent.parent))
from code_location_interview.assets.lib.datawarehouse import create

log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

group_name = "data_ingestion"


@asset(
    group_name=group_name,
    deps=["core_data"]
)
def raw_core_data(core_data, database: DuckDBResource):
    """
    Store and return raw core data as asset.
    """
    logger.info("Processing raw core data")
    
    # Store in database for audit/storage purposes
    with database.get_connection() as conn:
        logger.info("Storing raw core data in database")
        conn.execute("""
            INSERT OR REPLACE INTO raw_core_data (
                rating_account_id, customer_id, age, contract_lifetime_days, 
                remaining_binding_days, has_special_offer, is_magenta1_customer, 
                available_gb, gross_mrc, smartphone_brand, has_done_upselling,
                ingestion_timestamp
            )
            SELECT *, CURRENT_TIMESTAMP FROM core_data
        """)
        
        logger.info(f"Stored {len(core_data)} core data records")
    
    # Return the asset for downstream consumption
    return pl.from_pandas(core_data)


@asset(
    group_name=group_name,
    deps=["usage_info"]
)
def raw_usage_info(usage_info, database: DuckDBResource):
    """
    Store and return raw usage info as asset.
    """
    logger.info("Processing raw usage info")
    
    # Store in database for audit/storage purposes
    with database.get_connection() as conn:
        logger.info("Storing raw usage info in database")
        conn.execute("""
            INSERT OR REPLACE INTO raw_usage_info (
                rating_account_id, billed_period_month_d, has_used_roaming, used_gb,
                ingestion_timestamp
            )
            SELECT *, CURRENT_TIMESTAMP FROM usage_info
        """)
        
        logger.info(f"Stored {len(usage_info)} usage info records")
    
    # Return the asset for downstream consumption
    return pl.from_pandas(usage_info)


@asset(
    group_name=group_name,
    deps=["customer_interactions"]
)
def raw_customer_interactions(customer_interactions, database: DuckDBResource):
    """
    Store and return raw customer interactions as asset.
    """
    logger.info("Processing raw customer interactions")
    
    # Store in database for audit/storage purposes
    with database.get_connection() as conn:
        logger.info("Storing raw customer interactions in database")
        conn.execute("""
            INSERT OR REPLACE INTO raw_customer_interactions (
                customer_id, type_subtype, n, days_since_last,
                ingestion_timestamp
            )
            SELECT *, CURRENT_TIMESTAMP FROM customer_interactions
        """)
        
        logger.info(f"Stored {len(customer_interactions)} interaction records")
    
    # Return the asset for downstream consumption
    return pl.from_pandas(customer_interactions)


@asset(
    group_name=group_name,
    deps=["core_data"]
)
def raw_labels(core_data, database: DuckDBResource):
    """
    Store and return labels as asset.
    """
    logger.info("Processing labels")
    
    # Store in database for audit/storage purposes
    with database.get_connection() as conn:
        logger.info("Storing labels in database")
        conn.execute("""
            INSERT OR REPLACE INTO raw_labels (rating_account_id, target_label, created_at)
            SELECT rating_account_id, has_done_upselling, CURRENT_TIMESTAMP 
            FROM core_data
        """)
        
        logger.info(f"Stored {len(core_data)} label records")
    
    # Extract labels for downstream consumption
    labels_df = core_data[['rating_account_id', 'has_done_upselling']].copy()
    labels_df = labels_df.rename(columns={'has_done_upselling': 'target_label'})
    
    return pl.from_pandas(labels_df)