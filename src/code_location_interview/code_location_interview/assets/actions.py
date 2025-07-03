import logging
import sys
import uuid
from datetime import datetime, timedelta

import polars as pl
from dagster import asset, get_dagster_logger
from dagster_duckdb import DuckDBResource

log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

group_name = "actions"


@asset(
    group_name=group_name,
    deps=["predictions", "current_model"],
    description="Generate actions for positive predictions, excluding recently actioned customers"
)
def actions_activities(predictions, current_model, database: DuckDBResource):
    """
    Generate actions for customers with positive predictions.
    
    Takes all positive class predictions and filters out customers/contracts
    that have been actioned in the last X days to avoid spam.
    
    Args:
        predictions: Predictions dataframe from predictions asset
        current_model: Current model info from current_model asset
        database: DuckDB database resource
    
    Returns:
        Actions dataframe with customers selected for upselling offers
    """
    logger.info("Starting actions generation")
    
    # Configuration
    COOLDOWN_DAYS = 30  # Number of days to wait before re-actioning the same customer/contract
    
    # Extract model info
    model_id = current_model["model_id"]
    logger.info(f"Using model ID: {model_id}")
    
    with database.get_connection() as conn:
        
        # Filter for positive predictions only
        positive_predictions = predictions.filter(pl.col("predicted_class") == True)
        logger.info(f"Found {len(positive_predictions)} positive predictions out of {len(predictions)} total predictions")
        
        if len(positive_predictions) == 0:
            logger.info("No positive predictions found, no actions to generate")
            # Return empty actions dataframe with correct schema
            empty_actions = pl.DataFrame({
                "action_id": [],
                "customer_id": [],
                "rating_account_id": [],
                "model_id": [],
                "action_timestamp": [],
                "action_type": [],
                "prediction_id": [],
                "status": []
            })
            return empty_actions
        
        logger.info("Checking for recently actioned customers/contracts")
        
        # Query for recent actions to exclude
        cutoff_date = datetime.now() - timedelta(days=COOLDOWN_DAYS)
        
        recent_actions_query = """
            SELECT DISTINCT 
                customer_id,
                rating_account_id
            FROM actions 
            WHERE action_timestamp >= $1
        """
        
        recent_actions_result = conn.execute(recent_actions_query, [cutoff_date]).fetchall()
        
        if recent_actions_result:
            # Create set of recently actioned customer+contract combinations
            recently_actioned = {(row[0], row[1]) for row in recent_actions_result}
            logger.info(f"Found {len(recently_actioned)} customer/contract combinations actioned in last {COOLDOWN_DAYS} days")
            
            # Filter out recently actioned customers/contracts
            def is_not_recently_actioned(customer_id: str, rating_account_id: str) -> bool:
                return (customer_id, rating_account_id) not in recently_actioned
            
            # Apply filter using polars
            eligible_predictions = positive_predictions.filter(
                pl.struct(["customer_id", "rating_account_id"])
                .map_elements(lambda x: is_not_recently_actioned(x["customer_id"], x["rating_account_id"]))
            )
        else:
            logger.info("No recent actions found, all positive predictions are eligible")
            eligible_predictions = positive_predictions
        
        logger.info(f"After filtering recent actions: {len(eligible_predictions)} eligible predictions")
        
        if len(eligible_predictions) == 0:
            logger.info("No eligible predictions after filtering recent actions")
            # Return empty actions dataframe
            empty_actions = pl.DataFrame({
                "action_id": [],
                "customer_id": [],
                "rating_account_id": [],
                "model_id": [],
                "action_timestamp": [],
                "action_type": [],
                "prediction_id": [],
                "status": []
            })
            return empty_actions
        
        # Create actions for eligible predictions
        logger.info("Creating actions for eligible predictions")
        
        current_timestamp = datetime.now()
        
        actions_data = {
            "action_id": [str(uuid.uuid4()) for _ in range(len(eligible_predictions))],
            "customer_id": eligible_predictions.select("customer_id").to_series().to_list(),
            "rating_account_id": eligible_predictions.select("rating_account_id").to_series().to_list(),
            "model_id": [model_id] * len(eligible_predictions),
            "action_timestamp": [current_timestamp] * len(eligible_predictions),
            "action_type": ["upselling_offer"] * len(eligible_predictions),
            "prediction_id": eligible_predictions.select("prediction_id").to_series().to_list(),
            "status": ["sent"] * len(eligible_predictions)
        }
        
        actions_df = pl.DataFrame(actions_data)
        
        logger.info("Storing actions in database")
        
        # Store actions in database
        conn.execute("INSERT INTO actions SELECT * FROM actions_df")
        
        # Verify insertion
        total_actions = conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
        new_actions = len(eligible_predictions)
        logger.info(f"Successfully stored {new_actions} new actions. Total actions in database: {total_actions}")
        
        # Log summary statistics
        unique_customers = len(eligible_predictions.select("customer_id").unique())
        unique_contracts = len(eligible_predictions.select("rating_account_id").unique())
        
        logger.info(f"Actions summary:")
        logger.info(f"  - Total actions created: {new_actions}")
        logger.info(f"  - Unique customers: {unique_customers}")
        logger.info(f"  - Unique contracts: {unique_contracts}")
        logger.info(f"  - Model used: {model_id}")
        logger.info(f"  - Cooldown period: {COOLDOWN_DAYS} days")
        
        logger.info("Actions generation completed successfully")
        
        return actions_df