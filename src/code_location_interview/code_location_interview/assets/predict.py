import logging
import pickle
import sys
import uuid
from datetime import datetime

import polars as pl
from dagster import asset, get_dagster_logger
from dagster_duckdb import DuckDBResource

log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

group_name = "predict"


@asset(
    group_name=group_name,
)
def current_model(database: DuckDBResource):
    """
    Load the current deployed model from the model registry.
    
    Queries the model_registry table to find the model marked as current
    and returns the deserialized model object.
    """
    logger.info("Loading current deployed model from model registry")
    
    with database.get_connection() as conn:
        
        # Query for the current deployed model
        current_model_query = """
            SELECT 
                model_id,
                model_name,
                version,
                model_artifact,
                metadata,
                created_at
            FROM model_registry
            WHERE is_current_model = TRUE
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        result = conn.execute(current_model_query).fetchone()
        
        if not result:
            logger.error("No current model found in model registry")
            raise ValueError("No current model found in model registry. Please ensure a model is deployed.")
        
        model_id, model_name, version, model_artifact, metadata, created_at = result
        
        logger.info(f"Found current model: {model_name} v{version} (ID: {model_id})")
        
        # Deserialize the model from BLOB
        if not model_artifact:
            logger.error(f"Model artifact is empty for model {model_id}")
            raise ValueError(f"Model artifact is empty for model {model_id}")
        
        try:
            # Deserialize the pickled model
            model = pickle.loads(model_artifact)
            logger.info(f"Successfully loaded model {model_name} v{version}")
            
            # Return model with metadata
            return {
                "model": model,
                "model_id": model_id,
            }
            
        except Exception as e:
            logger.error(f"Failed to deserialize model {model_id}: {str(e)}")
            raise ValueError(f"Failed to deserialize model {model_id}: {str(e)}")


@asset(
    group_name=group_name,
    deps=["processed_features", "current_model"],
)
def predictions(processed_features, current_model, database: DuckDBResource):
    """
    Generate predictions using the current model on processed features.
    
    Takes processed features and the current model, generates predictions,
    and stores results in the predictions table.
    """
    logger.info("Starting prediction generation")
    
    # Extract model and metadata
    model = current_model["model"]
    model_id = current_model["model_id"]
    
    logger.info(f"Using model: ID: {model_id}")
    
    with database.get_connection() as conn:
        
        # Prepare features for prediction
        logger.info(f"Processing {len(processed_features)} feature records for prediction")
        
        # Get feature columns for prediction (exclude ID columns and target)
        feature_columns = [col for col in processed_features.columns 
                          if col not in ['rating_account_id', 'customer_id', 'has_done_upselling', 'created_at']]
        
        logger.info(f"Using {len(feature_columns)} features for prediction")
        
        # Extract features for prediction
        X = processed_features.select(feature_columns).to_numpy()
        
        logger.info("Generating predictions with model")
        
        try:
            # Generate prediction probabilities
            prediction_probabilities = model.predict_proba(X)
            
            # Extract probability for positive class (class 1)
            positive_class_probs = prediction_probabilities[:, 1]
            
            # Apply threshold of 0.5 for binary classification
            predicted_classes = (positive_class_probs >= 0.5).astype(bool)
            
            logger.info(f"Generated {len(predicted_classes)} predictions")
            logger.info(f"Positive predictions: {sum(predicted_classes)} ({sum(predicted_classes)/len(predicted_classes)*100:.3f}%)")
            
            # Create predictions dataframe
            predictions_data = {
                "prediction_id": [str(uuid.uuid4()) for _ in range(len(predicted_classes))],
                "rating_account_id": processed_features.select("rating_account_id").to_series().to_list(),
                "customer_id": processed_features.select("customer_id").to_series().to_list(),
                "prediction_score": positive_class_probs.tolist(),
                "predicted_class": predicted_classes.tolist(),
                "model_id": [model_id] * len(predicted_classes),
                "features_created_at": processed_features.select("created_at").to_series().to_list(),
                "prediction_timestamp": [datetime.now()] * len(predicted_classes)
            }
            
            predictions_df = pl.DataFrame(predictions_data)
            
            logger.info("Storing predictions in database")
            
            # Store predictions in database
            conn.execute("INSERT INTO predictions SELECT * FROM predictions_df")
            
            # Verify insertion
            total_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            new_predictions = len(predicted_classes)
            logger.info(f"Successfully stored {new_predictions} predictions. Total predictions in database: {total_predictions}")
            
            logger.info("Prediction generation completed successfully")
            
            # Return predictions dataframe for downstream consumption
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error during prediction generation: {str(e)}")
            raise
