import logging
import sys

from dagster import AssetIn, asset
from dagster import get_dagster_logger


log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

group_name = "predict"


@asset(
    group_name=group_name,
    ins={
        "rf_model": AssetIn(),
        "X_test": AssetIn(),
        "feature_selection": AssetIn()
    }
)
def predictions(rf_model, X_test, feature_selection):

    predictions_df = rf_model.predict_proba(X_test[feature_selection])

    logger.info(f"Generated predictions for {len(predictions_df)} instances.")
    logger.info(f"Predictions sample: {predictions_df[:5]}")

    return predictions_df
