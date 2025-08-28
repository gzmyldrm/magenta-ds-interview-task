import logging
import sys
from sklearn.model_selection import StratifiedGroupKFold

from dagster import get_dagster_logger,AssetIn, AssetOut, multi_asset

# from dagstermill import define_dagstermill_asset

log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

group_name = "data_split"


@multi_asset(
    group_name=group_name,
    ins={
        "processed_and_merged_data": AssetIn()
    },
    outs={
        "X_train": AssetOut(),
        "X_test": AssetOut(),
        "y_train": AssetOut(),
        "y_test": AssetOut(),
    },description="Split the processed and merged data into train and test sets with corresponding targets.",
)
def split_train_test(processed_and_merged_data):

    # Train Test Split
    # recall that customers can have multiple contracts dont put these customers in the test set

    # stratification for keeping positive rate across folds
    # grouping for avoiding leakage (one customer multiple accounts)
    # splits for cross validation
    # random state for reproducibility

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) 

    target = processed_and_merged_data['has_done_upselling']
    X = processed_and_merged_data.drop(columns=['has_done_upselling'])
    groups = processed_and_merged_data['customer_id']

    train_idx, test_idx = next(sgkf.split(X, target, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

    # check distribution
    logger.info(f"Train positives: {y_train.mean():.3f}, Test positives: {y_test.mean():.3f}")


    return X_train, X_test, y_train, y_test