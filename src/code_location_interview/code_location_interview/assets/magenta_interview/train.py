import logging
import sys
from xgboost import XGBClassifier
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from dagster import asset, get_dagster_logger, AssetOut,AssetIn, multi_asset

# from dagstermill import define_dagstermill_asset

log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

group_name = "training"

@asset(group_name=group_name)
def feature_selection():
    # manual feature selection for now
    # TODO: move feature selection as seperate process and systematic
    selected_features = ['remaining_binding_days_bucket_0-30','remaining_binding_days_bucket_31-90',
                        'remaining_binding_days_bucket_91-180','remaining_binding_days_bucket_>180',
                        'mrc_bucket_Mid (30-50€)','mrc_bucket_High (>50€)',
                        'contract_term_bucket_1-2y','contract_term_bucket_2-3y','contract_term_bucket_3-5y',
                        'age_bucket_25-45','age_bucket_46-60','age_bucket_>60',
                        'available_gb','available_gb_missing','has_special_offer','is_magenta1_customer','is_premium_device']
    return selected_features

@asset(group_name=group_name,ins={"X_train": AssetIn(),"y_train": AssetIn(),"feature_selection": AssetIn()},description="Train Random Forest Classifier.")
def rf_model(X_train, y_train, feature_selection):
    # Simple setup
    # TODO: add dedicated hyperparameter tuning
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=50,
        class_weight="balanced",
        max_features="sqrt",
        oob_score=True,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train[feature_selection], y_train)

     ##############################

    pos_rate = float(y_train.mean())
    logger.info(f"Train positives rate: {pos_rate:.4f}  (n={len(y_train)})")

    oob_info = {}
    if hasattr(model, "oob_score_") and model.oob_score_ is not None:
        # oob_score_ is accuracy; keep it and the error
        oob_info = {
            "oob_accuracy": float(model.oob_score_),
            "oob_error": float(1.0 - model.oob_score_),
            "train_pos_rate": pos_rate,
            "n_train": int(len(y_train)),
        }

    # --- Feature importances  ---
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        imp_df = pd.DataFrame(columns=["feature", "importance"])
    else:
        imp_df = (
            pd.DataFrame({"feature": X_train[feature_selection].columns, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    logger.info("Feature importances:\n%s", imp_df.head(10).to_string(index=False))
    logger.info("OOB info: %s", oob_info)

    return model


# # Preferable experiment with 2-3 models
# # logistic regression and xgboost could be other candidates
# # For now as placeholders
# @asset(group_name="training", ins={"X_train": AssetIn(), "y_train": AssetIn(), "feature_selection": AssetIn()}, description="Train Logistic Regression baseline.")
# def lr_model(X_train, y_train, feature_selection):
#     model = LogisticRegression(max_iter=1000, class_weight="balanced")
#     model.fit(X_train[feature_selection], y_train)
#     return model


# @asset(group_name="training", ins={"X_train": AssetIn(), "y_train": AssetIn(), "feature_selection": AssetIn()}, description="Train XGBoost classifier.")
# def xgb_model(X_train, y_train, feature_selection):
   
#     model = XGBClassifier(
#         random_state=42,
#         n_estimators=300,
#         max_depth=4,
#         learning_rate=0.05,
#         subsample=0.8)
#     model.fit(X_train[feature_selection], y_train)
#     return model    