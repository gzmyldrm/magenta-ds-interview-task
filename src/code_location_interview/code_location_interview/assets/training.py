import logging
import pickle
import sys
import uuid
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import optuna
import polars as pl
from dagster import asset, get_dagster_logger
from dagster_duckdb import DuckDBResource
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

group_name = "training"


def _store_feature_importance(conn, model_id, model, logger):
    """
    Store feature importance for a deployed Random Forest model.
    
    Args:
        conn: Database connection
        model_id: Unique identifier for the model
        model: Trained RandomForest model
        logger: Logger instance
    """
    logger.info("Extracting feature importance from Random Forest model")
    
    # Define the feature columns in the same order as used in training
    # These should match the feature_columns from the tune() function
    feature_columns = [
        'age', 'contract_lifetime_days', 'remaining_binding_days', 'has_special_offer',
        'is_magenta1_customer', 'available_gb', 'gross_mrc', 'completion_rate', 'is_bounded',
        'is_huawei', 'is_oneplus', 'is_samsung', 'is_xiaomi', 'is_iphone',
        'n_contracts_per_customer', 'avg_monthly_usage_gb', 'total_usage_gb',
        'max_monthly_usage_gb', 'months_with_roaming', 'ever_used_roaming',
        'active_usage_months', 'months_with_no_delta_1mo_change', 'avg_delta_2mo',
        'delta_2mo_volatility', 'max_delta_2mo_increase', 'max_delta_2mo_decrease',
        'months_with_delta_2mo_increase', 'months_with_no_delta_2mo_change',
        'months_with_delta_3mo_increase', 'months_with_no_delta_3mo_change',
        'last_1_delta_1mo', 'last_2_delta_1mo', 'last_3_delta_1mo',
        'last_1_delta_2mo', 'last_2_delta_2mo', 'last_1_delta_3mo',
        'n_rechnungsanfragen', 'n_produkte_services_tarifdetails', 'n_prolongation',
        'n_produkte_services_tarifwechsel', 'days_since_last_rechnungsanfragen',
        'days_since_last_produkte_services_tarifdetails', 'days_since_last_prolongation',
        'days_since_last_produkte_services_tarifwechsel', 'times_in_p1', 'times_in_p2',
        'times_in_p3', 'times_in_p4', 'times_in_p5'
    ]
    
    try:
        # Get feature importance from Random Forest model
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            logger.info("Successfully extracted feature importances from Random Forest model")
        else:
            raise ValueError(f"Model {type(model)} does not have feature_importances_ attribute")
        
        # Verify that we have the right number of feature importances
        if len(feature_importances) != len(feature_columns):
            logger.warning(f"Mismatch between expected features ({len(feature_columns)}) and "
                          f"actual feature importances ({len(feature_importances)})")
            # Adjust feature_columns to match actual importance length
            if len(feature_importances) < len(feature_columns):
                feature_columns = feature_columns[:len(feature_importances)]
            else:
                # Pad with None if we have more importances than expected features
                feature_columns.extend([f"unknown_feature_{i}" for i in range(len(feature_columns), len(feature_importances))])
        
        logger.info("Preparing feature importance data for storage")
        
        # Create importance data structure
        importance_data = {
            "model_id": model_id,
        }
        
        # Add individual feature importance columns
        for i, feature_name in enumerate(feature_columns):
            importance_column_name = f"{feature_name}_importance"
            importance_data[importance_column_name] = feature_importances[i]
        
        # Create Polars DataFrame with feature importance
        importance_df = pl.DataFrame([importance_data])
        
        logger.info("Storing feature importance in database")
        
        # Store feature importance in database
        conn.execute("INSERT INTO features_importance SELECT * FROM importance_df")
        
        # Log top 5 most important features
        feature_importance_pairs = list(zip(feature_columns, feature_importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        logger.info("Top 5 most important features:")
        for i, (feature, importance) in enumerate(feature_importance_pairs[:5]):
            logger.info(f"  {i+1}. {feature}: {importance:.4f}")
        
        logger.info("Feature importance storage completed successfully")
        
    except Exception as e:
        logger.error(f"Error storing feature importance: {str(e)}")
        logger.error(f"Model type: {type(model)}")
        if hasattr(model, '__dict__'):
            logger.error(f"Model attributes: {list(model.__dict__.keys())}")
        raise


def random_forest_objective(trial, X, y, skf, n_splits=5):
    """
    Random Forest objective function using stratified cross-validation
    Adapted from notebook 3a.tuning-models.ipynb
    
    Args:
        trial: Optuna trial object
        X: Feature matrix
        y: Target vector
        skf: Stratified K-Fold cross-validator
        n_splits: Number of folds for cross-validation (default: 5)
    """
    cv_scores = []

    # Random Forest hyperparameters
    param = {
        'n_jobs':-1,
        'random_state': 42,
        'verbose': 0,

        # Core tree parameters
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),

        # Feature sampling parameters
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'max_samples': trial.suggest_float('max_samples', 0.1, 1.0),

        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 1000),

        # Class balancing
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
    }

    rf = RandomForestClassifier(**param)

    for train_idx, valid_idx in skf.split(X, y):
        train_x = X[train_idx]
        valid_x = X[valid_idx]
        train_y = y[train_idx]
        valid_y = y[valid_idx]

        rf.fit(train_x, train_y)
        preds = rf.predict(valid_x)
        f1 = f1_score(valid_y, preds)
        cv_scores.append(f1)

    return np.mean(cv_scores)


@asset(
    group_name=group_name,
    description="Tune Random Forest model using Optuna optimization"
)
def tune(database: DuckDBResource):
    """
    Tune Random Forest model using hyperparameter optimization.
    
    Reads features and labels from database, splits into train/test,
    performs hyperparameter tuning, and returns the best model with test data.
    
    Returns:
        dict: Contains best_model, test_features, test_labels, best_params, cv_score
    """
    logger.info("Starting model tuning process")
    
    with database.get_connection() as conn:
        
        # Read processed features with labels
        logger.info("Loading processed features and labels from database")
        
        # Query for the most recent features per rating_account_id
        features_query = """
            WITH ranked_features AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY rating_account_id ORDER BY created_at DESC) as rn
                FROM processed_features
            )
            SELECT * FROM ranked_features WHERE rn = 1
        """
        
        features_result = conn.execute(features_query).fetchall()
        
        if not features_result:
            logger.error("No processed features found in database")
            raise ValueError("No processed features found in database")
        
        # Convert to polars DataFrame
        column_names = [desc[0] for desc in conn.description]
        features_df = pl.DataFrame(features_result, schema=column_names).drop('rn')
        
        logger.info(f"Loaded {len(features_df)} feature records")
        
        # Read labels from raw data (has_done_upselling is the target)
        labels_query = """
            SELECT rating_account_id, has_done_upselling
            FROM processed_features
            WHERE rating_account_id IN (
                SELECT rating_account_id FROM ({}) t
            )
        """.format(features_query.replace('SELECT *', 'SELECT rating_account_id'))
        
        # Join features with labels to create a labeled dataset
        labels_result = conn.execute(labels_query).fetchall()
        if not labels_result:
            logger.error("No labels found in database")
            raise ValueError("No labels found in database")
        labels_column_names = [desc[0] for desc in conn.description]
        labels_df = pl.DataFrame(labels_result, schema=labels_column_names)

        # This would be the ideal code but it is not needed
        # Join on rating_account_id to ensure alignment
        # features_df = features_df.drop('has_done_upselling').join(
        #     labels_df, on="rating_account_id", how="inner"
        # ).drop('rn')
        target_column = 'has_done_upselling'
        
        if target_column not in features_df.columns:
            logger.error(f"Target column '{target_column}' not found in features")
            raise ValueError(f"Target column '{target_column}' not found in features")
        
        # Prepare features and target
        feature_columns = [col for col in features_df.columns 
                          if col not in ['rating_account_id', 'customer_id', 'has_done_upselling', 'created_at']]
        
        logger.info(f"All columns in features_df: {features_df.columns}")
        logger.info(f"Excluded columns: ['rating_account_id', 'customer_id', 'has_done_upselling', 'created_at']")
        logger.info(f"Feature columns ({len(feature_columns)}): {feature_columns}")
        logger.info(f"Using {len(feature_columns)} features for training")
        
        X = features_df.select(feature_columns).to_numpy()
        y = features_df.select(target_column).to_numpy().ravel()
        
        logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
        logger.info(f"Target distribution: {np.bincount(y.astype(int))}")
        
        # Ensure we have enough data for cross-validation
        if len(y) < 10:
            logger.error(f"Insufficient data for training: only {len(y)} samples")
            raise ValueError(f"Insufficient data for training: only {len(y)} samples. Need at least 10 samples.")
        
        # Check target class distribution
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.error(f"Target has only one class: {unique_classes}")
            raise ValueError(f"Target has only one class: {unique_classes}. Need binary classification.")
        
        # Split into train and test sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Set up cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Create artifacts directory for Optuna studies
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        study_db_path = f"sqlite:///{artifacts_dir}/rf_study.db"
        
        logger.info("Starting hyperparameter tuning with Optuna")
        
        # Create and run Optuna study
        study = optuna.create_study(
            study_name=f"random_forest_optimization_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction="maximize",
            storage=study_db_path,
            load_if_exists=True
        )
        
        # Number of trials for tuning
        n_trials = 2  # Reduced from notebook for faster retraining
        
        study.optimize(
            lambda trial: random_forest_objective(trial, X_train, y_train, skf), 
            n_trials=n_trials,
            show_progress_bar=False
        )
        
        logger.info(f"Hyperparameter tuning completed with {len(study.trials)} trials")
        logger.info(f"Best CV score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        
        best_params = study.best_params.copy()
        best_params.update({
            'n_jobs': -1,
            'random_state': 42,
            'verbose': 0
        })
        
        best_model = RandomForestClassifier(**best_params)
        best_model.fit(X_train, y_train)
        
        logger.info("Model training completed successfully")
        
        # Prepare return data
        result = {
            'best_model': best_model,
            'test_features': X_test,
            'test_labels': y_test,
            'best_params': best_params,
            'cv_score': study.best_value,
        }
        
        logger.info(f"Tuning completed - CV F1 Score: {study.best_value:.4f}")
        
        return result


@asset(
    group_name=group_name,
    deps=["tune"],
    description="Validate new model against current model and decide if update is needed"
)
def validate(tune, database: DuckDBResource):
    """
    Compare the newly tuned model with the current deployed model.
    
    Tests both models on the same test set and compares F1 scores.
    
    Args:
        tune_result: Result from tune asset containing new model and test data
        current_model_info: Current model information from current_model asset
        
    Returns:
        dict: Validation results with recommendation for model registry update
    """
    logger.info("Starting model validation process")
    
    # Extract data from tune result
    new_model = tune['best_model']
    X_test = tune['test_features']
    y_test = tune['test_labels']
    new_model_params = tune['best_params']
    cv_score = tune['cv_score']
    
    # Extract current model
    # Retrieve current model from the model registry database
    with database.get_connection() as conn:
        result = conn.execute("""
            SELECT model_id, model_artifact
            FROM model_registry
            WHERE is_current_model = TRUE
            ORDER BY created_at DESC
            LIMIT 1
        """).fetchone()
        if result is None:
            logger.warning("No current model found in model registry, assuming no deployed model")
            current_model = None
            current_model_id = None
        else:
            current_model_id = result[0]
            model_bytes = result[1]
            current_model = pickle.loads(model_bytes)
    
    logger.info("Evaluating new model on test set")
    
    # Evaluate new model
    new_predictions = new_model.predict(X_test)    
    new_metrics = {
        'f1_score': f1_score(y_test, new_predictions),
        'accuracy': accuracy_score(y_test, new_predictions),
        'precision': precision_score(y_test, new_predictions),
        'recall': recall_score(y_test, new_predictions),
    }
    
    logger.info(f"New model metrics: F1={new_metrics['f1_score']:.4f}, "
                f"Accuracy={new_metrics['accuracy']:.4f}, "
                f"Precision={new_metrics['precision']:.4f}, "
                f"Recall={new_metrics['recall']:.4f}")
    
    logger.info("Evaluating current model on test set")
    
    # Evaluate current model on same test set
    current_predictions = current_model.predict(X_test)
    
    current_metrics = {
        'f1_score': f1_score(y_test, current_predictions),
        'accuracy': accuracy_score(y_test, current_predictions),
        'precision': precision_score(y_test, current_predictions),
        'recall': recall_score(y_test, current_predictions),
    }
    
    logger.info(f"Current model metrics: F1={current_metrics['f1_score']:.4f}, "
                f"Accuracy={current_metrics['accuracy']:.4f}, "
                f"Precision={current_metrics['precision']:.4f}, "
                f"Recall={current_metrics['recall']:.4f}")
    
    # Compare models using F1 score as primary metric
    f1_improvement = new_metrics['f1_score'] - current_metrics['f1_score']
    should_update = new_metrics['f1_score'] > current_metrics['f1_score']
    
    # Prepare validation results
    validation_result = {
        'should_update_model': should_update,
        'new_model_metrics': new_metrics,
        'new_model': new_model,
        'new_model_params': new_model_params,
        'current_model_id': current_model_id,
    }
    
    if should_update:
        logger.info(f"New model performs better! F1 improvement: {f1_improvement:.4f}")
        logger.info("Recommending model registry update")
    else:
        logger.info(f"Current model performs better. F1 difference: {f1_improvement:.4f}")
        logger.info("No model update recommended")
    
    return validation_result


@asset(
    group_name=group_name,
    deps=["validate"],
    description="Update model registry based on validation results"
)
def model_registry_update(validate, database: DuckDBResource):
    """
    Update the model registry based on validation results.
    
    Always stores the new model in the registry as 'trained'.
    If the new model performs better, also deploys it as current model.
    
    Args:
        validation_result: Results from the validate asset
        
    Returns:
        dict: Summary of the registry update operation
    """
    logger.info("Starting model registry update process")
    
    should_update = validate['should_update_model']
    
    logger.info("Storing new model in registry")
    
    with database.get_connection() as conn:
        
        # Generate new model metadata
        new_model_id = str(uuid.uuid4())
        model_name = "RandomForest-General"
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Serialize the new model
        model_bytes = pickle.dumps(validate['new_model'])
        
        # Create model metadata
        metadata = {
            'model_type': 'RandomForestClassifier',
            'hyperparameters': validate['new_model_params'],
        }
        
        # Determine model status and current flag
        model_status = 'deployed' if should_update else 'trained'
        is_current = should_update
        
        try:
            # Start transaction
            conn.begin()
            
            old_model_id = validate['current_model_id']
            
            # If deploying new model, mark current model as not current
            if should_update and old_model_id:
                conn.execute("""
                    UPDATE model_registry 
                    SET is_current_model = FALSE, status = 'retired'
                    WHERE model_id = ?
                """, [old_model_id])
                
                logger.info(f"Marked old model {old_model_id} as retired")
            
            # Always insert the new model
            conn.execute("""
                INSERT INTO model_registry (
                    model_id, model_name, version, model_artifact, metadata,
                    status, is_current_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                new_model_id,
                model_name,
                version,
                model_bytes,
                json.dumps(metadata),
                model_status,  # status: 'deployed' or 'trained'
                is_current  # is_current_model: True if deploying, False if just storing
            ])
            
            # If deploying the model, compute and store feature importance
            if should_update:
                logger.info("Computing feature importance for deployed model")
                _store_feature_importance(conn, new_model_id, validate['new_model'], logger)
            
            # Commit transaction
            conn.commit()
            
            if should_update:
                logger.info(f"Successfully deployed new model {new_model_id}")
                logger.info(f"   Model: {model_name} {version}")
                logger.info(f"   F1 Score: {validate['new_model_metrics']['f1_score']:.4f}")
                logger.info(f"   Status: deployed (current model)")
            else:
                logger.info(f"Successfully stored new model {new_model_id}")
                logger.info(f"   Model: {model_name} {version}")
                logger.info(f"   F1 Score: {validate['new_model_metrics']['f1_score']:.4f}")
                logger.info(f"   Status: trained (not deployed - current model performs better)")
            
            # Verify the insertion
            verification = conn.execute("""
                SELECT model_id, model_name, version, status, is_current_model
                FROM model_registry 
                WHERE model_id = ?
            """, [new_model_id]).fetchone()
            
            if verification:
                logger.info(f"Model registry update verified: {verification}")
            else:
                logger.error("Model registry update verification failed")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating model registry: {e}")
            raise
    
    return {
        'stored': True,
        'deployed': should_update,
        'new_model_id': new_model_id,
        'model_name': model_name,
        'version': version,
        'old_model_id': old_model_id,
        'new_f1_score': validate['new_model_metrics']['f1_score'],
        'model_status': model_status,
        'is_current_model': is_current,
        'update_timestamp': datetime.now()
    }