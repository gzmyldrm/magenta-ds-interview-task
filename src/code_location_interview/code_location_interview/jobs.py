"""
Dagster Jobs for ML Pipeline
"""

import dagster as dg

# Data Pipeline Job - handles data ingestion and feature engineering
data_pipeline_job = dg.define_asset_job(
    name="data_pipeline_job",
    description="Ingests raw data and processes features",
    selection=dg.AssetSelection.groups("get_data", "data_ingestion", "feature_engineering")
)

# Action Pipeline Job - handles predictions and customer actions
action_pipeline_job = dg.define_asset_job(
    name="action_pipeline_job", 
    description="Generates predictions and creates customer actions",
    selection=dg.AssetSelection.groups("predict", "actions")
)

# Training Pipeline Job - handles model retraining
training_pipeline_job = dg.define_asset_job(
    name="training_pipeline_job",
    description="Retrains and validates ML models",
    selection=dg.AssetSelection.groups("training")
)

# Master Pipeline - runs data first, then actions (dependencies handle the sequence)
master_pipeline_job = dg.define_asset_job(
    name="master_pipeline_job",
    description="Runs data pipeline followed by actions pipeline",
    selection=dg.AssetSelection.groups("get_data", "data_ingestion", "feature_engineering", "predict", "actions")
)