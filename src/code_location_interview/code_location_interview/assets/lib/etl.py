"""
ETL Pipeline for Customer Upselling Feature Engineering

This module processes raw customer data (core_data, customer_interactions, usage_info)
and transforms it into cleaned features ready for ML model prediction.

Based on notebook 1a.features-engineering.ipynb
"""

import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional

import polars as pl
import numpy as np

from sklearn.linear_model import Lasso

from .constants import ARTIFACTS_PATH

class ContractFeatureETL:
    """
    Production ETL pipeline for customer upselling feature engineering.
    
    Processes core customer data, usage information, and interaction history
    to create ML-ready features with proper imputation and cleaning.
    """
    
    def __init__(self, artifacts_dir: Optional[Path] = None):
        """
        Initialize ETL pipeline.
        
        Args:
            artifacts_dir: Path to artifacts directory containing imputer model.
                          If None, uses default path structure.
        """
        if artifacts_dir is None:
            # Default path assuming script is in lib/ and artifacts in notebooks/data/models/artifacts/
            artifacts_dir = ARTIFACTS_PATH

        self.artifacts_dir = Path(artifacts_dir)
        self.imputer_model: Lasso
        self.available_values_imputer = [0, 10, 20, 30, 40, 50]
        self.imputation_features = ['last_1_month_usage_gb', 'last_2_month_usage_gb', 'last_3_month_usage_gb', 'last_4_month_usage_gb']
        
        self.final_columns = [
            'rating_account_id',
            'customer_id',
            'age',
            'contract_lifetime_days',
            'remaining_binding_days',
            'has_special_offer',
            'is_magenta1_customer',
            'available_gb',
            'gross_mrc',
            'has_done_upselling',
            'completion_rate',
            'is_bounded',
            'is_huawei',
            'is_oneplus',
            'is_samsung',
            'is_xiaomi',
            'is_iphone',
            'n_contracts_per_customer',
            'avg_monthly_usage_gb',
            'total_usage_gb',
            'max_monthly_usage_gb',
            'months_with_roaming',
            'ever_used_roaming',
            'active_usage_months',
            'months_with_no_delta_1mo_change',
            'avg_delta_2mo',
            'delta_2mo_volatility',
            'max_delta_2mo_increase',
            'max_delta_2mo_decrease',
            'months_with_delta_2mo_increase',
            'months_with_no_delta_2mo_change',
            'months_with_delta_3mo_increase',
            'months_with_no_delta_3mo_change',
            'last_1_delta_1mo',
            'last_2_delta_1mo',
            'last_3_delta_1mo',
            'last_1_delta_2mo',
            'last_2_delta_2mo',
            'last_1_delta_3mo',
            'n_rechnungsanfragen',
            'n_produkte&services-tarifdetails',
            'n_prolongation',
            'n_produkte&services-tarifwechsel',
            'days_since_last_rechnungsanfragen',
            'days_since_last_produkte&services-tarifdetails',
            'days_since_last_prolongation',
            'days_since_last_produkte&services-tarifwechsel',
            'times_in_p1',
            'times_in_p2',
            'times_in_p3',
            'times_in_p4',
            'times_in_p5'
        ]

        self._load_imputer()
    
    def _load_imputer(self) -> None:
        """Load the pre-trained imputer model from artifacts."""
        imputer_path = self.artifacts_dir / 'imputer_model.pkl'
        
        if not imputer_path.exists():
            raise FileNotFoundError(f"Imputer model not found at {imputer_path}")
        
        with open(imputer_path, 'rb') as f:
            self.imputer_model = pickle.load(f)
    
    def _find_closest(self, val: float, avail_list: List[int]) -> Optional[int]:
        """Find closest value from available_values list."""
        if val is None or np.isnan(val):
            return None
        return min(avail_list, key=lambda x: abs(x - val))
    
    def process_core_data(self, core_data: pl.DataFrame) -> pl.DataFrame:
        """
        Process core customer data with feature engineering.
        
        Args:
            core_data: Raw core customer data
            
        Returns:
            Processed core data with engineered features
        """
        # Cast data types
        core_data = core_data.with_columns([
            pl.col('rating_account_id').cast(pl.Utf8),
            pl.col("has_done_upselling").cast(pl.Boolean),
            pl.col("has_special_offer").cast(pl.Boolean),
            pl.col("is_magenta1_customer").cast(pl.Boolean)
        ])
        
        # Create binding and completion features
        core_data = core_data.with_columns([
            (pl.col('contract_lifetime_days') / (pl.col('contract_lifetime_days') + pl.col('remaining_binding_days'))).round(2).alias('completion_rate'),
            pl.when(pl.col('remaining_binding_days') > 0)
                .then(True)
                .otherwise(False)
                .alias('is_bounded')
        ])
        
        # One-hot-encode smartphone brands
        smartphone_brands_list = core_data.select(pl.col('smartphone_brand')).unique().to_series().sort().to_list()
        core_data = core_data.with_columns([
            pl.when(pl.col("smartphone_brand") == brand)
            .then(True)
            .otherwise(False)
            .alias(f"is_{brand.lower()}")
            for brand in smartphone_brands_list
        ]).drop("smartphone_brand")
        
        # Add contract count per customer
        n_contract_per_customer = core_data.group_by("customer_id").agg(
            pl.col("rating_account_id").count().alias("n_contracts_per_customer")
        )
        core_data = core_data.join(n_contract_per_customer, on="customer_id", how="left")
        
        return core_data
    
    def process_usage_info(self, usage_info: pl.DataFrame) -> pl.DataFrame:
        """
        Process usage information with comprehensive feature engineering.
        
        Args:
            usage_info: Raw usage information data
            
        Returns:
            Processed usage features
        """
        # Cast data types and sort
        usage_info = usage_info.with_columns([
            pl.col('rating_account_id').cast(pl.Utf8),
            pl.col('billed_period_month_d').cast(pl.Date),
            pl.col('has_used_roaming').cast(pl.Boolean),
            pl.col('used_gb').cast(pl.Float64)
        ]).sort(['rating_account_id', 'billed_period_month_d'])
        
        # Monthly usage features
        month_usage = usage_info.group_by('rating_account_id').agg([
            pl.col('used_gb')
        ]).with_columns([  # Used by the imputer but to be removed
            pl.col('used_gb').list.get(0).alias('last_1_month_usage_gb'),
            pl.col('used_gb').list.get(1).alias('last_2_month_usage_gb'),
            pl.col('used_gb').list.get(2).alias('last_3_month_usage_gb'),
            pl.col('used_gb').list.get(3).alias('last_4_month_usage_gb'),
        ]).drop('used_gb')
        

        # Aggregated usage statistics
        aggregated_features = usage_info.group_by('rating_account_id').agg([
            
            # Basic usage statistics
            pl.col('used_gb').mean().round(2).alias('avg_monthly_usage_gb'),
            pl.col('used_gb').sum().round(2).alias('total_usage_gb'),
            pl.col('used_gb').max().round(2).alias('max_monthly_usage_gb'),
            
            # Roaming statistics
            pl.col('has_used_roaming').sum().alias('months_with_roaming'),
            pl.col('has_used_roaming').any().alias('ever_used_roaming'),
            
            # Usage intensity categories
            (pl.col('used_gb') > 0).sum().alias('active_usage_months'),
        ])
        
        # Trend and delta features
        trend_features = usage_info.group_by('rating_account_id').agg([
            
            # Rolling averages
            pl.col('used_gb').rolling_mean_by(
                'billed_period_month_d', window_size='2mo'
            ).alias('avg_2month_rolling_usage_gb'),
            
            pl.col('used_gb').rolling_mean_by(
                'billed_period_month_d', window_size='3mo'
            ).alias('avg_3month_rolling_usage_gb'),
            
            # Period-over-period deltas
            (pl.col('used_gb') - pl.col('used_gb').shift(1)).alias('delta_1mo'),
            (pl.col('used_gb') - pl.col('used_gb').shift(2)).alias('delta_2mo'),
            (pl.col('used_gb') - pl.col('used_gb').shift(3)).alias('delta_3mo'),
            
            # Rolling standard deviation
            pl.col('used_gb').rolling_std_by(
                'billed_period_month_d', window_size='2mo'
            ).alias('std_2month_rolling_usage_gb')
        ])
        
        # Process trend features
        trend_features = trend_features.with_columns([
            # Delta statistics
            pl.col('delta_1mo').list.std().round(2).alias('delta_1mo_volatility'),
            pl.col('delta_1mo').list.eval(pl.element() == 0).list.sum().alias('months_with_no_delta_1mo_change'),
            
            pl.col('delta_2mo').list.mean().round(2).alias('avg_delta_2mo'),
            pl.col('delta_2mo').list.std().round(2).alias('delta_2mo_volatility'),
            pl.col('delta_2mo').list.max().round(2).alias('max_delta_2mo_increase'),
            pl.col('delta_2mo').list.min().round(2).alias('max_delta_2mo_decrease'),
            pl.col('delta_2mo').list.eval(pl.element() > 0).list.sum().alias('months_with_delta_2mo_increase'),
            pl.col('delta_2mo').list.eval(pl.element() == 0).list.sum().alias('months_with_no_delta_2mo_change'),
            
            pl.col('delta_3mo').list.eval(pl.element() > 0).list.sum().alias('months_with_delta_3mo_increase'),
            pl.col('delta_3mo').list.eval(pl.element() == 0).list.sum().alias('months_with_no_delta_3mo_change'),
        ])
        
        # Extract rolling and delta values
        trend_features = trend_features.drop('avg_2month_rolling_usage_gb')
        
        trend_features = trend_features.drop('avg_3month_rolling_usage_gb')
        
        trend_features = trend_features.with_columns([
            pl.col('delta_1mo').list.get(-(i+1)).round(2).alias(f'last_{i+1}_delta_1mo')
            for i in range(3)
        ]).drop('delta_1mo')
        
        trend_features = trend_features.with_columns([
            pl.col('delta_2mo').list.get(-(i+1)).round(2).alias(f'last_{i+1}_delta_2mo')
            for i in range(2)
        ]).drop('delta_2mo')
        
        trend_features = trend_features.with_columns([
            pl.col('delta_3mo').list.get(-(i+1)).round(2).alias(f'last_{i+1}_delta_3mo')
            for i in range(1)
        ]).drop('delta_3mo')
        
        trend_features = trend_features.drop('std_2month_rolling_usage_gb')
        
        # Combine all usage features
        usage_features = aggregated_features.join(
            trend_features, on='rating_account_id', how='left'
        ).join(
            month_usage, on='rating_account_id', how='left'
        )
        
        return usage_features
    
    def process_customer_interactions(self, customer_interactions: pl.DataFrame) -> pl.DataFrame:
        """
        Process customer interaction data.
        
        Args:
            customer_interactions: Raw customer interaction data
            
        Returns:
            Processed interaction features
        """
        interactions_features = customer_interactions.pivot(
            index='customer_id',
            on='type_subtype', 
            values=['n', 'days_since_last'],
            aggregate_function='first'
        )
        
        return interactions_features
    
    def impute_available_gb(self, features: pl.DataFrame) -> pl.DataFrame:
        """
        Impute missing available_gb values using pre-trained Lasso model.
        
        Args:
            features: Features dataframe with potential missing available_gb
            
        Returns:
            Features with imputed available_gb values
        """
        if self.imputer_model is None:
            raise ValueError("Imputer not loaded. Call _load_imputer() first.")
        
        # Check if there are any missing values
        missing_mask = features.select(pl.col('available_gb').is_null()).to_series()
        if not missing_mask.any():
            return features  # No missing values
        
        # Separate rows with and without missing values
        df_complete = features.filter(pl.col('available_gb').is_not_null())
        df_missing = features.filter(pl.col('available_gb').is_null()).drop('available_gb')
        
        # Get imputation features
        lasso_model = self.imputer_model
        
        # Predict missing values
        X_missing = df_missing.select(self.imputation_features).to_numpy()
        predicted_values = lasso_model.predict(X_missing)

        # Add predicted values as a new column
        df_missing = df_missing.with_columns([
            pl.Series('predicted', predicted_values)
        ])
        
        df_missing = df_missing.with_columns([
            # Find closest available value
            pl.col('predicted').map_elements(
                lambda x: self._find_closest(x, self.available_values_imputer), 
                return_dtype=pl.Int64
            ).alias('available_gb')
        ]).drop('predicted')
                
        # Combine complete and imputed data
        features_imputed = pl.concat([df_complete, df_missing], how='diagonal')
        
        return features_imputed
    
    def create_usage_percentile_features(self, features: pl.DataFrame) -> pl.DataFrame:
        """
        Create usage percentile features based on available_gb thresholds.
        
        Args:
            features: Features with available_gb column
            
        Returns:
            Features with added percentile usage features
        """
        # Compute thresholds per account
        thresholds_available_gb = features.group_by('rating_account_id').agg([
            (pl.col('available_gb') / 100 * 25).get(0).round(2).alias('p25'),
            (pl.col('available_gb') / 100 * 50).get(0).round(2).alias('p50'),
            (pl.col('available_gb') / 100 * 70).get(0).round(2).alias('p75'),
        ])
        
        # Join thresholds and create percentile expressions
        features = features.join(thresholds_available_gb, on='rating_account_id', how='left')
        
        percentile_exprs = []
        for i in range(1, 5):
            percentile_expr = (
                pl.when(pl.col(f'last_{i}_month_usage_gb').is_between(-1, pl.col('p25'), closed='right'))
                .then(pl.lit('P1'))
                .when(pl.col(f'last_{i}_month_usage_gb').is_between(pl.col('p25'), pl.col('p50'), closed='right'))
                .then(pl.lit('P2'))
                .when(pl.col(f'last_{i}_month_usage_gb').is_between(pl.col('p50'), pl.col('p75'), closed='right'))
                .then(pl.lit('P3'))
                .when(pl.col(f'last_{i}_month_usage_gb').is_between(pl.col('p75'), pl.col('available_gb'), closed='right'))
                .then(pl.lit('P4'))
                .when(pl.col(f'last_{i}_month_usage_gb') > pl.col('available_gb'))
                .then(pl.lit('P5'))  # how many times has exceeded the available data
                .otherwise(pl.lit(None))
                .alias(f'month_{i}_threshold')
            )
            percentile_exprs.append(percentile_expr)
        
        # Count times in each percentile
        count_exprs = []
        for p in range(1, 6):
            count_expr = sum(
                (pl.col(f'month_{i}_threshold') == f'P{p}').cast(pl.Int32)
                for i in range(1, 5)
            ).alias(f'times_in_p{p}')
            count_exprs.append(count_expr)
        
        # Apply transformations and clean up - also remove monthly usage features after percentile calculation
        features = (
            features
            .with_columns(percentile_exprs)
            .with_columns(count_exprs)
            .drop(['p25', 'p50', 'p75'] + [f'month_{i}_threshold' for i in range(1, 5)] + 
                  [f'last_{i}_month_usage_gb' for i in range(1, 5)])  # Remove monthly usage features
        )
        
        return features
    
    def fill_interaction_nulls(self, features: pl.DataFrame) -> pl.DataFrame:
        """
        Fill null values in interaction features with appropriate defaults.
        
        Args:
            features: Features dataframe
            
        Returns:
            Features with filled interaction nulls
        """
        # Fill n_ columns with 0, days_since_last columns with -1
        n_columns = [col for col in features.columns if col.startswith('n_')]
        days_columns = [col for col in features.columns if col.startswith('days_since_last')]
        
        fill_exprs = []
        
        # Fill n_ columns with 0
        for col in n_columns:
            fill_exprs.append(
                pl.when(pl.col(col).is_null())
                .then(0)
                .otherwise(pl.col(col))
                .alias(col)
            )
        
        # Fill days_since_last columns with -1
        for col in days_columns:
            fill_exprs.append(
                pl.when(pl.col(col).is_null())
                .then(-1)
                .otherwise(pl.col(col))
                .alias(col)
            )
        
        if fill_exprs:
            features = features.with_columns(fill_exprs)
        
        return features
    
    def select_final_features(self, features: pl.DataFrame) -> pl.DataFrame:
        """
        Select only the final features needed for model inference.
        
        Args:
            features: Full feature set
            
        Returns:
            Final feature set with only required columns
        """
        # Select only the columns that exist in both the dataframe and final_columns
        existing_final_columns = [
            col for col in self.final_columns 
            if col in features.columns
        ]
        
        # Warning for missing columns
        missing_columns = [
            col for col in self.final_columns 
            if col not in features.columns
        ]
        
        if missing_columns:
            print(f"Warning: Missing expected columns: {missing_columns}")
        
        # Select final features
        features_final = features.select(existing_final_columns)
        
        return features_final
    
    def run_transform(
        self, 
        core_data: pl.DataFrame,
        usage_info: pl.DataFrame, 
        customer_interactions: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Main ETL pipeline processing all input data into cleaned features.
        
        Args:
            core_data: Raw core customer data
            usage_info: Raw usage information
            customer_interactions: Raw customer interactions
            
        Returns:
            Cleaned features ready for ML model inference
        """
        # Start ETL pipeline
        print("Starting ETL pipeline...")

        # Process individual datasets
        print("Processing core customer data...")
        core_processed = self.process_core_data(core_data)

        print("Processing usage information...")
        usage_processed = self.process_usage_info(usage_info)

        print("Processing customer interactions...")
        interactions_processed = self.process_customer_interactions(customer_interactions)

        # Combine all features
        print("Combining feature sets...")
        features = core_processed.join(
            usage_processed, on='rating_account_id', how='left'
        ).join(
            interactions_processed, on='customer_id', how='left'
        )

        # Fill interaction nulls
        print("Filling missing values in interaction features...")
        features = self.fill_interaction_nulls(features)

        # Impute missing available_gb (needs monthly usage features)
        print("Imputing missing 'available_gb' values...")
        features = self.impute_available_gb(features)

        # Create usage percentile features (will remove monthly usage features after)
        print("Creating usage percentile features...")
        features = self.create_usage_percentile_features(features)

        # Select final features for model inference
        print("Selecting final features for model inference...")
        features_final = self.select_final_features(features)

        print(f"ETL pipeline completed. Output shape: {features_final.shape}")

        return features_final