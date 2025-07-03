"""
Data Warehouse Initialization Script

Creates all required tables for the ML pipeline based on pipeline_design.md
and data structures from notebook 0.data_exploration and lib/etl.py

Tables created:
- Raw data tables (core_data, usage_info, customer_interactions, labels)
- Feature tables (processed_features, feature_statistics)
- Model registry and artifacts
- Prediction and monitoring tables
- Performance tracking tables
"""

import duckdb
from pathlib import Path
from typing import Optional
import sys


def create_raw_data_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create raw data ingestion tables."""
    
    # Core customer and contract data
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_core_data (
            rating_account_id VARCHAR PRIMARY KEY,
            customer_id VARCHAR NOT NULL,
            age INTEGER,
            contract_lifetime_days INTEGER,
            remaining_binding_days INTEGER,
            has_special_offer BOOLEAN,
            is_magenta1_customer BOOLEAN,
            available_gb INTEGER,
            gross_mrc DOUBLE,
            smartphone_brand VARCHAR,
            has_done_upselling BOOLEAN,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Monthly usage information
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_usage_info (
            rating_account_id VARCHAR NOT NULL,
            billed_period_month_d DATE NOT NULL,
            has_used_roaming BOOLEAN,
            used_gb DOUBLE,
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (rating_account_id, billed_period_month_d),
        )
    """)
    
    # Customer service interactions
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_customer_interactions (
            customer_id VARCHAR NOT NULL,
            type_subtype VARCHAR NOT NULL,
            n INTEGER,  -- Number of interactions in last 6 months
            days_since_last INTEGER,  -- Days since last interaction (-1 if never)
            ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (customer_id, type_subtype)
        )
    """)
    
    # Labels table for supervised learning
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_labels (
            rating_account_id VARCHAR PRIMARY KEY,
            target_label BOOLEAN NOT NULL,  -- has_done_upselling
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        )
    """)


def create_feature_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create processed features and statistics tables."""
    
    # Processed features ready for ML (based on final_columns from etl.py)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_features (
            rating_account_id VARCHAR PRIMARY KEY,
            customer_id VARCHAR NOT NULL,
            
            -- Customer demographics
            age INTEGER,
            
            -- Contract information
            contract_lifetime_days INTEGER,
            remaining_binding_days INTEGER,
            has_special_offer BOOLEAN,
            is_magenta1_customer BOOLEAN,
            available_gb INTEGER,
            gross_mrc DOUBLE,
            has_done_upselling BOOLEAN,
            completion_rate DOUBLE,
            is_bounded BOOLEAN,
            
            -- Device information (one-hot encoded smartphone brands)
            is_huawei BOOLEAN,
            is_oneplus BOOLEAN,
            is_samsung BOOLEAN,
            is_xiaomi BOOLEAN,
            is_iphone BOOLEAN,
            
            -- Customer aggregations
            n_contracts_per_customer INTEGER,
            
            -- Usage statistics
            avg_monthly_usage_gb DOUBLE,
            total_usage_gb DOUBLE,
            max_monthly_usage_gb DOUBLE,
            months_with_roaming INTEGER,
            ever_used_roaming BOOLEAN,
            active_usage_months INTEGER,
            
            -- Usage trends and deltas
            months_with_no_delta_1mo_change INTEGER,
            avg_delta_2mo DOUBLE,
            delta_2mo_volatility DOUBLE,
            max_delta_2mo_increase DOUBLE,
            max_delta_2mo_decrease DOUBLE,
            months_with_delta_2mo_increase INTEGER,
            months_with_no_delta_2mo_change INTEGER,
            months_with_delta_3mo_increase INTEGER,
            months_with_no_delta_3mo_change INTEGER,
            
            -- Recent usage deltas
            last_1_delta_1mo DOUBLE,
            last_2_delta_1mo DOUBLE,
            last_3_delta_1mo DOUBLE,
            last_1_delta_2mo DOUBLE,
            last_2_delta_2mo DOUBLE,
            last_1_delta_3mo DOUBLE,
            
            -- Customer service interactions
            n_rechnungsanfragen INTEGER DEFAULT 0,
            n_produkte_services_tarifdetails INTEGER DEFAULT 0,
            n_prolongation INTEGER DEFAULT 0,
            n_produkte_services_tarifwechsel INTEGER DEFAULT 0,
            days_since_last_rechnungsanfragen INTEGER DEFAULT -1,
            days_since_last_produkte_services_tarifdetails INTEGER DEFAULT -1,
            days_since_last_prolongation INTEGER DEFAULT -1,
            days_since_last_produkte_services_tarifwechsel INTEGER DEFAULT -1,
            
            -- Usage percentile features
            times_in_p1 INTEGER DEFAULT 0,
            times_in_p2 INTEGER DEFAULT 0,
            times_in_p3 INTEGER DEFAULT 0,
            times_in_p4 INTEGER DEFAULT 0,
            times_in_p5 INTEGER DEFAULT 0,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        )
    """)
    
    # Feature statistics for drift detection
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_statistics (
            feature_name VARCHAR NOT NULL,
            
            -- Numerical statistics
            mean_value DOUBLE,
            std_value DOUBLE,
            
            -- Metadata
            computation_timestamp DATETIME NOT NULL,
            PRIMARY KEY (feature_name, computation_timestamp)
        )
    """)


def create_model_registry_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create model registry and artifact storage tables."""
    
    # Model registry with versioning
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id VARCHAR PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            version VARCHAR NOT NULL,
            model_artifact BLOB,  -- Store pickled model here
            metadata JSON,  -- Model parameters, hyperparameters, etc.
            
            -- Model status and lifecycle
            status VARCHAR DEFAULT 'trained',  -- 'trained', 'validated', 'deployed', 'retired'
            is_current_model BOOLEAN DEFAULT FALSE,
            
            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def create_prediction_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create prediction and inference tables."""
    
    # Prediction results
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id VARCHAR PRIMARY KEY,
            rating_account_id VARCHAR NOT NULL,
            customer_id VARCHAR NOT NULL,
            
            -- Prediction results

            prediction_score DOUBLE NOT NULL,
            predicted_class BOOLEAN NOT NULL,

            -- Model information
            model_id VARCHAR NOT NULL,
                 
            -- Data information
            features_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
            
            -- Prediction metadata
            prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def create_actions_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create actions tracking tables."""
    
    # Actions table to track when customers/contracts have been actioned
    conn.execute("""
        CREATE TABLE IF NOT EXISTS actions (
            action_id VARCHAR PRIMARY KEY,
            customer_id VARCHAR NOT NULL,
            rating_account_id VARCHAR NOT NULL,
            model_id VARCHAR NOT NULL,
            
            -- Action metadata
            action_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            action_type VARCHAR DEFAULT 'upselling_offer',
            
            -- Reference to prediction that triggered the action
            prediction_id VARCHAR,
            
            -- Action outcome tracking (can be updated later)
            status VARCHAR DEFAULT 'sent',  -- 'sent', 'accepted', 'rejected', 'expired'
            
            FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
        )
    """)


# def create_indexes(conn: duckdb.DuckDBPyConnection) -> None:
#     """Create indexes for performance optimization."""
    
#     print("Creating performance indexes...")
    
#     # Raw data indexes
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_core_customer ON raw_core_data(customer_id)")
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_usage_account_date ON raw_usage_info(rating_account_id, billed_period_month_d)")
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_interactions_customer ON raw_customer_interactions(customer_id)")
    
#     # Feature table indexes
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_features_customer ON processed_features(customer_id)")
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_features_created ON processed_features(created_at)")
    
#     # Feature statistics indexes
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_feature_stats_name_date ON feature_statistics(feature_name, computation_date)")
    
#     # Model registry indexes
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name_version ON model_registry(model_name, version)")
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_model_status ON model_registry(status)")
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_model_current ON model_registry(is_current_model)")
    
#     # Prediction indexes
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_account ON predictions(rating_account_id)")
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(prediction_timestamp)")
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_id)")
    
#     # Monitoring indexes
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_drift_alerts_type_status ON drift_alerts(alert_type, status)")
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_drift_alerts_created ON drift_alerts(created_at)")
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_model_date ON model_performance(model_id, evaluation_start_date)")
    
#     # Job monitoring indexes
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_job_logs_name_status ON job_execution_logs(job_name, status)")
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_job_logs_timestamp ON job_execution_logs(start_timestamp)")
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_status_type ON sensor_status(sensor_type)")


def create(db_path: Optional[str] = None) -> None:
    """
    Initialize the complete data warehouse with all required tables.
    
    Args:
        db_path: Path to DuckDB database file. If None, creates in-memory database.
    """
    if db_path is None:
        db_path = "magenta_datawarehouse.duckdb"
    
    print(f"Initializing data warehouse at: {db_path}")
    
    # Connect to DuckDB
    conn = duckdb.connect(db_path)
    
    try:
        # Create all table groups
        print("Creating raw data tables...")
        create_raw_data_tables(conn)
        
        print("Creating feature tables...")
        create_feature_tables(conn)
        
        print("Creating model registry tables...")
        create_model_registry_tables(conn)
        
        print("Creating prediction tables...")
        create_prediction_tables(conn)

        print("Creating actions tables...")
        create_actions_tables(conn)
        
        # Verify tables were created
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"\nSuccessfully created {len(tables)} tables:")
        for table in sorted(tables):
            print(f"  - {table[0]}")
        
        print(f"\nData warehouse initialization completed successfully!")
        print(f"Database file: {Path(db_path).absolute()}")
        
    except Exception as e:
        print(f"Error initializing data warehouse: {e}")
        raise
    finally:
        conn.close()