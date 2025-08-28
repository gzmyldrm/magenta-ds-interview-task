import logging
import sys
from typing import List

import pandas as pd
from dagster import AssetIn, asset, get_dagster_logger

log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

GROUP_NAME = "process_raw_data"


@asset(
    group_name=GROUP_NAME,
    ins={"core_data": AssetIn()},
    description="Process the core data including feature engineering and bucketing.",
)
def processed_core_data(core_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process core data and derive engineered features.

    Notes:
    - Never mutate the input DataFrame in place (copy first).
    - All categorical bucket features are one-hot encoded with drop_first=True to avoid multicollinearity.
    - Missing numeric (available_gb) is flagged explicitly to preserve information.
    """
    df = core_data.copy()
    logger.info("Processing raw core data with %s records.", len(df))

    # Numeric coercion & missingness flag
    df["available_gb"] = pd.to_numeric(df["available_gb"], errors="coerce")
    df["available_gb_missing"] = df["available_gb"].isnull().astype(int)
    df["available_gb"] = df["available_gb"].fillna(df["available_gb"].median())

    # Device related (premium heuristic)
    df["smartphone_brand"] = df["smartphone_brand"].astype("category")
    # Create a binary feature for premium devices
    df["is_premium_device"] = df["smartphone_brand"].isin(["iPhone", "Samsung"]).astype(int)

    # Discrete buckets (simple, interpretable segmentation)
    df["remaining_binding_days_bucket"] = pd.cut(
        df["remaining_binding_days"],
        bins=[-1000, -1, 30, 90, 180, 1000],
        labels=["Expired", "0-30", "31-90", "91-180", ">180"],
    )
    df["mrc_bucket"] = pd.cut(
        df["gross_mrc"],
        bins=[0, 30, 50, 70],
        labels=["Low (<30€)", "Mid (30-50€)", "High (>50€)"],
    )
    df["contract_term_bucket"] = pd.cut(
        df["contract_lifetime_days"],
        bins=[0, 365, 730, 1095, 1825],
        labels=["<1y", "1-2y", "2-3y", "3-5y"],
    )
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 25, 45, 60, 100],
        labels=["<25", "25-45", "46-60", ">60"],
    )

    # One-hot encode selected categorical columns 
    cat_cols: List[str] = [
        "remaining_binding_days_bucket",
        "mrc_bucket",
        "contract_term_bucket",
        "age_bucket",
        "smartphone_brand",
    ]
    dummies = pd.get_dummies(df[cat_cols], drop_first=True, dtype=int)
    df = pd.concat([df, dummies], axis=1)

    logger.info("Core data shape after processing: %s (cols=%s)", df.shape, len(df.columns))
    logger.info("Core data null counts after processing:\n%s", df.isnull().sum())
    return df


@asset(
    group_name=GROUP_NAME,
    ins={"usage_info": AssetIn()},
    description="Process and aggregate the usage information.",
)
def agg_processed_usage_info(usage_info: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate usage metrics per rating_account_id.

    Adds:
    - Basic distribution stats over used_gb (sum/mean/std/min/max).
    - Month count, roaming counts & ratio.
    - Simple trend features (absolute change + increase flag).
    """
    df = usage_info.copy()
    df["billed_period_month_d"] = pd.to_datetime(df["billed_period_month_d"])
    logger.info("Usage data null counts after processing:\n%s", df.isnull().sum())

    # Core aggregations
    agg_df = (
        df.groupby("rating_account_id")
        .agg(
            total_used_gb=("used_gb", "sum"),
            avg_used_gb=("used_gb", "mean"),
            std_used_gb=("used_gb", "std"),
            min_used_gb=("used_gb", "min"),
            max_used_gb=("used_gb", "max"),
            usage_month_count=("billed_period_month_d", "count"),
            roaming_months=("has_used_roaming", "sum"),
        )
        .reset_index()
    )

    # Trend features: first vs last month usage
    usage_trend_df = (
        df.sort_values(["rating_account_id", "billed_period_month_d"])
        .groupby("rating_account_id")["used_gb"]
        .agg(["first", "last"])
        .reset_index()
    )
    usage_trend_df["usage_increase"] = (usage_trend_df["last"] > usage_trend_df["first"]).astype(int)
    usage_trend_df["usage_trend"] = usage_trend_df["last"] - usage_trend_df["first"]

    # Derived ratios / indicators
    agg_df["roaming_ratio"] = agg_df["roaming_months"] / agg_df["usage_month_count"]
    agg_df["roaming_ratio"] = agg_df["roaming_ratio"].fillna(0.0)
    agg_df["had_roaming"] = (agg_df["roaming_months"] > 0).astype(int)

    # Merge trend features
    agg_df = agg_df.merge(
        usage_trend_df[["rating_account_id", "usage_increase", "usage_trend"]],
        on="rating_account_id",
        how="left",
    )
    return agg_df


@asset(
    group_name=GROUP_NAME,
    ins={"customer_interactions": AssetIn()},
    description="Process and aggregate the customer interactions.",
)
def agg_processed_customer_interactions(customer_interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate interaction metrics keyed by customer_id.

    - Total & unique interaction counts.
    - Recency (min days_since_last).    
    """
    df = customer_interactions.copy()

    def count_interaction(pattern: str, col_name: str) -> pd.Series:
        return (
            df[df["type_subtype"].str.contains(pattern, case=False, na=False)]
            .groupby("customer_id")["n"]
            .sum()
            .rename(col_name)
        )

    prolongations = count_interaction("prolongation", "interact_prolongations")
    tarifdetails = count_interaction("tarifdetails", "interact_tarifdetails")
    tarifwechsel = count_interaction("tarifwechsel", "interact_tarifwechsel")
    rechnungsanfragen = count_interaction("rechnungsanfragen", "interact_rechnungsanfragen")

    interactions_per_customer = df.groupby("customer_id")["n"].sum().rename("num_interactions")
    unique_interactions_per_customer = (
        df.groupby("customer_id")["type_subtype"].nunique().rename("num_unique_interactions")
    )
    recency = df.groupby("customer_id")["days_since_last"].min().rename("recency")

    feature_series = [
        interactions_per_customer,
        unique_interactions_per_customer,
        recency,
        prolongations,
        tarifdetails,
        tarifwechsel,
        rechnungsanfragen,
    ]
    out_df = pd.concat(feature_series, axis=1).fillna(0).astype(int).reset_index()
    
    # for recency 0 does not make sense
    # filled with a large number
    out_df["recency"] = out_df["recency"].fillna(999)

    return out_df


@asset(
    group_name=GROUP_NAME,
    ins={
        "processed_core_data": AssetIn(),
        "agg_processed_usage_info": AssetIn(),
        "agg_processed_customer_interactions": AssetIn(),
    },
    description="Merge processed core, usage, and interaction datasets.",
)
def processed_and_merged_data(
    processed_core_data: pd.DataFrame,
    agg_processed_usage_info: pd.DataFrame,
    agg_processed_customer_interactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all engineered feature sets.

    Keys:
    - rating_account_id for usage aggregation.
    - customer_id for interaction aggregation.

    """
    merged = (
        processed_core_data.merge(agg_processed_usage_info, on="rating_account_id", how="left")
        .merge(agg_processed_customer_interactions, on="customer_id", how="left")
    )

    logger.info(
        "Merged processed data shape: %s (cols=%s)",
        merged.shape,
        len(merged.columns),
    )
    return merged
