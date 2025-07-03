import os

from dagster import get_dagster_logger
from dagster_duckdb import DuckDBResource

from shared_library.orchestration.resources.utils import (
    get_dagster_deployment_environment,
)

# Set environment variable
# export DUCKDB_DATABASE="/path/to/my/custom/database.duckdb"

# Main DuckDB resource for the data warehouse
dev_database_resource = DuckDBResource(
    database=os.getenv("DUCKDB_DATABASE", "/workspaces/data-scientist-at-magenta/src/code_location_interview/code_location_interview/dwh/database.duckdb")
)

prod_database_resource = DuckDBResource(
    database=os.getenv("PROD_DUCKDB_DATABASE", "/workspaces/data-scientist-at-magenta/src/code_location_interview/code_location_interview/dwh/production_database.duckdb")
)

RESOURCES_LOCAL = {
    "database": dev_database_resource
}

RESOURCES_PRODUCTION = {
    "prod_database": prod_database_resource
}


resource_defs_by_deployment_name = {
    "dev": RESOURCES_LOCAL,
    "prod": RESOURCES_PRODUCTION
}


def get_resources_for_deployment(log_env: bool = True):
    deployment_name = get_dagster_deployment_environment()
    if log_env:
        get_dagster_logger().info(f"Using deployment of: {deployment_name}")

    return resource_defs_by_deployment_name[deployment_name]