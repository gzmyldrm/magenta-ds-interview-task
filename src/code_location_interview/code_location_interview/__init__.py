from dagster import (
    AutomationCondition,
    Definitions,
    load_asset_checks_from_package_module,
    load_assets_from_package_module,
    with_source_code_references,
)

# from dagster_cloud.metadata.source_code import link_code_references_to_git_if_cloud
from code_location_interview import assets

from .resources import get_resources_for_deployment

# Import jobs, schedules, and sensors
from .jobs import (
    data_pipeline_job,
    action_pipeline_job,
    training_pipeline_job,
    master_pipeline_job
)
from .schedules import (
    master_pipeline_schedule,
    training_pipeline_schedule
)
from .sensors import drift_detection_sensor

resource_defs = get_resources_for_deployment()

# resource_defs = get_resources_for_deployment()
all_assets = with_source_code_references(
    [
        *load_assets_from_package_module(
            assets,
            automation_condition=AutomationCondition.eager(),
        ),
    ]
)
all_asset_checks = [*load_asset_checks_from_package_module(assets)]

defs = Definitions(
    assets=all_assets,
    asset_checks=all_asset_checks,
    schedules=[
        master_pipeline_schedule,
        training_pipeline_schedule,
    ],
    sensors=[
        drift_detection_sensor,
    ],
    jobs=[
        data_pipeline_job,
        action_pipeline_job,
        training_pipeline_job,
        master_pipeline_job,
    ],
    resources=resource_defs,
)