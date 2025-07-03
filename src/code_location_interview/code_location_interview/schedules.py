"""
Dagster Schedules for ML Pipeline
"""

import dagster as dg
from .jobs import master_pipeline_job, training_pipeline_job

# Master Pipeline Schedule
master_pipeline_schedule = dg.ScheduleDefinition(
    job=master_pipeline_job,
    cron_schedule="0 2 5 * *",  # 2 AM on the 5th day of every month
    name="master_pipeline_schedule",
    description="Runs master pipeline monthly on the 5th"
)

# Training Pipeline Schedule
training_pipeline_schedule = dg.ScheduleDefinition(
    job=training_pipeline_job,
    cron_schedule="0 2 1 */3 *",  # 2 AM on the 1st day every 3 months
    name="training_pipeline_schedule", 
    description="Runs training pipeline quarterly"
)