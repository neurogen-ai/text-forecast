"""Client-side MLflow run creation shared by every runtime (plan 2.0 P6).

The runtime wrapper owns the run lifecycle: it creates the run on the client
(so naming and parenting stay client-side), stamps the id onto the job via
``dataclasses.replace``, and only then dispatches. The pipeline resumes the
run by id inside whatever venue executes it.
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from config.env import Env


def create_run(
    *,
    env: Env,
    experiment_name: str,
    run_name: str,
    parent_id: str | None = None,
) -> str:
    """Create an MLflow run client-side and return its id.

    Uses the low-level client so the fluent active-run stack stays free for
    the pipeline to resume by ``run_id``.
    """
    mlflow.set_tracking_uri(env.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    tags = {MLFLOW_PARENT_RUN_ID: parent_id} if parent_id else None
    run = client.create_run(
        experiment_id=experiment_id, run_name=run_name, tags=tags
    )
    return run.info.run_id


def set_run_name(run_id: str, name: str) -> None:
    """Update the display name of an existing run.

    The wrapper creates runs before the model class is known; pipelines call
    this once they can compose the final name.
    """
    MlflowClient().set_tag(run_id, "mlflow.runName", name)
