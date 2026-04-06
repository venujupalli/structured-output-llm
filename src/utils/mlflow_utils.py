from __future__ import annotations

import logging
import os
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import urlparse


LOGGER = logging.getLogger(__name__)


def configure_mlflow(mlflow_module: Any, experiment_name: str, *, root_dir: Path, logger: logging.Logger | None = None) -> None:
    logger = logger or LOGGER

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = f"sqlite:///{(root_dir / 'mlflow.db').resolve()}"
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    artifact_root = Path(
        os.getenv("MLFLOW_ARTIFACT_ROOT", str((root_dir / "mlruns").resolve()))
    ).expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_ARTIFACT_ROOT", str(artifact_root))

    mlflow_module.set_tracking_uri(tracking_uri)
    _ensure_local_experiment_artifact_root(
        mlflow_module,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
        logger=logger,
    )
    mlflow_module.set_experiment(experiment_name)


def safe_log_artifacts(
    mlflow_module: Any,
    local_dir: str,
    *,
    artifact_path: str | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    logger = logger or LOGGER
    try:
        mlflow_module.log_artifacts(local_dir, artifact_path=artifact_path)
    except Exception as exc:  # pragma: no cover - depends on runtime store config
        logger.warning("MLflow artifact logging failed for %s: %s", local_dir, exc)
        return False
    return True


def safe_log_artifact(
    mlflow_module: Any,
    local_path: str,
    *,
    artifact_path: str | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    logger = logger or LOGGER
    try:
        mlflow_module.log_artifact(local_path, artifact_path=artifact_path)
    except Exception as exc:  # pragma: no cover - depends on runtime store config
        logger.warning("MLflow artifact logging failed for %s: %s", local_path, exc)
        return False
    return True


def _ensure_local_experiment_artifact_root(
    mlflow_module: Any,
    *,
    experiment_name: str,
    tracking_uri: str,
    artifact_root: Path,
    logger: logging.Logger,
) -> None:
    desired_artifact_location = str((artifact_root / experiment_name).resolve())
    Path(desired_artifact_location).mkdir(parents=True, exist_ok=True)

    client = mlflow_module.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name, artifact_location=desired_artifact_location)
        logger.info(
            "Created MLflow experiment %s with artifact root %s",
            experiment_name,
            desired_artifact_location,
        )
        return

    if experiment.artifact_location == desired_artifact_location:
        return

    db_path = _sqlite_db_path(tracking_uri)
    if db_path is None:
        logger.warning(
            "MLflow experiment %s uses artifact root %s, but the tracking store is not local sqlite; leaving it unchanged.",
            experiment_name,
            experiment.artifact_location,
        )
        return

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
            (desired_artifact_location, experiment.experiment_id),
        )
        connection.commit()

    logger.warning(
        "Updated MLflow artifact root for experiment %s from %s to %s",
        experiment_name,
        experiment.artifact_location,
        desired_artifact_location,
    )


def _sqlite_db_path(tracking_uri: str) -> Path | None:
    parsed = urlparse(tracking_uri)
    if parsed.scheme != "sqlite":
        return None
    return Path(parsed.path).expanduser().resolve()
