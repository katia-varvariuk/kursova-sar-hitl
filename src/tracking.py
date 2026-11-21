"""Налаштування MLflow (сервер з docker-compose або локальний SQLite).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MLRUNS = ROOT / "mlruns"


SERVER_URI = "http://localhost:5000"


def _server_alive(uri: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{uri}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def setup(experiment: str) -> str:
    """Основний сервер — MLflow з docker-compose (артефакти через HTTP)."""
    MLRUNS.mkdir(exist_ok=True)
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        uri = SERVER_URI if _server_alive(SERVER_URI) else f"sqlite:///{(MLRUNS / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        if uri.startswith("sqlite"):
            mlflow.create_experiment(experiment, artifact_location=(MLRUNS / "artifacts").as_uri())
        else:
            mlflow.create_experiment(experiment)
    mlflow.set_experiment(experiment)
    print(f"MLflow: {uri} / {experiment}")
    return uri


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "uncommitted"


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
