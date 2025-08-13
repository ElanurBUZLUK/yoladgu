from __future__ import annotations

from datetime import timedelta
import os

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator


def _repo_root() -> str:
    # Prefer Airflow Variable if set; else env; else fallback to hardcoded path
    try:
        from airflow.models import Variable  # type: ignore
        val = Variable.get("REPO_ROOT", default_var=None)
        if val:
            return val
    except Exception:
        pass
    return os.getenv("REPO_ROOT", "/home/ela/Downloads/yoladgu_full_repo_v2")


REPO = _repo_root()
ENV = os.path.join(REPO, "backend", ".env")


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="cf_retrain_and_index",
    default_args=default_args,
    description="Export interactions -> train CF -> build index -> swap active index",
    schedule_interval="0 2 * * *",  # daily at 02:00
    start_date=days_ago(1),
    catchup=False,
) as dag:
    export_interactions = BashOperator(
        task_id="export_interactions",
        bash_command=f"cd {REPO} && ENV_FILE={ENV} PYTHONPATH=backend python tools/export_interactions.py --out examples/interactions.jsonl --lookback_days 180",
    )

    train_cf = BashOperator(
        task_id="train_cf",
        bash_command=f"cd {REPO} && ENV_FILE={ENV} PYTHONPATH=backend python tools/train_cf.py --data examples/interactions.jsonl",
    )

    build_index = BashOperator(
        task_id="build_index",
        bash_command=f"cd {REPO} && ENV_FILE={ENV} PYTHONPATH=backend python tools/batch_indexer.py",
    )

    swap_index = BashOperator(
        task_id="swap_index",
        bash_command=(
            "cd {repo} && ENV_FILE={env} PYTHONPATH=backend "
            "python -c \"from app.services.vector_index_manager import VectorIndexManager; "
            "from app.core.config import settings; print(VectorIndexManager(settings.REDIS_URL).swap())\""
        ).format(repo=REPO, env=ENV),
    )

    export_interactions >> train_cf >> build_index >> swap_index


