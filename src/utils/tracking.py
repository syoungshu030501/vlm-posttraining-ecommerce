from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional


def init_swanlab(
    *,
    stage: str,
    config: Dict[str, Any],
    project: str = "vlm-posttraining",
    experiment_name: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    description: Optional[str] = None,
):
    """Initialise SwanLab when credentials are available."""
    api_key = os.environ.get("SWANLAB_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        import swanlab
    except Exception as exc:
        print(f"[WARN] SwanLab import failed: {exc}")
        return None

    try:
        swanlab.login(api_key=api_key, save=False)
        swanlab.init(
            project=project,
            experiment_name=experiment_name or stage,
            config={**config, "stage": stage},
            tags=list(tags or [stage]),
            description=description,
            mode=os.environ.get("SWANLAB_MODE", "cloud"),
        )
        return swanlab
    except Exception as exc:
        print(f"[WARN] SwanLab init failed: {exc}")
        return None


def log_metrics(tracker, metrics: Dict[str, Any]) -> None:
    if tracker is None:
        return
    try:
        tracker.log(metrics)
    except Exception as exc:
        print(f"[WARN] SwanLab log failed: {exc}")


def finish_run(tracker) -> None:
    if tracker is None:
        return
    try:
        if hasattr(tracker, "finish"):
            tracker.finish()
    except Exception as exc:
        print(f"[WARN] SwanLab finish failed: {exc}")
