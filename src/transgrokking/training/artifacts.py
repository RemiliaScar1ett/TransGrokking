"""Run-directory creation and machine-readable logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transgrokking.utils.atomic import write_json


def create_run_dir(runs_dir: str | Path, config_hash: str) -> Path:
    """Create a unique run directory with the M0 artifact layout."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(runs_dir) / f"{timestamp}_{config_hash[:8]}"
    for child in ("metrics", "checkpoints", "tensors", "figures", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    return run_dir


def append_scalar(path: str | Path, record: dict[str, Any]) -> None:
    """Append one compact JSON record and flush it to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_status(run_dir: str | Path, state: str, **details: Any) -> None:
    """Atomically update run status."""
    write_json(
        Path(run_dir) / "status.json",
        {"state": state, "updated_at": datetime.now(timezone.utc).isoformat(), **details},
    )
