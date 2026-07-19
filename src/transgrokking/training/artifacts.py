"""Run-directory lifecycle and validated machine-readable logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transgrokking.utils.atomic import write_json


def create_run_dir(runs_dir: str | Path, config_hash: str) -> Path:
    """Create a unique run directory and immediately mark it initializing."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(runs_dir) / f"{timestamp}_{config_hash[:8]}"
    for child in ("metrics", "checkpoints", "tensors", "figures", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    write_status(run_dir, "initializing", global_step=None)
    return run_dir


def scalar_steps(path: str | Path) -> list[int]:
    """Read and validate strictly increasing scalar steps."""
    source = Path(path)
    if not source.exists():
        return []
    steps = [json.loads(line)["step"] for line in source.read_text(encoding="utf-8").splitlines()]
    if any(type(step) is not int for step in steps) or any(
        current <= previous for previous, current in zip(steps, steps[1:], strict=False)
    ):
        raise ValueError(f"scalar steps are not strictly increasing: {steps}")
    return steps


def append_scalar(path: str | Path, record: dict[str, Any]) -> None:
    """Append one scalar record only when its step is strictly newer."""
    destination = Path(path)
    previous = scalar_steps(destination)
    step = record.get("step")
    if type(step) is not int or (previous and step <= previous[-1]):
        raise ValueError(f"scalar step {step!r} must be greater than {previous[-1:]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def load_manifest(
    run_dir: str | Path, *, allow_unreferenced: set[Path] | None = None
) -> list[dict[str, object]]:
    """Load a manifest and verify unique steps and exact checkpoint file agreement."""
    root = Path(run_dir)
    manifest_path = root / "checkpoints" / "manifest.json"
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("checkpoints")
    if payload.get("schema_version") != 2 or not isinstance(entries, list):
        raise ValueError("invalid checkpoint manifest schema")
    steps = [entry.get("step") for entry in entries]
    if any(type(step) is not int for step in steps) or len(steps) != len(set(steps)):
        raise ValueError(f"manifest checkpoint steps must be unique integers: {steps}")
    paths = [entry.get("path") for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("manifest checkpoint paths must be unique")
    referenced = {root / "checkpoints" / str(relative) for relative in paths}
    if any(not path.is_file() for path in referenced):
        raise ValueError("manifest references a missing checkpoint file")
    actual = set((root / "checkpoints").glob("step_*.pt"))
    allowed = allow_unreferenced or set()
    if actual - allowed != referenced or not allowed.issubset(actual):
        raise ValueError("checkpoint files and manifest entries do not match")
    return sorted(entries, key=lambda entry: int(entry["step"]))


def add_manifest_checkpoint(run_dir: str | Path, step: int, checkpoint: str | Path) -> None:
    """Atomically add a fully written checkpoint to the manifest."""
    root = Path(run_dir)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise ValueError(f"cannot manifest incomplete checkpoint: {checkpoint_path}")
    entries = load_manifest(root, allow_unreferenced={checkpoint_path})
    if any(entry["step"] == step for entry in entries):
        raise ValueError(f"manifest already contains checkpoint step {step}")
    relative = checkpoint_path.relative_to(root / "checkpoints").as_posix()
    entries.append({"step": step, "path": relative})
    entries.sort(key=lambda entry: int(entry["step"]))
    write_json(
        root / "checkpoints" / "manifest.json",
        {"schema_version": 2, "checkpoints": entries},
    )


def write_status(run_dir: str | Path, state: str, **details: Any) -> None:
    """Atomically update run lifecycle state."""
    write_json(
        Path(run_dir) / "status.json",
        {"state": state, "updated_at": datetime.now(timezone.utc).isoformat(), **details},
    )
