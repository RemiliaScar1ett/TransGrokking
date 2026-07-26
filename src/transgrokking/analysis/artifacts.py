"""Atomic lifecycle and provenance helpers for read-only M2 analyses."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transgrokking.utils.atomic import write_json


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def repository_relative(path: str | Path) -> str:
    """Return a portable repository-relative POSIX path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repository_root()).as_posix()
    except ValueError as error:
        raise ValueError(f"path is outside repository: {resolved}") from error


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_tree(path: str | Path) -> list[dict[str, Any]]:
    """Hash every file under a source tree without modifying it."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"inventory root is not a directory: {root}")
    records: list[dict[str, Any]] = []
    for file in sorted(item for item in root.rglob("*") if item.is_file()):
        if file.is_symlink():
            raise ValueError(f"inventory refuses symbolic link: {file}")
        try:
            file.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"inventory file escapes source root: {file}") from error
        records.append(
            {
                "path": file.relative_to(root).as_posix(),
                "size": file.stat().st_size,
                "sha256": sha256_file(file),
            }
        )
    return records


def inventory_files(root: str | Path, relative_paths: Iterable[str]) -> list[dict[str, Any]]:
    source = Path(root).resolve()
    records: list[dict[str, Any]] = []
    for relative in sorted(set(relative_paths)):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"inventory path must be normalized and relative: {relative}")
        path = (source / relative).resolve()
        try:
            path.relative_to(source)
        except ValueError as error:
            raise ValueError(f"inventory path escapes source root: {relative}") from error
        if not path.is_file():
            raise ValueError(f"inventory source file is missing: {path}")
        records.append(
            {
                "path": Path(relative).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def create_analysis_dir(parent: str | Path, analysis_hash: str) -> Path:
    """Create an independent M2 analysis lifecycle directory."""
    expected_parent = (repository_root() / "analysis_runs").resolve()
    selected_parent = Path(parent).resolve()
    if selected_parent != expected_parent:
        raise ValueError(f"analysis parent must equal {expected_parent}, got {selected_parent}")
    selected_parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = selected_parent / f"{timestamp}_{analysis_hash[:8]}"
    root.mkdir(exist_ok=False)
    write_analysis_status(
        root,
        analysis_status="initializing",
        m2a_status="pending",
        m2b_status="pending",
        export_status="pending",
    )
    try:
        for child in (
            "audit",
            "cache",
            "context",
            "figures",
            "logs",
            "m2a",
            "m2b",
            "provenance",
            "selected_tensors",
        ):
            (root / child).mkdir(exist_ok=False)
    except Exception as error:
        write_analysis_status(
            root,
            analysis_status="failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    return root


def write_analysis_status(root: str | Path, **updates: Any) -> dict[str, Any]:
    """Atomically merge lifecycle fields into status.json."""
    path = Path(root) / "status.json"
    current: dict[str, Any] = {}
    if path.is_file():
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("analysis status must be a JSON object")
        current = value
    allowed_states = {
        "pending",
        "initializing",
        "running",
        "completed",
        "failed",
        "interrupted",
        "blocked",
        "ready",
    }
    for key in ("analysis_status", "m2a_status", "m2b_status", "export_status"):
        if key in updates and updates[key] not in allowed_states:
            raise ValueError(f"{key}: unsupported lifecycle state {updates[key]!r}")
    current.update(updates)
    current.setdefault("schema_version", 1)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, current)
    return current


def write_csv(path: str | Path, records: list[dict[str, Any]], columns: list[str]) -> None:
    """Atomically write a CSV with an explicit stable column order."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                if not finite_tree(record):
                    raise ValueError(
                        f"CSV record contains non-finite or unsupported value: {record}"
                    )
                writer.writerow(
                    {
                        column: (
                            ""
                            if record.get(column) is None
                            else json.dumps(
                                record.get(column),
                                ensure_ascii=False,
                                allow_nan=False,
                                separators=(",", ":"),
                            )
                            if isinstance(record.get(column), (list, dict))
                            else record.get(column)
                        )
                        for column in columns
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def finite_tree(value: Any) -> bool:
    """Return whether a JSON-shaped value contains only finite numeric leaves."""
    if value is None or isinstance(value, (str, bool)):
        return True
    if type(value) in {int, float}:
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and finite_tree(item) for key, item in value.items())
    return False
