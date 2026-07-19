"""Atomic local artifact writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml


def _replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, value: Any) -> None:
    """Atomically write indented UTF-8 JSON."""
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    _replace_bytes(Path(path), payload.encode())


def write_json_lines(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Atomically replace a UTF-8 JSONL file with finite, parseable records."""
    payload = "".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        for record in records
    )
    _replace_bytes(Path(path), payload.encode())


def write_yaml(path: str | Path, value: Any) -> None:
    """Atomically write UTF-8 YAML."""
    _replace_bytes(Path(path), yaml.safe_dump(value, sort_keys=False).encode())


def torch_save(path: str | Path, value: Any) -> None:
    """Atomically serialize a PyTorch artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
