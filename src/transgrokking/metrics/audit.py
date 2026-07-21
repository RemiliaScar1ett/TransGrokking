"""Machine-readable M1 CE-reference run audit."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from transgrokking.config import load_config
from transgrokking.metrics.evaluator import evaluate_run_checkpoint
from transgrokking.training.artifacts import (
    load_error_offset_records,
    load_manifest,
    load_scalar_records,
)
from transgrokking.training.checkpoint import read_checkpoint
from transgrokking.utils.atomic import write_json


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lineage(root: Path) -> list[Path]:
    chain = [root]
    seen = {root.resolve()}
    current = root
    while True:
        metadata = _json(current / "metadata.json")
        parent_id = metadata.get("parent_run_id")
        if parent_id is None:
            return list(reversed(chain))
        parent = (current.parent / str(parent_id)).resolve()
        if parent in seen or not parent.is_dir():
            raise ValueError(f"invalid parent run lineage at {current}")
        chain.append(parent)
        seen.add(parent)
        current = parent


def _close(actual: object, expected: object) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if type(actual) in {int, float} and type(expected) in {int, float}:
        return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-7)
    return actual == expected


def audit_m1_ce_reference(run_dir: str | Path) -> dict[str, Any]:
    """Audit a terminal M1 run and persist a derived JSON report."""
    root = Path(run_dir).resolve()
    config = load_config(root / "config.resolved.yaml")
    status = _json(root / "status.json")
    metadata = _json(root / "metadata.json")
    events = _json(root / "metrics" / "events.json")
    scalars = load_scalar_records(root / "metrics" / "scalars.jsonl")
    offsets = load_error_offset_records(root / "metrics" / "error_offsets.jsonl")
    manifest = load_manifest(root)
    lineage = _lineage(root)
    lineage_metadata = [_json(path / "metadata.json") for path in lineage]
    latest = manifest[-1]
    latest_path = root / "checkpoints" / str(latest["path"])
    checkpoint = read_checkpoint(latest_path)
    evaluation = evaluate_run_checkpoint(root)
    scalar = scalars[-1]
    comparable = {key: value for key, value in evaluation.items() if key in scalar}
    metric_match = all(_close(scalar[key], value) for key, value in comparable.items())
    offset_steps = [int(offsets[index]["step"]) for index in range(0, len(offsets), 2)]
    scalar_steps = [int(record["step"]) for record in scalars]
    final_step = int(status.get("global_step", -1))
    grok99 = events["t_grok99"]
    reached = grok99.get("status") == "reached"
    stop_rule = (
        final_step >= int(grok99["event_step"]) + 20 * config.logging.eval_interval
        if reached
        else final_step == 50000
    )
    groups = metadata.get("optimizer_parameter_groups", [])
    group_policy_ok = [group.get("group_name") for group in groups] == ["decay", "no_decay"] and [
        group.get("weight_decay") for group in groups
    ] == [config.optimization.weight_decay, 0.0]
    lineage_hashes = {item.get("scientific_config_hash") for item in lineage_metadata}
    split_hashes = {item.get("split_hash") for item in lineage_metadata}
    parent_links_ok = all(
        child.get("parent_run_id") == parent.name
        and child.get("parent_checkpoint")
        and type(child.get("parent_global_step")) is int
        for parent, child in zip(lineage, lineage_metadata[1:], strict=False)
    )
    forbidden_files = [
        str(path.relative_to(root))
        for directory in (root / "tensors", root / "figures")
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file()
    ]
    checks = {
        "completed": status.get("state") == "completed",
        "formal_run": metadata.get("formal_run") is True and config.hardware.formal_run,
        "target_device": metadata.get("doctor", {}).get("device_name")
        == "NVIDIA GeForce RTX 4060 Laptop GPU",
        "fp32_policy": config.optimization.precision == "fp32"
        and not config.optimization.allow_tf32
        and not config.optimization.use_amp,
        "ce_only": config.loss.cross_entropy_weight == 1.0 and config.loss.congruence_weight == 0.0,
        "lineage_scientific_hash": len(lineage_hashes) == 1
        and metadata.get("scientific_config_hash") == config.scientific_hash(),
        "lineage_split_hash": len(split_hashes) == 1
        and checkpoint.get("split_hash") == metadata.get("split_hash"),
        "parent_links": parent_links_ok,
        "scalar_timeline": bool(scalar_steps)
        and scalar_steps == sorted(set(scalar_steps))
        and scalar_steps[-1] == final_step,
        "offset_timeline": offset_steps == scalar_steps,
        "events_timeline": events.get("last_evaluated_step") == final_step,
        "manifest": int(latest["step"]) == final_step,
        "checkpoint_interval": config.logging.checkpoint_interval == 100,
        "optimizer_groups": group_policy_ok,
        "peak_vram": int(metadata.get("max_memory_allocated", 0)) > 0
        and int(metadata.get("max_memory_reserved", 0)) > 0,
        "offline_evaluator": evaluation.get("step") == final_step and metric_match,
        "stop_rule": stop_rule,
        "no_m2_plus_artifacts": not forbidden_files,
    }
    report = {
        "schema_version": 1,
        "run_id": root.name,
        "canonical_run": str(root),
        "lineage": [path.name for path in lineage],
        "scientific_config_hash": metadata.get("scientific_config_hash"),
        "split_hash": metadata.get("split_hash"),
        "final_global_step": final_step,
        "t_fit": events.get("t_fit"),
        "t_grok50": events.get("t_grok50"),
        "t_grok99": grok99,
        "stop_reason": "t_grok99_plus_20_evaluations" if reached else "max_steps_50000",
        "checks": checks,
        "passed": all(checks.values()),
        "forbidden_files": forbidden_files,
    }
    write_json(root / "audit" / "m1_ce_reference.json", report)
    return report
