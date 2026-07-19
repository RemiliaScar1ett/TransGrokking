"""Stable command-line entry points for M0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transgrokking.config import load_config
from transgrokking.data import generate_modular_addition, split_artifact
from transgrokking.training import train
from transgrokking.utils.atomic import torch_save
from transgrokking.utils.doctor import collect_doctor_report, validate_doctor_report


def _doctor(args: argparse.Namespace) -> int:
    report = collect_doctor_report()
    errors = validate_doctor_report(
        report,
        require_cuda=args.require_cuda,
        expected_device=args.expected_device,
        expected_vram_gb=args.expected_vram_gb,
    )
    print(json.dumps({**report.to_dict(), "errors": errors}, indent=2, ensure_ascii=False))
    return 1 if errors else 0


def _generate_data(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    data = generate_modular_addition(
        config.task.modulus, config.task.train_fraction, config.task.split_seed
    )
    digest = hashlib.sha256(data.split_hash.encode()).hexdigest()[:8]
    output = Path(config.logging.runs_dir) / f"data_{digest}" / "split.pt"
    torch_save(output, split_artifact(data, config.task.modulus, config.task.split_seed))
    print(output)
    return 0


def _train(args: argparse.Namespace) -> int:
    run_dir = train(load_config(args.config), resume_from=args.resume_from)
    print(run_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the M0 command parser."""
    parser = argparse.ArgumentParser(prog="transgrokking")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--require-cuda", action="store_true")
    doctor.add_argument("--expected-device")
    doctor.add_argument("--expected-vram-gb", type=float)
    doctor.set_defaults(handler=_doctor)
    generate = subparsers.add_parser("generate-data")
    generate.add_argument("--config", required=True)
    generate.set_defaults(handler=_generate_data)
    training = subparsers.add_parser("train")
    training.add_argument("--config", required=True)
    training.add_argument("--resume-from")
    training.set_defaults(handler=_train)
    return parser


def main() -> int:
    """Execute the selected command."""
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
