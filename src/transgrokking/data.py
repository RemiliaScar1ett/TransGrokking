"""Deterministic modular-addition data generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModularAdditionData:
    """Full table and split tensors, all stored on CPU."""

    inputs: torch.Tensor
    labels: torch.Tensor
    train_indices: torch.Tensor
    test_indices: torch.Tensor
    split_hash: str


def _split_digest(modulus: int, train_indices: torch.Tensor, test_indices: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(f"transgrokking-split-v1:{modulus}:".encode())
    digest.update(train_indices.contiguous().numpy().tobytes())
    digest.update(test_indices.contiguous().numpy().tobytes())
    return digest.hexdigest()


def generate_modular_addition(
    modulus: int, train_fraction: float, split_seed: int
) -> ModularAdditionData:
    """Generate `[p*p, 2]` inputs, labels, and a deterministic disjoint split."""
    if modulus < 2:
        raise ValueError("modulus must be >= 2")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    values = torch.arange(modulus, dtype=torch.long)
    inputs = torch.cartesian_prod(values, values)
    labels = (inputs[:, 0] + inputs[:, 1]).remainder(modulus)
    generator = torch.Generator(device="cpu").manual_seed(split_seed)
    permutation = torch.randperm(modulus * modulus, generator=generator)
    train_size = int(modulus * modulus * train_fraction)
    train_indices = permutation[:train_size].clone()
    test_indices = permutation[train_size:].clone()
    split_hash = _split_digest(modulus, train_indices, test_indices)
    return ModularAdditionData(inputs, labels, train_indices, test_indices, split_hash)


def split_artifact(data: ModularAdditionData, modulus: int, split_seed: int) -> dict[str, object]:
    """Return the serializable split artifact."""
    return {
        "schema_version": 1,
        "modulus": modulus,
        "split_seed": split_seed,
        "split_hash": data.split_hash,
        "train_indices": data.train_indices,
        "test_indices": data.test_indices,
    }
