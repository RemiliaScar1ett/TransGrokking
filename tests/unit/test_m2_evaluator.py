"""Fast CPU tests for M2 full-table evaluation helpers."""

from __future__ import annotations

import torch

from transgrokking.analysis.evaluator import behavior_snapshot, full_table_logits
from transgrokking.analysis.runner import _function_behavior_alignment
from transgrokking.config import config_from_dict
from transgrokking.data import generate_modular_addition
from transgrokking.training.optimizer import build_adamw
from transgrokking.training.trainer import build_model


def _config():
    return config_from_dict(
        {
            "task": {"modulus": 5, "train_fraction": 0.4, "split_seed": 2},
            "model": {
                "d_model": 8,
                "n_heads": 2,
                "n_layers": 1,
                "d_mlp": 16,
                "dropout": 0.0,
                "activation": "relu",
                "norm_first": True,
                "final_norm": False,
            },
            "optimization": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.5,
                "decay_policy": {
                    "matrix_weights": True,
                    "embeddings": True,
                    "biases": False,
                    "layer_norm": False,
                },
                "max_steps": 1,
                "precision": "fp32",
                "allow_tf32": False,
                "use_amp": False,
                "deterministic": True,
                "seed": 1,
                "device": "cpu",
            },
            "hardware": {
                "expected_device": "CPU",
                "expected_vram_gb": 1,
                "formal_run": False,
                "analysis_batch_size": 8,
                "activation_offload": True,
            },
            "loss": {"cross_entropy_weight": 1.0, "congruence_weight": 0.0},
            "logging": {
                "eval_interval": 1,
                "checkpoint_interval": 1,
                "activation_steps": [],
                "runs_dir": "runs",
            },
            "events": {
                "fit_accuracy": 0.999,
                "fit_consecutive": 5,
                "grok50_consecutive": 3,
                "grok99_accuracy": 0.99,
                "grok99_consecutive": 3,
            },
        }
    )


def test_full_table_logits_preserve_a_major_order_and_training_mode() -> None:
    config = _config()
    data = generate_modular_addition(5, 0.4, 2)
    model = build_model(config)
    model.train()

    logits = full_table_logits(model, data.inputs, 5, batch_size=7)

    assert logits.shape == (5, 5, 5)
    assert logits.dtype == torch.float32
    assert model.training is True


def test_behavior_snapshot_includes_full_table_fields() -> None:
    config = _config()
    data = generate_modular_addition(5, 0.4, 2)
    model = build_model(config)
    _, grouping = build_adamw(model, config.optimization)
    logits = full_table_logits(model, data.inputs, 5, batch_size=8)

    scalars, offsets = behavior_snapshot(
        model, data, grouping, torch.device("cpu"), full_logits=logits
    )

    assert "full_cross_entropy" in scalars
    assert "full_margin_min" in scalars
    assert sum(offsets["full"]) == scalars["full_error_count"]


def test_function_alignment_uses_error_counts_for_exact_predictions() -> None:
    config = type(
        "ToleranceConfig",
        (),
        {"behavior_validation_atol": 1.0e-6, "behavior_validation_rtol": 1.0e-6},
    )()
    metrics = {
        "train_cross_entropy": 0.1 + 5.0e-8,
        "test_cross_entropy": 0.2 - 5.0e-8,
        "train_accuracy": 1.0,
        "test_accuracy": 68.0 / 71.0,
        "train_error_count": 0,
        "test_error_count": 3,
    }
    committed = {
        "train_cross_entropy": 0.1,
        "test_cross_entropy": 0.2,
        "train_accuracy": 1.0,
        "test_accuracy": float(torch.tensor(68.0 / 71.0, dtype=torch.float32)),
        "train_error_count": 0,
        "test_error_count": 3,
    }

    alignment = _function_behavior_alignment(100, metrics, {100: committed}, config)

    assert alignment["committed_ce_within_tolerance"] is True
    assert alignment["batched_predictions_match_committed"] is True
    assert alignment["committed_ce_max_abs_diff"] <= 1.0e-6


def test_function_alignment_exposes_batch_sensitive_prediction_count() -> None:
    config = type(
        "ToleranceConfig",
        (),
        {"behavior_validation_atol": 1.0e-6, "behavior_validation_rtol": 1.0e-6},
    )()
    metrics = {
        "train_cross_entropy": 0.1,
        "test_cross_entropy": 0.2,
        "train_accuracy": 1.0,
        "test_accuracy": 0.5,
        "train_error_count": 0,
        "test_error_count": 5,
    }
    committed = {
        "train_cross_entropy": 0.1,
        "test_cross_entropy": 0.2,
        "train_accuracy": 1.0,
        "test_accuracy": 0.4,
        "train_error_count": 0,
        "test_error_count": 6,
    }

    alignment = _function_behavior_alignment(100, metrics, {100: committed}, config)

    assert alignment["committed_ce_within_tolerance"] is True
    assert alignment["batched_predictions_match_committed"] is False
    assert alignment["committed_test_error_count_diff"] == -1
