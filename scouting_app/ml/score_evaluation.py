"""Evaluación independiente y pareada de probabilidades y score combinado."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from ml.scoring import combined_probability
from player_logic import ATTRIBUTE_FIELDS, weighted_score_from_attrs
from train_model import classification_metrics, select_best_threshold


def combined_scores(df: pd.DataFrame, probabilities: np.ndarray) -> np.ndarray:
    if len(df) != len(probabilities):
        raise ValueError("Las probabilidades no están alineadas con las filas.")
    values = []
    for (_, row), probability in zip(df.iterrows(), probabilities):
        attrs = {field: float(row[field]) for field in ATTRIBUTE_FIELDS}
        fit_score = weighted_score_from_attrs(attrs, str(row["position"]))
        rating = row.get("avg_final_score_hist")
        average_final_score: Optional[float] = None if pd.isna(rating) else float(rating)
        values.append(
            combined_probability(
                float(probability),
                average_final_score=average_final_score,
                fit_score=fit_score,
            )
        )
    return np.asarray(values, dtype=np.float32)


def probability_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, object]:
    result = classification_metrics(y_true, y_prob, threshold)
    clipped = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    result["brier_score"] = float(brier_score_loss(y_true, clipped))
    result["log_loss"] = float(log_loss(y_true, clipped, labels=[0, 1]))
    return result


def evaluate_probability_variants(
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    validation_probabilities: Mapping[str, np.ndarray],
    test_probabilities: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    if set(validation_probabilities) != set(test_probabilities):
        raise ValueError("Las variantes de validation y test no coinciden.")
    y_validation = validation_df["temporal_target_label"].astype(int).to_numpy()
    y_test = test_df["temporal_target_label"].astype(int).to_numpy()
    variants: Dict[str, object] = {}
    for name in validation_probabilities:
        val_prob = np.asarray(validation_probabilities[name], dtype=np.float32)
        test_prob = np.asarray(test_probabilities[name], dtype=np.float32)
        if len(val_prob) != len(validation_df) or len(test_prob) != len(test_df):
            raise ValueError(f"La variante {name} no coincide con los splits.")
        threshold, _ = select_best_threshold(y_validation, val_prob)
        variants[name] = {
            "threshold_selected_on_validation": float(threshold),
            "validation": probability_metrics(y_validation, val_prob, threshold),
            "test": probability_metrics(y_test, test_prob, threshold),
            "visual_high_band_0_80_test": probability_metrics(y_test, test_prob, 0.80),
        }
    return variants


def paired_bootstrap_deltas(
    y_true: np.ndarray,
    reference_probability: np.ndarray,
    candidate_probability: np.ndarray,
    iterations: int = 2000,
    seed: int = 42,
) -> Dict[str, object]:
    """IC percentiles candidato menos referencia sobre las mismas filas de test."""
    y_true = np.asarray(y_true, dtype=int)
    reference_probability = np.asarray(reference_probability, dtype=float)
    candidate_probability = np.asarray(candidate_probability, dtype=float)
    if not (len(y_true) == len(reference_probability) == len(candidate_probability)):
        raise ValueError("Bootstrap requiere vectores pareados de igual longitud.")
    rng = np.random.default_rng(seed)
    deltas = {"roc_auc": [], "pr_auc": [], "brier_score": []}
    skipped = 0
    for _ in range(iterations):
        index = rng.integers(0, len(y_true), size=len(y_true))
        labels = y_true[index]
        if len(np.unique(labels)) < 2:
            skipped += 1
            continue
        ref = reference_probability[index]
        candidate = candidate_probability[index]
        deltas["roc_auc"].append(roc_auc_score(labels, candidate) - roc_auc_score(labels, ref))
        deltas["pr_auc"].append(
            average_precision_score(labels, candidate) - average_precision_score(labels, ref)
        )
        deltas["brier_score"].append(
            brier_score_loss(labels, candidate) - brier_score_loss(labels, ref)
        )

    intervals: Dict[str, object] = {}
    for metric, values in deltas.items():
        array = np.asarray(values, dtype=float)
        intervals[metric] = {
            "mean_delta": float(array.mean()),
            "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
            "direction": "higher_is_better" if metric != "brier_score" else "lower_is_better",
        }
    return {
        "comparison": "candidate_minus_reference",
        "iterations_requested": int(iterations),
        "iterations_used": int(iterations - skipped),
        "skipped_single_class_samples": int(skipped),
        "metrics": intervals,
    }
