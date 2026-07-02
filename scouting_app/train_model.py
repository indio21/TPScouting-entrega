"""Entrenamiento del modelo de prediccion de potencial."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import dump
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from db_utils import create_app_engine, ensure_player_columns, normalize_db_url
from models import Base
from player_logic import ATTRIBUTE_FIELDS, EVAL_MAX_AGE, EVAL_MIN_AGE
from preprocessing import (
    build_preprocessor,
    fit_transform_features,
    preprocessor_input_dim,
    save_preprocessor,
    TEMPORAL_TARGET_COLUMN,
    temporal_training_dataframe_from_engine,
    training_dataframe_from_engine,
    transform_features,
)

SEED = int(os.environ.get("SEED", "42"))
DEFAULT_THRESHOLDS = np.linspace(0.10, 0.90, 33)
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.17647058823529413  # 15% del total tras separar test
BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "256"))
DEFAULT_WEIGHT_DECAY = float(os.environ.get("TRAIN_WEIGHT_DECAY", "0.0005"))
DEFAULT_DROPOUT = float(os.environ.get("TRAIN_DROPOUT", "0.15"))
DEFAULT_LOSS_KIND = os.environ.get("TRAIN_LOSS_KIND", "bce").strip().lower()
DEFAULT_FOCAL_GAMMA = float(os.environ.get("TRAIN_FOCAL_GAMMA", "1.5"))
DEFAULT_SAMPLER_STRATEGY = os.environ.get("TRAIN_SAMPLER_STRATEGY", "shuffle").strip().lower()
DEFAULT_LINEAR_PRIOR_STRENGTH = float(os.environ.get("TRAIN_LINEAR_PRIOR_STRENGTH", "0.0001"))
DEFAULT_SPLITS_FILENAME = "training_splits.json"
MODEL_CHECKPOINT_VERSION = 1
logger = logging.getLogger(__name__)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class PlayerNet(nn.Module):
    def __init__(self, input_dim: int, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.wide = nn.Linear(input_dim, 1)
        self.base_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.base_bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.residual = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(max(0.05, dropout - 0.03)),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(max(0.05, dropout - 0.06)),
            nn.Linear(64, 1),
        )
        # Empezamos desde la solucion lineal y dejamos que la rama residual aprenda solo
        # donde haya interacciones no lineales utiles.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        base_logit = self.wide(x)
        residual_logit = self.residual(x)
        return self.base_scale * base_logit + self.base_bias + residual_logit

    def initialize_from_linear_model(self, linear_model: LogisticRegression) -> None:
        with torch.no_grad():
            coef = torch.tensor(linear_model.coef_, dtype=torch.float32)
            intercept = torch.tensor(linear_model.intercept_, dtype=torch.float32)
            self.wide.weight.copy_(coef)
            self.wide.bias.copy_(intercept.reshape(-1))


def model_input_dim(model: nn.Module) -> int:
    """Devuelve la dimension de entrada esperada por el modelo guardado."""
    wide_layer = getattr(model, "wide", None)
    input_dim = getattr(wide_layer, "in_features", None)
    if input_dim is None:
        raise ValueError("No se pudo inferir input_dim desde la capa wide del modelo.")
    return int(input_dim)


def model_checkpoint(model: nn.Module, input_dim: Optional[int] = None) -> Dict[str, object]:
    return {
        "version": MODEL_CHECKPOINT_VERSION,
        "model_class": model.__class__.__name__,
        "input_dim": int(input_dim if input_dim is not None else model_input_dim(model)),
        "seed": int(SEED),
        "model_state": model.state_dict(),
    }


def load_model_checkpoint(
    path: str,
    expected_input_dim: Optional[int] = None,
    map_location: object = "cpu",
) -> Dict[str, object]:
    """Carga checkpoints nuevos y state_dicts legacy con validacion opcional."""
    try:
        artifact = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        artifact = torch.load(path, map_location=map_location)

    if isinstance(artifact, dict) and "model_state" in artifact:
        state_dict = artifact["model_state"]
        stored_input_dim = artifact.get("input_dim")
        version = artifact.get("version", 0)
        seed = artifact.get("seed")
    else:
        state_dict = artifact
        stored_input_dim = None
        version = 0
        seed = None

    if stored_input_dim is not None and expected_input_dim is not None:
        if int(stored_input_dim) != int(expected_input_dim):
            raise RuntimeError(
                "Modelo incompatible con el preprocesador actual: "
                f"checkpoint input_dim={stored_input_dim}, esperado={expected_input_dim}."
            )

    return {
        "version": version,
        "input_dim": stored_input_dim,
        "seed": seed,
        "state_dict": state_dict,
        "is_legacy_state_dict": stored_input_dim is None,
    }


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, gamma: float = DEFAULT_FOCAL_GAMMA, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        probabilities = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
        focal_weight = torch.pow(torch.clamp(1.0 - pt, min=1e-6), self.gamma)
        return (focal_weight * bce_loss).mean()


def load_data(
    db_url: str,
    age_min: int = EVAL_MIN_AGE,
    age_max: int = EVAL_MAX_AGE,
    use_cache: bool = True,
    cache_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    """Carga jugadores y devuelve features, etiquetas y resumen del dataset."""
    normalized_db_url = normalize_db_url(db_url, base_dir=os.path.dirname(os.path.abspath(__file__)))
    engine = create_app_engine(normalized_db_url)
    Base.metadata.create_all(engine)
    ensure_player_columns(engine)
    raw_df = temporal_training_dataframe_from_engine(engine, use_cache=use_cache, cache_path=cache_path)
    if raw_df.empty:
        raise ValueError("No hay jugadores disponibles para entrenar el modelo.")

    filtered_df = raw_df[(raw_df["age"] >= age_min) & (raw_df["age"] <= age_max)].copy()
    if filtered_df.empty:
        raise ValueError(
            f"No hay jugadores disponibles en el rango etario {age_min}-{age_max} para entrenar el modelo."
        )

    y = filtered_df[TEMPORAL_TARGET_COLUMN].astype(np.float32).to_numpy()
    summary = dataset_summary(raw_df, filtered_df, TEMPORAL_TARGET_COLUMN)
    return filtered_df, y, summary


def dataset_summary(raw_df: pd.DataFrame, filtered_df: pd.DataFrame, target_column: str) -> Dict[str, object]:
    y_filtered = filtered_df[target_column].astype(int)
    positive_count = int(y_filtered.sum())
    total_count = int(len(filtered_df))
    negative_count = int(total_count - positive_count)
    summary = {
        "raw_rows": int(len(raw_df)),
        "filtered_rows": total_count,
        "age_range_raw": {
            "min": int(raw_df["age"].min()),
            "max": int(raw_df["age"].max()),
        },
        "age_range_filtered": {
            "min": int(filtered_df["age"].min()),
            "max": int(filtered_df["age"].max()),
        },
        "class_distribution": {
            "positive": positive_count,
            "negative": negative_count,
            "positive_rate": round(positive_count / total_count, 4) if total_count else 0.0,
        },
        "position_distribution": {
            str(key): int(value)
            for key, value in filtered_df["position"].value_counts().sort_index().items()
        },
        "target_column": target_column,
    }
    if "temporal_target_threshold" in filtered_df.columns and filtered_df["temporal_target_threshold"].notna().any():
        summary["temporal_target_threshold"] = float(filtered_df["temporal_target_threshold"].dropna().median())
    if "temporal_future_score_threshold" in filtered_df.columns and filtered_df["temporal_future_score_threshold"].notna().any():
        summary["temporal_future_score_threshold"] = float(filtered_df["temporal_future_score_threshold"].dropna().median())
    if "progression_score" in filtered_df.columns and filtered_df["progression_score"].notna().any():
        summary["progression_score_quantiles"] = {
            "q50": round(float(filtered_df["progression_score"].quantile(0.50)), 4),
            "q75": round(float(filtered_df["progression_score"].quantile(0.75)), 4),
            "q84": round(float(filtered_df["progression_score"].quantile(0.84)), 4),
            "q88": round(float(filtered_df["progression_score"].quantile(0.88)), 4),
            "q90": round(float(filtered_df["progression_score"].quantile(0.90)), 4),
            "q92": round(float(filtered_df["progression_score"].quantile(0.92)), 4),
        }
    if "temporal_consolidation_path" in filtered_df.columns:
        summary["temporal_consolidation_count"] = int(filtered_df["temporal_consolidation_path"].astype(int).sum())
    if "temporal_breakout_path" in filtered_df.columns:
        summary["temporal_breakout_count"] = int(filtered_df["temporal_breakout_path"].astype(int).sum())
    if "temporal_quality_gate" in filtered_df.columns:
        summary["temporal_quality_gate_count"] = int(filtered_df["temporal_quality_gate"].astype(int).sum())
    if "temporal_target_candidate" in filtered_df.columns:
        summary["temporal_target_candidate_count"] = int(filtered_df["temporal_target_candidate"].astype(int).sum())
    return summary


def choose_stratify_target(y: np.ndarray) -> Optional[np.ndarray]:
    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) <= 1 or class_counts.min() < 2:
        return None
    return y


def safe_train_test_split(*arrays, test_size: float, stratify: Optional[np.ndarray], random_state: int):
    try:
        return train_test_split(
            *arrays,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        if stratify is None:
            raise
        print(f"Advertencia: no se pudo estratificar el split: {exc}. Se continua sin estratificacion.")
        return train_test_split(
            *arrays,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


def build_split_artifact(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    return {
        "version": 1,
        "seed": int(SEED),
        "train_player_ids": train_df["player_id"].astype(int).tolist(),
        "validation_player_ids": val_df["player_id"].astype(int).tolist(),
        "test_player_ids": test_df["player_id"].astype(int).tolist(),
        "train_positive_rate": round(float(np.mean(y_train)), 4) if len(y_train) else 0.0,
        "validation_positive_rate": round(float(np.mean(y_val)), 4) if len(y_val) else 0.0,
        "test_positive_rate": round(float(np.mean(y_test)), 4) if len(y_test) else 0.0,
    }


def save_split_artifact(split_artifact: Dict[str, object], path: str) -> None:
    Path(path).write_text(json.dumps(split_artifact, indent=2, ensure_ascii=False), encoding="utf-8")


def load_split_artifact(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def default_splits_output_path(metadata_out: str) -> str:
    return str(Path(metadata_out).resolve().with_name(DEFAULT_SPLITS_FILENAME))


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    if len(y_true) == 0:
        return {
            "threshold": float(threshold),
            "accuracy": "",
            "roc_auc": "",
            "pr_auc": "",
            "f1": "",
            "precision": "",
            "recall": "",
            "confusion_matrix": [],
            "warnings": ["No se calcularon metricas: y_true esta vacio."],
        }

    y_pred = (y_prob >= threshold).astype(int)
    accuracy = float((y_pred == y_true).mean())
    metric_warnings: List[str] = []

    try:
        if len(np.unique(y_true)) < 2:
            raise ValueError("Se requieren al menos dos clases reales para ROC-AUC.")
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception as exc:
        roc_auc = ""
        metric_warnings.append(f"ROC-AUC no calculado: {exc}")
        logger.warning("ROC-AUC no calculado: %s", exc)

    try:
        if len(np.unique(y_true)) < 2:
            raise ValueError("Se requieren al menos dos clases reales para PR-AUC.")
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception as exc:
        pr_auc = ""
        metric_warnings.append(f"PR-AUC no calculado: {exc}")
        logger.warning("PR-AUC no calculado: %s", exc)

    try:
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    except Exception as exc:
        f1 = precision = recall = ""
        cm = []
        metric_warnings.append(f"F1/precision/recall no calculados: {exc}")
        logger.warning("F1/precision/recall no calculados: %s", exc)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "warnings": metric_warnings,
    }


def select_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, object]]:
    best_threshold = 0.5
    best_metrics = classification_metrics(y_true, y_prob, best_threshold)
    best_f1 = float(best_metrics["f1"] or 0.0)
    best_recall = float(best_metrics["recall"] or 0.0)

    for threshold in DEFAULT_THRESHOLDS:
        metrics = classification_metrics(y_true, y_prob, float(threshold))
        f1 = float(metrics["f1"] or 0.0)
        recall = float(metrics["recall"] or 0.0)
        if f1 > best_f1 or (f1 == best_f1 and recall > best_recall):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_f1 = f1
            best_recall = recall
    return best_threshold, best_metrics


def tensor_from_array(values: np.ndarray) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def seeded_torch_generator(offset: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(SEED) + int(offset))
    return generator


def sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def avg_skill_scores(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.asarray([], dtype=np.float32)
    return (df.loc[:, ATTRIBUTE_FIELDS].mean(axis=1).to_numpy(dtype=np.float32) / 20.0).reshape(-1)


def make_train_loader(
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = BATCH_SIZE,
    strategy: str = DEFAULT_SAMPLER_STRATEGY,
) -> DataLoader:
    dataset = TensorDataset(tensor_from_array(X_train), tensor_from_array(y_train).view(-1, 1))
    normalized_strategy = (strategy or "shuffle").strip().lower()
    if normalized_strategy == "weighted":
        sample_weights = np.where(
            y_train > 0.5,
            max(1.0, (len(y_train) - np.sum(y_train)) / max(np.sum(y_train), 1.0)),
            1.0,
        )
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(sample_weights),
            replacement=True,
            generator=seeded_torch_generator(offset=1),
        )
        return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), sampler=sampler)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=seeded_torch_generator(offset=2),
    )


def fit_probability_calibrator(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[Optional[object], str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    if len(np.unique(y_true)) < 2 or min(np.sum(y_true == 1), np.sum(y_true == 0)) < 10:
        fallback_threshold, _ = select_best_threshold(y_true, y_prob)
        return None, "none", float(fallback_threshold)

    candidates: list[tuple[str, Optional[object], np.ndarray]] = [("none", None, y_prob)]
    try:
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        isotonic.fit(y_prob, y_true)
        iso_prob = np.clip(np.asarray(isotonic.predict(y_prob), dtype=np.float32), 0.0, 1.0)
        candidates.append(("isotonic", isotonic, iso_prob))
    except Exception as exc:
        logger.warning("Calibrador isotonic omitido: %s", exc)

    try:
        platt = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        platt.fit(y_prob.reshape(-1, 1), y_true)
        platt_prob = platt.predict_proba(y_prob.reshape(-1, 1))[:, 1].astype(np.float32)
        candidates.append(("platt", platt, platt_prob))
    except Exception as exc:
        logger.warning("Calibrador Platt omitido: %s", exc)

    best_method = "none"
    best_calibrator = None
    best_prob = y_prob
    best_threshold, best_metrics = select_best_threshold(y_true, y_prob)
    best_score = (float(best_metrics["f1"] or 0.0), float(best_metrics["pr_auc"] or 0.0), -(float(brier_score_loss(y_true, y_prob))))

    for method, calibrator, calibrated_prob in candidates[1:]:
        threshold, metrics = select_best_threshold(y_true, calibrated_prob)
        score = (
            float(metrics["f1"] or 0.0),
            float(metrics["pr_auc"] or 0.0),
            -(float(brier_score_loss(y_true, calibrated_prob))),
        )
        if score > best_score:
            best_method = method
            best_calibrator = calibrator
            best_prob = calibrated_prob
            best_threshold = threshold
            best_metrics = metrics
            best_score = score

    return best_calibrator, best_method, float(best_threshold)


def apply_probability_calibrator(calibrator: Optional[object], y_prob: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.asarray(y_prob, dtype=np.float32).reshape(-1)
    try:
        if hasattr(calibrator, "predict_proba"):
            return calibrator.predict_proba(np.asarray(y_prob).reshape(-1, 1))[:, 1].astype(np.float32)
        return np.clip(np.asarray(calibrator.predict(y_prob), dtype=np.float32), 0.0, 1.0)
    except Exception as exc:
        logger.warning("No se pudo aplicar calibrador de probabilidad: %s", exc)
        return np.asarray(y_prob, dtype=np.float32).reshape(-1)


def train_model(
    features_df: pd.DataFrame,
    y: np.ndarray,
    epochs: int = 30,
    lr: float = 5e-4,
    patience: int = 8,
):
    """Entrena la red y devuelve modelo, preprocesador, calibrador, metadata y splits."""
    splits = safe_train_test_split(
        features_df,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=SEED,
        stratify=choose_stratify_target(y),
    )
    X_train_val_df, X_test_df, y_train_val, y_test = splits
    splits = safe_train_test_split(
        X_train_val_df,
        y_train_val,
        test_size=DEFAULT_VAL_SIZE,
        random_state=SEED,
        stratify=choose_stratify_target(y_train_val),
    )
    X_train_df, X_val_df, y_train, y_val = splits
    split_artifact = build_split_artifact(X_train_df, X_val_df, X_test_df, y_train, y_val, y_test)

    preprocessor = build_preprocessor()
    X_train, preprocessor = fit_transform_features(X_train_df, preprocessor)
    X_val = transform_features(X_val_df, preprocessor)
    X_test = transform_features(X_test_df, preprocessor)

    X_val_tensor = tensor_from_array(X_val)
    X_test_tensor = tensor_from_array(X_test)
    train_loader = make_train_loader(X_train, y_train)

    positive_count = float(np.sum(y_train))
    negative_count = float(len(y_train) - positive_count)
    pos_weight_value = (negative_count / positive_count) if positive_count > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)

    baseline_method = "logistic_regression_balanced"
    if len(np.unique(y_train)) < 2:
        baseline_model = None
        baseline_constant_prob = np.full(len(X_val), float(np.mean(y_train)), dtype=np.float32)
        baseline_threshold, baseline_val_metrics = select_best_threshold(y_val, baseline_constant_prob)
        baseline_test_prob = np.full(len(X_test), float(np.mean(y_train)), dtype=np.float32)
        baseline_test_metrics = classification_metrics(y_test, baseline_test_prob, baseline_threshold)
        baseline_method = "constant_single_class_fallback"
    else:
        baseline_model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        )
        baseline_model.fit(X_train, y_train)
        baseline_val_prob = baseline_model.predict_proba(X_val)[:, 1]
        baseline_threshold, baseline_val_metrics = select_best_threshold(y_val, baseline_val_prob)
        baseline_test_prob = baseline_model.predict_proba(X_test)[:, 1]
        baseline_test_metrics = classification_metrics(y_test, baseline_test_prob, baseline_threshold)

    input_dim = preprocessor_input_dim(preprocessor)
    model = PlayerNet(input_dim=input_dim)
    if baseline_model is not None:
        model.initialize_from_linear_model(baseline_model)
    if DEFAULT_LOSS_KIND == "focal":
        criterion = BinaryFocalLossWithLogits(gamma=DEFAULT_FOCAL_GAMMA, pos_weight=pos_weight)
        loss_name = f"BinaryFocalLossWithLogits(gamma={DEFAULT_FOCAL_GAMMA})"
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_name = "BCEWithLogitsLoss"
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    wide_reference_weight = model.wide.weight.detach().clone()
    wide_reference_bias = model.wide.bias.detach().clone()

    best_state = None
    best_threshold = 0.5
    best_calibrator = None
    best_calibration_method = "none"
    best_epoch = 1
    best_monitor = (-1.0, -1.0)
    best_val_metrics: Dict[str, object] = classification_metrics(y_val, np.zeros(len(y_val)), 0.5)
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            prior_penalty = DEFAULT_LINEAR_PRIOR_STRENGTH * (
                torch.mean((model.wide.weight - wide_reference_weight) ** 2)
                + torch.mean((model.wide.bias - wide_reference_bias) ** 2)
                + torch.mean((model.base_scale - 1.0) ** 2)
                + torch.mean(model.base_bias**2)
            )
            (loss + prior_penalty).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor).cpu().numpy().reshape(-1)
        raw_val_prob = sigmoid_numpy(val_logits)
        calibrator, calibration_method, threshold = fit_probability_calibrator(y_val, raw_val_prob)
        val_prob = apply_probability_calibrator(calibrator, raw_val_prob)
        val_metrics = classification_metrics(y_val, val_prob, threshold)
        monitor_pr_auc = float(val_metrics["pr_auc"] or 0.0)
        monitor_f1 = float(val_metrics["f1"] or 0.0)
        scheduler.step(monitor_pr_auc)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(batch_losses) if batch_losses else 0.0),
                "val_pr_auc": monitor_pr_auc,
                "val_f1": monitor_f1,
                "threshold": float(threshold),
                "calibration_method": calibration_method,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        is_better = (monitor_pr_auc, monitor_f1) > best_monitor
        if is_better:
            best_monitor = (monitor_pr_auc, monitor_f1)
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = float(threshold)
            best_calibrator = calibrator
            best_calibration_method = calibration_method
            best_epoch = epoch + 1
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoca {epoch + 1}/{epochs}, perdida: {np.mean(batch_losses) if batch_losses else 0.0:.4f}, "
                f"val_pr_auc: {monitor_pr_auc:.4f}, val_f1: {monitor_f1:.4f}, threshold: {threshold:.2f}, "
                f"calibracion: {calibration_method}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping activado en epoca {epoch + 1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_logits = model(X_val_tensor).cpu().numpy().reshape(-1)
    raw_val_prob = sigmoid_numpy(val_logits)
    raw_val_threshold, raw_val_metrics = select_best_threshold(y_val, raw_val_prob)
    val_prob = apply_probability_calibrator(best_calibrator, raw_val_prob)
    calibrated_val_metrics = classification_metrics(y_val, val_prob, best_threshold)

    with torch.no_grad():
        test_logits = model(X_test_tensor).cpu().numpy().reshape(-1)
    raw_test_prob = sigmoid_numpy(test_logits)
    raw_test_metrics = classification_metrics(y_test, raw_test_prob, raw_val_threshold)
    test_prob = apply_probability_calibrator(best_calibrator, raw_test_prob)
    test_metrics = classification_metrics(y_test, test_prob, best_threshold)

    avg_skill_val_prob = avg_skill_scores(X_val_df)
    avg_skill_threshold, avg_skill_val_metrics = select_best_threshold(y_val, avg_skill_val_prob)
    avg_skill_test_prob = avg_skill_scores(X_test_df)
    avg_skill_test_metrics = classification_metrics(y_test, avg_skill_test_prob, avg_skill_threshold)

    print(f"Precision en test (PyTorch): {float(test_metrics['accuracy']) * 100:.2f}%")
    if test_metrics["roc_auc"] != "":
        print(f"ROC-AUC (PyTorch): {float(test_metrics['roc_auc']):.4f}")
    if test_metrics["pr_auc"] != "":
        print(f"PR-AUC (PyTorch): {float(test_metrics['pr_auc']):.4f}")
    if test_metrics["f1"] != "":
        print(
            "F1 (PyTorch): "
            f"{float(test_metrics['f1']):.4f} | "
            f"Precision: {float(test_metrics['precision']):.4f} | "
            f"Recall: {float(test_metrics['recall']):.4f}"
        )
    print("Confusion matrix (PyTorch):", test_metrics["confusion_matrix"])
    print(
        "Baseline LogisticRegression balanced | "
        f"ROC-AUC: {baseline_test_metrics['roc_auc']} | "
        f"PR-AUC: {baseline_test_metrics['pr_auc']} | "
        f"F1: {baseline_test_metrics['f1']}"
    )
    print(
        "Baseline promedio atributos | "
        f"ROC-AUC: {avg_skill_test_metrics['roc_auc']} | "
        f"PR-AUC: {avg_skill_test_metrics['pr_auc']} | "
        f"F1: {avg_skill_test_metrics['f1']}"
    )

    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "seed": SEED,
        "config": {
            "epochs_requested": int(epochs),
            "epochs_trained": int(history[-1]["epoch"]) if history else 0,
            "learning_rate": float(lr),
            "patience": int(patience),
            "batch_size": int(min(BATCH_SIZE, len(X_train_df))) if len(X_train_df) else 0,
            "optimizer": "AdamW",
            "weight_decay": float(DEFAULT_WEIGHT_DECAY),
            "dropout": float(DEFAULT_DROPOUT),
            "age_min": int(EVAL_MIN_AGE),
            "age_max": int(EVAL_MAX_AGE),
            "loss": loss_name,
            "target_type": "temporal_progression",
            "pos_weight": float(pos_weight_value),
            "pos_weight_strategy": "full_ratio",
            "sampler_strategy": DEFAULT_SAMPLER_STRATEGY,
            "linear_bootstrap": bool(baseline_model is not None),
            "linear_prior_strength": float(DEFAULT_LINEAR_PRIOR_STRENGTH),
            "test_size": float(DEFAULT_TEST_SIZE),
            "validation_size_on_train_pool": float(DEFAULT_VAL_SIZE),
        },
        "model": {
            "checkpoint_version": MODEL_CHECKPOINT_VERSION,
            "input_dim": int(input_dim),
            "class": "PlayerNet",
        },
        "dataset": {
            "train_size": int(len(X_train_df)),
            "validation_size": int(len(X_val_df)),
            "test_size": int(len(X_test_df)),
            "train_positive_rate": round(float(np.mean(y_train)), 4),
            "validation_positive_rate": round(float(np.mean(y_val)), 4),
            "test_positive_rate": round(float(np.mean(y_test)), 4),
        },
        "pytorch": {
            "best_epoch": int(best_epoch),
            "selected_threshold": float(best_threshold),
            "calibration_method": best_calibration_method,
            "validation": calibrated_val_metrics,
            "test": test_metrics,
            "raw_validation": raw_val_metrics,
            "raw_validation_threshold": float(raw_val_threshold),
            "raw_test": raw_test_metrics,
            "history": history,
        },
        "calibration": {
            "method": best_calibration_method,
            "brier_validation": float(brier_score_loss(y_val, val_prob)) if len(y_val) else "",
            "brier_test": float(brier_score_loss(y_test, test_prob)) if len(y_test) else "",
        },
        "scoring_policy": {
            "primary_model_probability": "raw_pytorch_sigmoid",
            "primary_app_score": "combined_score_from_raw_pytorch_plus_history_and_position_fit",
            "secondary_probability": "calibrated_pytorch_probability",
            "raw_validation_threshold": float(raw_val_threshold),
            "calibrated_validation_threshold": float(best_threshold),
            "reason": (
                "La salida cruda queda como score principal porque en la evidencia actual "
                "prioriza mejor candidatos por PR-AUC; la calibrada se conserva como referencia secundaria."
            ),
        },
        "splits": {
            "version": int(split_artifact["version"]),
            "train_count": int(len(split_artifact["train_player_ids"])),
            "validation_count": int(len(split_artifact["validation_player_ids"])),
            "test_count": int(len(split_artifact["test_player_ids"])),
        },
        "baselines": {
            "logistic_regression_balanced": {
                "method": baseline_method,
                "selected_threshold": float(baseline_threshold),
                "validation": baseline_val_metrics,
                "test": baseline_test_metrics,
            },
            "avg_skill_score": {
                "selected_threshold": float(avg_skill_threshold),
                "validation": avg_skill_val_metrics,
                "test": avg_skill_test_metrics,
            },
        },
    }
    return model, preprocessor, best_calibrator, metadata, split_artifact


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model_checkpoint(model), path)


def save_calibrator(calibrator: Optional[object], path: str) -> None:
    if calibrator is None:
        return
    dump(calibrator, path)


def save_metadata(metadata: Dict[str, object], path: str) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main(
    db_url: str,
    model_out: str,
    preprocessor_out: str,
    calibrator_out: str,
    metadata_out: str,
    epochs: int,
    lr: float,
    patience: int = 8,
    splits_out: Optional[str] = None,
) -> None:
    feature_df, y, data_summary = load_data(db_url)
    model, preprocessor, calibrator, metadata, split_artifact = train_model(
        feature_df,
        y,
        epochs=epochs,
        lr=lr,
        patience=patience,
    )
    metadata["dataset_summary"] = data_summary
    resolved_splits_out = splits_out or default_splits_output_path(metadata_out)
    metadata["artifacts"] = {
        "model_path": str(Path(model_out).resolve()),
        "preprocessor_path": str(Path(preprocessor_out).resolve()),
        "calibrator_path": str(Path(calibrator_out).resolve()),
        "metadata_path": str(Path(metadata_out).resolve()),
        "splits_path": str(Path(resolved_splits_out).resolve()),
    }
    save_model(model, model_out)
    save_preprocessor(preprocessor, preprocessor_out)
    save_calibrator(calibrator, calibrator_out)
    save_split_artifact(split_artifact, resolved_splits_out)
    save_metadata(metadata, metadata_out)
    print(f"Modelo guardado en {model_out}")
    print(f"Preprocesador guardado en {preprocessor_out}")
    if calibrator is not None:
        print(f"Calibrador guardado en {calibrator_out}")
    print(f"Splits guardados en {resolved_splits_out}")
    print(f"Metadata guardada en {metadata_out}")

    try:
        row = {
            "timestamp": metadata["timestamp"],
            "seed": SEED,
            "epochs": int(metadata["config"]["epochs_trained"]),
            "lr": float(lr),
            "model_path": model_out,
            "preprocessor_path": preprocessor_out,
            "calibration_method": metadata["calibration"]["method"],
            "accuracy": metadata["pytorch"]["test"]["accuracy"],
            "roc_auc": metadata["pytorch"]["test"]["roc_auc"],
            "pr_auc": metadata["pytorch"]["test"]["pr_auc"],
            "f1": metadata["pytorch"]["test"]["f1"],
            "precision": metadata["pytorch"]["test"]["precision"],
            "recall": metadata["pytorch"]["test"]["recall"],
            "train_size": metadata["dataset"]["train_size"],
            "test_size": metadata["dataset"]["test_size"],
            "target_positive_rate": data_summary["class_distribution"]["positive_rate"],
        }
        log_experiment(row)
    except Exception as exc:
        print(f"Advertencia: no se pudo registrar la corrida del experimento: {exc}")


def log_experiment(row: dict) -> None:
    exp_path = os.path.join(os.path.dirname(__file__), "experiments.csv")
    expected_fields = [
        "timestamp",
        "seed",
        "epochs",
        "lr",
        "model_path",
        "preprocessor_path",
        "calibration_method",
        "accuracy",
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "train_size",
        "test_size",
        "target_positive_rate",
    ]
    write_header = not os.path.exists(exp_path)
    with open(exp_path, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=expected_fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in expected_fields})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo de scouting")
    parser.add_argument(
        "--db-url",
        type=str,
        default="sqlite:///players_training.db",
        help="URL de la base de datos de entrenamiento",
    )
    parser.add_argument("--model-out", type=str, default="model.pt", help="Ruta de salida del modelo")
    parser.add_argument(
        "--preprocessor-out",
        type=str,
        default="preprocessor.joblib",
        help="Ruta de salida del preprocesador",
    )
    parser.add_argument(
        "--calibrator-out",
        type=str,
        default="probability_calibrator.joblib",
        help="Ruta de salida del calibrador de probabilidades",
    )
    parser.add_argument(
        "--metadata-out",
        type=str,
        default="training_metadata.json",
        help="Ruta de salida de metadata del entrenamiento",
    )
    parser.add_argument(
        "--splits-out",
        type=str,
        default=None,
        help="Ruta de salida de splits de entrenamiento/evaluacion",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Numero de epocas")
    parser.add_argument("--lr", type=float, default=5e-4, help="Tasa de aprendizaje")
    parser.add_argument("--patience", type=int, default=8, help="Paciencia para early stopping")
    args = parser.parse_args()
    main(
        args.db_url,
        args.model_out,
        args.preprocessor_out,
        args.calibrator_out,
        args.metadata_out,
        args.epochs,
        args.lr,
        args.patience,
        args.splits_out,
    )
