"""Evaluacion rapida de artefactos entrenados usando cache y splits persistidos."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from joblib import load
from sklearn.linear_model import LogisticRegression

from preprocessing import (
    load_preprocessor,
    preprocessor_input_dim,
    transform_features,
)
from train_model import (
    DEFAULT_DROPOUT,
    PlayerNet,
    apply_probability_calibrator,
    classification_metrics,
    load_data,
    load_model_checkpoint,
    load_split_artifact,
    select_best_threshold,
    sigmoid_numpy,
)


def _resolve_artifact_path(cli_value: Optional[str], metadata: Dict[str, object], artifact_key: str, fallback: str) -> str:
    if cli_value:
        return cli_value
    artifacts = metadata.get("artifacts", {}) if isinstance(metadata, dict) else {}
    if isinstance(artifacts, dict) and artifacts.get(artifact_key):
        return str(artifacts[artifact_key])
    return fallback


def _select_split_dataframe(features_df: pd.DataFrame, player_ids: Sequence[int]) -> pd.DataFrame:
    indexed_df = features_df.set_index("player_id", drop=False)
    ordered_ids = [int(player_id) for player_id in player_ids if int(player_id) in indexed_df.index]
    missing_ids = len(player_ids) - len(ordered_ids)
    if missing_ids:
        raise ValueError(
            "Los splits persistidos no coinciden con el dataframe temporal actual: "
            f"faltan {missing_ids} player_id(s)."
        )
    return indexed_df.loc[ordered_ids].reset_index(drop=True)


def evaluate_saved_model(
    db_url: str,
    model_path: str,
    preprocessor_path: str,
    splits_path: str,
    metadata_path: Optional[str] = None,
    calibrator_path: Optional[str] = None,
    cache_path: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, object]:
    metadata = {}
    if metadata_path and Path(metadata_path).exists():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

    load_started_at = time.perf_counter()
    features_df, y, data_summary = load_data(
        db_url,
        use_cache=use_cache,
        cache_path=cache_path,
    )
    load_elapsed = time.perf_counter() - load_started_at

    split_artifact = load_split_artifact(splits_path)
    train_df = _select_split_dataframe(features_df, split_artifact["train_player_ids"])
    validation_df = _select_split_dataframe(features_df, split_artifact["validation_player_ids"])
    test_df = _select_split_dataframe(features_df, split_artifact["test_player_ids"])

    y_train = train_df["temporal_target_label"].astype(np.float32).to_numpy()
    y_val = validation_df["temporal_target_label"].astype(np.float32).to_numpy()
    y_test = test_df["temporal_target_label"].astype(np.float32).to_numpy()

    preprocessor = load_preprocessor(preprocessor_path)
    X_train = transform_features(train_df, preprocessor)
    X_val = transform_features(validation_df, preprocessor)
    X_test = transform_features(test_df, preprocessor)

    dropout = float(metadata.get("config", {}).get("dropout", DEFAULT_DROPOUT)) if metadata else DEFAULT_DROPOUT
    input_dim = preprocessor_input_dim(preprocessor)
    model = PlayerNet(input_dim=input_dim, dropout=dropout)
    checkpoint = load_model_checkpoint(model_path, expected_input_dim=input_dim, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        raw_val_prob = sigmoid_numpy(model(torch.tensor(X_val, dtype=torch.float32)).numpy().reshape(-1))
        raw_test_prob = sigmoid_numpy(model(torch.tensor(X_test, dtype=torch.float32)).numpy().reshape(-1))

    raw_threshold, raw_val_metrics = select_best_threshold(y_val, raw_val_prob)
    raw_test_metrics = classification_metrics(y_test, raw_test_prob, raw_threshold)

    calibrator = None
    if calibrator_path and Path(calibrator_path).exists():
        calibrator = load(calibrator_path)
    calibrated_val_metrics = None
    calibrated_test_metrics = None
    calibrated_threshold = None
    if calibrator is not None:
        calibrated_threshold = float(metadata.get("pytorch", {}).get("selected_threshold", 0.5)) if metadata else 0.5
        calibrated_val_prob = apply_probability_calibrator(calibrator, raw_val_prob)
        calibrated_test_prob = apply_probability_calibrator(calibrator, raw_test_prob)
        calibrated_val_metrics = classification_metrics(y_val, calibrated_val_prob, calibrated_threshold)
        calibrated_test_metrics = classification_metrics(y_test, calibrated_test_prob, calibrated_threshold)

    baseline_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(split_artifact["seed"]))
    baseline_model.fit(X_train, y_train)
    baseline_val_prob = baseline_model.predict_proba(X_val)[:, 1]
    baseline_threshold, baseline_val_metrics = select_best_threshold(y_val, baseline_val_prob)
    baseline_test_prob = baseline_model.predict_proba(X_test)[:, 1]
    baseline_test_metrics = classification_metrics(y_test, baseline_test_prob, baseline_threshold)

    return {
        "timing": {
            "load_data_seconds": round(float(load_elapsed), 4),
        },
        "dataset_summary": data_summary,
        "splits": {
            "train_size": int(len(train_df)),
            "validation_size": int(len(validation_df)),
            "test_size": int(len(test_df)),
        },
        "pytorch": {
            "raw_validation_threshold": float(raw_threshold),
            "raw_validation": raw_val_metrics,
            "raw_test": raw_test_metrics,
            "calibrated_threshold": calibrated_threshold,
            "calibrated_validation": calibrated_val_metrics,
            "calibrated_test": calibrated_test_metrics,
        },
        "baseline_logistic": {
            "selected_threshold": float(baseline_threshold),
            "validation": baseline_val_metrics,
            "test": baseline_test_metrics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua artefactos ya entrenados sin recalcular splits manualmente.")
    parser.add_argument("--db-url", type=str, required=True, help="URL de la base de entrenamiento")
    parser.add_argument("--model-path", type=str, default=None, help="Ruta del modelo PyTorch")
    parser.add_argument("--preprocessor-path", type=str, default=None, help="Ruta del preprocesador")
    parser.add_argument("--calibrator-path", type=str, default=None, help="Ruta del calibrador")
    parser.add_argument("--metadata-path", type=str, default="training_metadata.json", help="Ruta del metadata del entrenamiento")
    parser.add_argument("--splits-path", type=str, default=None, help="Ruta de los splits persistidos")
    parser.add_argument("--cache-path", type=str, default=None, help="Ruta del cache del dataframe temporal")
    parser.add_argument("--no-cache", action="store_true", help="Deshabilita el uso del cache del dataframe temporal")
    args = parser.parse_args()

    metadata = {}
    if args.metadata_path and Path(args.metadata_path).exists():
        metadata = json.loads(Path(args.metadata_path).read_text(encoding="utf-8"))

    model_path = _resolve_artifact_path(args.model_path, metadata, "model_path", "model.pt")
    preprocessor_path = _resolve_artifact_path(args.preprocessor_path, metadata, "preprocessor_path", "preprocessor.joblib")
    calibrator_path = _resolve_artifact_path(args.calibrator_path, metadata, "calibrator_path", "probability_calibrator.joblib")
    splits_path = _resolve_artifact_path(args.splits_path, metadata, "splits_path", "training_splits.json")

    results = evaluate_saved_model(
        db_url=args.db_url,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        splits_path=splits_path,
        metadata_path=args.metadata_path,
        calibrator_path=calibrator_path,
        cache_path=args.cache_path,
        use_cache=not args.no_cache,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
