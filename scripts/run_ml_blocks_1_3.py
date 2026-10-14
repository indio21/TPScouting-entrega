"""Ejecuta la evaluación histórica y una corrida leakage-safe sin reemplazar runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from joblib import dump, load


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "scouting_app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ml.score_evaluation import (  # noqa: E402
    combined_scores,
    evaluate_probability_variants,
    paired_bootstrap_deltas,
)
from ml.target_policy import leakage_safe_labeled_splits  # noqa: E402
from preprocessing import (  # noqa: E402
    MODEL_FEATURE_COLUMNS,
    TEMPORAL_TARGET_COLUMN,
    load_preprocessor,
    preprocessor_input_dim,
    save_preprocessor,
    transform_features,
)
from train_model import (  # noqa: E402
    DEFAULT_DROPOUT,
    PlayerNet,
    apply_probability_calibrator,
    load_model_checkpoint,
    save_calibrator,
    save_metadata,
    save_model,
    save_split_artifact,
    sigmoid_numpy,
    train_model,
)


FORBIDDEN_FEATURES = {
    "potential_label",
    "temporal_target_label",
    "progression_score",
    "temporal_target_threshold",
    "temporal_future_score_threshold",
    "temporal_target_candidate",
    "temporal_consolidation_path",
    "temporal_breakout_path",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_cached_dataframe(cache_path: Path) -> pd.DataFrame:
    artifact = load(cache_path)
    dataframe = artifact.get("dataframe") if isinstance(artifact, dict) else artifact
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("El cache temporal no contiene un DataFrame.")
    filtered = dataframe[dataframe["age"].between(12, 18)].copy()
    if filtered.empty or filtered["player_id"].duplicated().any():
        raise ValueError("El dataframe temporal filtrado debe tener player_id únicos.")
    leaked = sorted(FORBIDDEN_FEATURES.intersection(MODEL_FEATURE_COLUMNS))
    if leaked:
        raise RuntimeError(f"Columnas prohibidas presentes en MODEL_FEATURE_COLUMNS: {leaked}")
    return filtered


def model_probabilities(
    dataframe: pd.DataFrame,
    model_path: Path,
    preprocessor_path: Path,
    metadata: Dict[str, object],
) -> np.ndarray:
    preprocessor = load_preprocessor(str(preprocessor_path))
    matrix = transform_features(dataframe, preprocessor)
    input_dim = preprocessor_input_dim(preprocessor)
    dropout = float(metadata.get("config", {}).get("dropout", DEFAULT_DROPOUT))
    model = PlayerNet(input_dim=input_dim, dropout=dropout)
    checkpoint = load_model_checkpoint(str(model_path), expected_input_dim=input_dim, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(matrix, dtype=torch.float32)).numpy().reshape(-1)
    return sigmoid_numpy(logits)


def evaluate_artifacts(
    dataframe: pd.DataFrame,
    split_artifact: Dict[str, object],
    model_path: Path,
    preprocessor_path: Path,
    calibrator_path: Optional[Path],
    metadata: Dict[str, object],
    output_dir: Path,
    prefix: str,
) -> Dict[str, object]:
    indexed = dataframe.set_index("player_id", drop=False)

    def split(name: str) -> pd.DataFrame:
        ids = [int(value) for value in split_artifact[name]]
        missing = sorted(set(ids).difference(int(value) for value in indexed.index))
        if missing:
            raise ValueError(f"El split {name} contiene IDs ausentes: {missing[:10]}")
        return indexed.loc[ids].reset_index(drop=True)

    validation_df = split("validation_player_ids")
    test_df = split("test_player_ids")
    raw_validation = model_probabilities(validation_df, model_path, preprocessor_path, metadata)
    raw_test = model_probabilities(test_df, model_path, preprocessor_path, metadata)

    calibrator = load(calibrator_path) if calibrator_path and calibrator_path.exists() else None
    calibrated_validation = apply_probability_calibrator(calibrator, raw_validation)
    calibrated_test = apply_probability_calibrator(calibrator, raw_test)
    combined_raw_validation = combined_scores(validation_df, raw_validation)
    combined_raw_test = combined_scores(test_df, raw_test)
    combined_calibrated_validation = combined_scores(validation_df, calibrated_validation)
    combined_calibrated_test = combined_scores(test_df, calibrated_test)

    validation_probabilities = {
        "raw_probability": raw_validation,
        "calibrated_probability": calibrated_validation,
        "combined_from_raw": combined_raw_validation,
        "combined_from_calibrated": combined_calibrated_validation,
    }
    test_probabilities = {
        "raw_probability": raw_test,
        "calibrated_probability": calibrated_test,
        "combined_from_raw": combined_raw_test,
        "combined_from_calibrated": combined_calibrated_test,
    }
    variants = evaluate_probability_variants(
        validation_df, test_df, validation_probabilities, test_probabilities
    )
    bootstrap = {
        name: paired_bootstrap_deltas(
            test_df[TEMPORAL_TARGET_COLUMN].astype(int).to_numpy(),
            raw_test,
            probabilities,
        )
        for name, probabilities in test_probabilities.items()
        if name != "raw_probability"
    }
    predictions = pd.DataFrame(
        {
            "player_id": test_df["player_id"].astype(int),
            "target": test_df[TEMPORAL_TARGET_COLUMN].astype(int),
            **test_probabilities,
        }
    )
    predictions_path = output_dir / f"{prefix}_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    result = {
        "scope": prefix,
        "weights": {"model": 0.35, "historical_rating": 0.35, "position_fit": 0.30},
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "variants": variants,
        "paired_bootstrap_vs_raw": bootstrap,
        "predictions_path": predictions_path.name,
        "predictions_sha256": sha256(predictions_path),
        "visual_bands": {
            "medium": 0.60,
            "high": 0.80,
            "interpretation": "presentation_bands_not_selected_on_validation",
        },
    }
    json_write(output_dir / f"{prefix}_score_evaluation.json", result)
    return result


def artifact_manifest(output_dir: Path, started_at: str, duration: float) -> Dict[str, object]:
    dependency_names = ["torch", "numpy", "pandas", "scikit-learn", "joblib", "sqlalchemy"]
    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    diff = git_output("diff", "--binary")
    source_paths = [
        ROOT / "scouting_app/ml/scoring.py",
        ROOT / "scouting_app/ml/score_evaluation.py",
        ROOT / "scouting_app/ml/target_policy.py",
        ROOT / "scouting_app/train_model.py",
        ROOT / "scripts/run_ml_blocks_1_3.py",
    ]
    return {
        "manifest_version": 1,
        "run_id": output_dir.name,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_duration_seconds": round(float(duration), 4),
        "git": {
            "base_commit": git_output("rev-parse", "HEAD"),
            "working_tree_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
            "working_tree_was_dirty": bool(git_output("status", "--short")),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "dependencies": {
            name: importlib.metadata.version(name) for name in dependency_names
        },
        "source_files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in source_paths
        },
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    timer = perf_counter()
    run_id = args.run_id or started.strftime("%Y%m%dT%H%M%SZ_seed42")
    output_dir = ROOT / "artifacts" / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    cache_path = APP_DIR / "temporal_training_dataframe.joblib"
    raw_df = load_cached_dataframe(cache_path)
    historical_splits = json.loads((APP_DIR / "training_splits.json").read_text(encoding="utf-8"))
    historical_metadata = json.loads((APP_DIR / "training_metadata.json").read_text(encoding="utf-8"))
    historical_evaluation = evaluate_artifacts(
        raw_df,
        historical_splits,
        APP_DIR / "model.pt",
        APP_DIR / "preprocessor.joblib",
        APP_DIR / "probability_calibrator.joblib",
        historical_metadata,
        output_dir,
        "historical",
    )

    train_df, validation_df, test_df, target_policy, split_ids = leakage_safe_labeled_splits(
        raw_df, seed=42
    )
    split_artifact: Dict[str, object] = {
        "version": 2,
        "seed": 42,
        "split_strategy": "target_independent_position_age_cohort",
        **split_ids,
        "train_positive_rate": round(float(train_df[TEMPORAL_TARGET_COLUMN].mean()), 6),
        "validation_positive_rate": round(float(validation_df[TEMPORAL_TARGET_COLUMN].mean()), 6),
        "test_positive_rate": round(float(test_df[TEMPORAL_TARGET_COLUMN].mean()), 6),
    }
    combined_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
    model, preprocessor, calibrator, metadata, trained_split_artifact = train_model(
        combined_df,
        combined_df[TEMPORAL_TARGET_COLUMN].astype(np.float32).to_numpy(),
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        split_frames=(train_df, validation_df, test_df),
    )
    if trained_split_artifact["train_player_ids"] != split_artifact["train_player_ids"]:
        raise RuntimeError("El entrenamiento no conservó el orden del split prefijado.")

    model_path = output_dir / "model.pt"
    preprocessor_path = output_dir / "preprocessor.joblib"
    calibrator_path = output_dir / "probability_calibrator.joblib"
    metadata_path = output_dir / "training_metadata.json"
    splits_path = output_dir / "training_splits.json"
    policy_path = output_dir / "target_policy.json"
    dataframe_path = output_dir / "labeled_temporal_dataframe.joblib"

    save_model(model, str(model_path))
    save_preprocessor(preprocessor, str(preprocessor_path))
    save_calibrator(calibrator, str(calibrator_path))
    save_split_artifact(split_artifact, str(splits_path))
    json_write(policy_path, target_policy.to_dict())
    dump({"version": "leakage_safe_v1", "dataframe": combined_df}, dataframe_path)

    metadata["target_policy"] = target_policy.to_dict()
    metadata["split_strategy"] = split_artifact["split_strategy"]
    metadata["dataset_summary"] = {
        "rows": int(len(combined_df)),
        "target_column": TEMPORAL_TARGET_COLUMN,
        "positive_rates": {
            "train": split_artifact["train_positive_rate"],
            "validation": split_artifact["validation_positive_rate"],
            "test": split_artifact["test_positive_rate"],
        },
        "source_cache_sha256": sha256(cache_path),
        "features_before_encoding": list(MODEL_FEATURE_COLUMNS),
        "forbidden_features_absent": sorted(FORBIDDEN_FEATURES),
    }
    metadata["artifacts"] = {
        "model": model_path.name,
        "preprocessor": preprocessor_path.name,
        "calibrator": calibrator_path.name if calibrator_path.exists() else None,
        "splits": splits_path.name,
        "target_policy": policy_path.name,
        "labeled_dataframe": dataframe_path.name,
    }
    save_metadata(metadata, str(metadata_path))

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (output_dir / "python_environment_freeze.txt").write_text(freeze, encoding="utf-8")

    new_evaluation = evaluate_artifacts(
        combined_df,
        split_artifact,
        model_path,
        preprocessor_path,
        calibrator_path if calibrator_path.exists() else None,
        metadata,
        output_dir,
        "leakage_safe",
    )
    summary = {
        "run_id": run_id,
        "historical_score_evaluation": historical_evaluation,
        "leakage_safe_score_evaluation": new_evaluation,
        "target_policy": target_policy.to_dict(),
        "split_positive_rates": metadata["dataset_summary"]["positive_rates"],
        "runtime_artifacts_replaced": False,
    }
    json_write(output_dir / "summary.json", summary)
    manifest = artifact_manifest(output_dir, started.isoformat(), perf_counter() - timer)
    json_write(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
