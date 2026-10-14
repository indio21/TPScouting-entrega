"""Repite en memoria la corrida leakage-safe y compara sus resultados guardados."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from joblib import load


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "scouting_app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ml.target_policy import leakage_safe_labeled_splits  # noqa: E402
from preprocessing import TEMPORAL_TARGET_COLUMN, preprocessor_input_dim, transform_features  # noqa: E402
from train_model import load_model_checkpoint, train_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    cache = load(APP_DIR / "temporal_training_dataframe.joblib")
    raw_df = cache["dataframe"]
    raw_df = raw_df[raw_df["age"].between(12, 18)].copy()
    train_df, validation_df, test_df, policy, split_ids = leakage_safe_labeled_splits(raw_df, seed=42)
    combined_df = __import__("pandas").concat([train_df, validation_df, test_df], ignore_index=True)
    model, preprocessor, calibrator, metadata, split_artifact = train_model(
        combined_df,
        combined_df[TEMPORAL_TARGET_COLUMN].astype(np.float32).to_numpy(),
        epochs=30,
        lr=5e-4,
        patience=8,
        split_frames=(train_df, validation_df, test_df),
    )

    stored_metadata = json.loads((run_dir / "training_metadata.json").read_text(encoding="utf-8"))
    stored_policy = json.loads((run_dir / "target_policy.json").read_text(encoding="utf-8"))
    stored_splits = json.loads((run_dir / "training_splits.json").read_text(encoding="utf-8"))
    stored_preprocessor = load(run_dir / "preprocessor.joblib")
    stored_checkpoint = load_model_checkpoint(
        str(run_dir / "model.pt"), expected_input_dim=preprocessor_input_dim(stored_preprocessor)
    )

    state_equal = all(
        torch.equal(value.cpu(), stored_checkpoint["state_dict"][name].cpu())
        for name, value in model.state_dict().items()
    )
    sample = test_df.head(100)
    transformed_equal = np.array_equal(
        transform_features(sample, preprocessor),
        transform_features(sample, stored_preprocessor),
    )
    calibrator_equal = True
    if calibrator is not None:
        stored_calibrator = load(run_dir / "probability_calibrator.joblib")
        values = np.linspace(0.0, 1.0, 101, dtype=np.float32)
        calibrator_equal = np.array_equal(
            np.asarray(calibrator.predict(values)),
            np.asarray(stored_calibrator.predict(values)),
        )

    checks = {
        "target_policy_exact": policy.to_dict() == stored_policy,
        "split_ids_exact": all(split_ids[key] == stored_splits[key] for key in split_ids),
        "training_split_artifact_exact": all(
            split_artifact[key] == stored_splits[key]
            for key in ("train_player_ids", "validation_player_ids", "test_player_ids")
        ),
        "model_state_exact": state_equal,
        "preprocessor_transform_exact": transformed_equal,
        "calibrator_predictions_exact": calibrator_equal,
        "training_history_exact": metadata["pytorch"]["history"] == stored_metadata["pytorch"]["history"],
        "test_metrics_exact": metadata["pytorch"]["test"] == stored_metadata["pytorch"]["test"],
    }
    result = {
        "checks": checks,
        "passed": all(checks.values()),
        "repeat_training_duration_seconds": metadata["config"]["training_duration_seconds"],
        "note": "Timestamps and wall-clock duration are expected to differ and are not compared.",
    }
    output = run_dir / "reproducibility_check.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
