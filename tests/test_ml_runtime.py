import logging

import numpy as np
import pytest


class _ProbabilityCalibrator:
    def predict_proba(self, values):
        positive = np.clip(values[:, 0] + 0.1, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


class _ClippedCalibrator:
    def predict(self, values):
        return np.array([-0.5, 1.5], dtype=np.float32)


class _BrokenCalibrator:
    def predict(self, _values):
        raise RuntimeError("calibrador roto")


def test_probability_calibrator_supports_common_interfaces(app_module, caplog):
    probabilities = [0.2, 0.9]

    assert np.allclose(
        app_module.apply_probability_calibrator(_ProbabilityCalibrator(), probabilities),
        [0.3, 1.0],
    )
    assert np.allclose(
        app_module.apply_probability_calibrator(_ClippedCalibrator(), probabilities),
        [0.0, 1.0],
    )

    with caplog.at_level(logging.WARNING):
        fallback = app_module.apply_probability_calibrator(_BrokenCalibrator(), probabilities)
    assert np.allclose(fallback, probabilities)
    assert "calibrador roto" in caplog.text


def test_missing_or_invalid_calibrator_is_optional(app_module, tmp_path, caplog):
    missing = tmp_path / "missing.joblib"
    assert app_module.load_probability_calibrator(str(missing)) is None

    invalid = tmp_path / "invalid.joblib"
    invalid.write_text("no es joblib", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert app_module.load_probability_calibrator(str(invalid)) is None
    assert "No se pudo cargar" in caplog.text


def test_runtime_artifacts_do_not_retrain_without_explicit_callback(app_module, tmp_path):
    with pytest.raises(FileNotFoundError, match="modelo no encontrado"):
        app_module.load_runtime_artifacts(
            str(tmp_path / "model.pt"),
            str(tmp_path / "preprocessor.joblib"),
            allow_retrain=False,
        )


def test_runtime_artifacts_report_failed_retraining(app_module, tmp_path):
    calls = []

    def failed_update():
        calls.append(True)
        return False, ["fallo controlado"]

    with pytest.raises(RuntimeError, match="No se pudo reentrenar"):
        app_module.load_runtime_artifacts(
            str(tmp_path / "model.pt"),
            str(tmp_path / "preprocessor.joblib"),
            allow_retrain=True,
            update_callback=failed_update,
        )
    assert calls == [True]
