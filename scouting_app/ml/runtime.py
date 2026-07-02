"""Carga de artefactos de ML para inferencia en la app."""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from joblib import load as joblib_load

from preprocessing import load_preprocessor as load_saved_preprocessor, preprocessor_input_dim
from train_model import PlayerNet, load_model_checkpoint

logger = logging.getLogger(__name__)


def load_model_state(model_path: str, input_dim: int) -> PlayerNet:
    model = PlayerNet(input_dim=input_dim)
    checkpoint = load_model_checkpoint(
        model_path,
        expected_input_dim=input_dim,
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_probability_calibrator(calibrator_path: str) -> Optional[object]:
    if not os.path.exists(calibrator_path):
        return None
    try:
        return joblib_load(calibrator_path)
    except Exception as exc:
        logger.warning("No se pudo cargar el calibrador de probabilidad %s: %s", calibrator_path, exc)
        return None


def apply_probability_calibrator(calibrator: Optional[object], probabilities: List[float]) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if calibrator is None:
        return raw
    try:
        if hasattr(calibrator, "predict_proba"):
            return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1].astype(np.float32)
        return np.clip(np.asarray(calibrator.predict(raw), dtype=np.float32), 0.0, 1.0)
    except Exception as exc:
        logger.warning("No se pudo aplicar el calibrador de probabilidad en runtime: %s", exc)
        return raw


def load_runtime_artifacts(
    model_path: str,
    preprocessor_path: str,
    calibrator_path: Optional[str] = None,
    allow_retrain: bool = True,
    update_callback: Optional[Callable[[], Tuple[bool, List[str]]]] = None,
) -> Tuple[Optional[PlayerNet], Optional[object], Optional[object]]:
    def retrain_or_raise(message: str, original_exc: Optional[Exception] = None) -> None:
        if not allow_retrain or update_callback is None:
            if original_exc is not None:
                raise original_exc
            raise FileNotFoundError(message)
        print(message)
        success, logs = update_callback()
        for log in logs:
            print(log)
        if not success:
            raise RuntimeError(
                "No se pudo reentrenar el modelo automaticamente. Ejecute train_model.py manualmente."
            ) from original_exc

    if not os.path.exists(model_path):
        retrain_or_raise("Advertencia: modelo no encontrado. Intentando reentrenar automaticamente.")
    if not os.path.exists(preprocessor_path):
        retrain_or_raise("Advertencia: preprocesador no encontrado. Intentando reentrenar automaticamente.")

    preprocessor = load_saved_preprocessor(preprocessor_path)
    input_dim = preprocessor_input_dim(preprocessor)
    try:
        model = load_model_state(model_path, input_dim)
    except RuntimeError as exc:
        logger.warning("Modelo incompatible con preprocesador: %s", exc)
        retrain_or_raise(
            "Advertencia: modelo incompatible con el preprocesador actual. Intentando reentrenar automaticamente.",
            original_exc=exc,
        )
        preprocessor = load_saved_preprocessor(preprocessor_path)
        input_dim = preprocessor_input_dim(preprocessor)
        model = load_model_state(model_path, input_dim)
    calibrator = load_probability_calibrator(calibrator_path) if calibrator_path else None
    return model, preprocessor, calibrator
