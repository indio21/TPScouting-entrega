"""Funciones puras para construir y evaluar el score combinado del MVP."""

from __future__ import annotations

from typing import Optional


def combined_probability(
    base_probability: float,
    average_final_score: Optional[float] = None,
    fit_score: Optional[float] = None,
    model_weight: float = 0.35,
    rating_weight: float = 0.35,
    fit_weight: float = 0.30,
) -> float:
    """Combina señales normalizadas y renormaliza cuando falta alguna de ellas."""
    components = [(float(model_weight), min(max(float(base_probability), 0.0), 1.0))]
    if average_final_score is not None:
        components.append(
            (float(rating_weight), min(max(float(average_final_score) / 10.0, 0.0), 1.0))
        )
    if fit_score is not None:
        components.append((float(fit_weight), min(max(float(fit_score) / 20.0, 0.0), 1.0)))
    denominator = sum(weight for weight, _ in components)
    if denominator <= 0:
        raise ValueError("La suma de pesos del score combinado debe ser positiva.")
    value = sum(weight * component for weight, component in components) / denominator
    return max(0.0, min(float(value), 0.99))
