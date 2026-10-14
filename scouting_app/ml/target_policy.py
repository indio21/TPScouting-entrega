"""Política temporal ajustada solo con train para evitar fuga entre particiones."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "temporal_target_label"
TEST_SIZE = 0.15
VALIDATION_SIZE_ON_TRAIN_POOL = 0.17647058823529413


@dataclass(frozen=True)
class TemporalTargetPolicy:
    fitted_on: str
    fitted_player_count: int
    positive_rate_design_quantile: float
    progression_threshold: float
    future_final_floor: float
    future_score_threshold: float
    future_projection_floor: float
    future_projection_threshold: float
    volatility_threshold: float
    breakout_growth_threshold: float
    breakout_final_growth_threshold: float
    breakout_difficulty_threshold: float
    pressure_growth_threshold: float
    start_rate_growth_threshold: float
    scout_growth_threshold: float
    availability_floor: float
    fatigue_ceiling: float
    injury_ceiling: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _safe_quantile(df: pd.DataFrame, column: str, quantile: float, default: float) -> float:
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return float(default)
    result = float(values.quantile(quantile))
    return float(default) if np.isnan(result) else result


def fit_temporal_target_policy(train_df: pd.DataFrame) -> TemporalTargetPolicy:
    """Ajusta todos los umbrales dependientes de distribución únicamente con train."""
    if train_df.empty:
        raise ValueError("No se puede ajustar la política temporal con train vacío.")
    required = {
        "player_id",
        "progression_score",
        "future_final_score",
        "future_scout_projection_score",
        "observed_weighted_volatility",
        "weighted_score_growth",
        "final_score_growth",
        "future_high_difficulty_score",
        "pressure_score_growth",
        "start_rate_growth",
        "scout_projection_growth",
        "future_availability_pct",
        "future_fatigue_pct",
        "future_injury_rate",
    }
    missing = sorted(required.difference(train_df.columns))
    if missing:
        raise ValueError(f"Faltan columnas para ajustar la política temporal: {missing}")
    if train_df["player_id"].duplicated().any():
        raise ValueError("Train contiene player_id duplicados.")

    values = {
        "future_final_floor": max(_safe_quantile(train_df, "future_final_score", 0.35, 5.15), 4.85),
        "future_score_threshold": max(_safe_quantile(train_df, "future_final_score", 0.68, 5.75), 5.25),
        "future_projection_floor": max(
            _safe_quantile(train_df, "future_scout_projection_score", 0.30, 5.60), 5.30
        ),
        "future_projection_threshold": max(
            _safe_quantile(train_df, "future_scout_projection_score", 0.65, 6.30), 5.85
        ),
        "volatility_threshold": _safe_quantile(train_df, "observed_weighted_volatility", 0.88, 0.55),
        "breakout_growth_threshold": _safe_quantile(train_df, "weighted_score_growth", 0.64, 0.04),
        "breakout_final_growth_threshold": _safe_quantile(train_df, "final_score_growth", 0.62, 0.04),
        "breakout_difficulty_threshold": max(
            _safe_quantile(train_df, "future_high_difficulty_score", 0.55, 5.80), 5.20
        ),
        "pressure_growth_threshold": _safe_quantile(train_df, "pressure_score_growth", 0.56, -0.05),
        "start_rate_growth_threshold": _safe_quantile(train_df, "start_rate_growth", 0.55, -0.02),
        "scout_growth_threshold": _safe_quantile(train_df, "scout_projection_growth", 0.55, -0.02),
        "availability_floor": max(_safe_quantile(train_df, "future_availability_pct", 0.20, 58.0), 52.0),
        "fatigue_ceiling": min(_safe_quantile(train_df, "future_fatigue_pct", 0.86, 66.0), 72.0),
        "injury_ceiling": min(_safe_quantile(train_df, "future_injury_rate", 0.90, 0.42), 0.55),
    }
    quality_gate = (
        train_df["future_final_score"].fillna(0.0).ge(values["future_final_floor"])
        & train_df["future_scout_projection_score"].fillna(0.0).ge(values["future_projection_floor"])
        & train_df["future_availability_pct"].fillna(0.0).ge(values["availability_floor"])
        & train_df["future_fatigue_pct"].fillna(100.0).le(values["fatigue_ceiling"])
        & train_df["future_injury_rate"].fillna(1.0).le(values["injury_ceiling"])
        & train_df["future_match_entry_count"].fillna(0.0).ge(1.0)
        & train_df["observed_weighted_volatility"].fillna(0.0).le(values["volatility_threshold"])
    )
    quality_scores = pd.to_numeric(train_df.loc[quality_gate, "progression_score"], errors="coerce").dropna()
    desired_positive_count = max(1, int(round(len(train_df) * 0.08)))
    if quality_scores.empty:
        progression_threshold = _safe_quantile(train_df, "progression_score", 0.92, 0.24)
    else:
        selection_quantile = max(0.0, 1.0 - min(desired_positive_count / len(quality_scores), 1.0))
        progression_threshold = float(quality_scores.quantile(selection_quantile))

    return TemporalTargetPolicy(
        fitted_on="train_only",
        fitted_player_count=int(len(train_df)),
        positive_rate_design_quantile=0.08,
        progression_threshold=progression_threshold,
        **values,
    )


def apply_temporal_target_policy(df: pd.DataFrame, policy: TemporalTargetPolicy) -> pd.DataFrame:
    """Aplica umbrales congelados sin recalcular cuantiles ni cuotas en val/test."""
    result = df.copy()
    quality_gate = (
        result["future_final_score"].fillna(0.0).ge(policy.future_final_floor)
        & result["future_scout_projection_score"].fillna(0.0).ge(policy.future_projection_floor)
        & result["future_availability_pct"].fillna(0.0).ge(policy.availability_floor)
        & result["future_fatigue_pct"].fillna(100.0).le(policy.fatigue_ceiling)
        & result["future_injury_rate"].fillna(1.0).le(policy.injury_ceiling)
        & result["future_match_entry_count"].fillna(0.0).ge(1.0)
        & result["observed_weighted_volatility"].fillna(0.0).le(policy.volatility_threshold)
    )
    consolidation = (
        quality_gate
        & result["future_final_score"].fillna(0.0).ge(policy.future_score_threshold)
        & result["future_scout_projection_score"].fillna(0.0).ge(policy.future_projection_threshold)
        & (
            result["future_minutes_per_match"].fillna(0.0).ge(42.0)
            | result["future_start_rate"].fillna(0.0).ge(0.36)
        )
        & result["future_natural_position_rate"].fillna(0.0).ge(0.46)
    )
    breakout = (
        quality_gate
        & (
            result["weighted_score_growth"].fillna(-99.0).ge(policy.breakout_growth_threshold)
            | result["final_score_growth"].fillna(-99.0).ge(policy.breakout_final_growth_threshold)
        )
        & (
            result["future_high_difficulty_score"].fillna(0.0).ge(policy.breakout_difficulty_threshold)
            | result["pressure_score_growth"].fillna(-99.0).ge(policy.pressure_growth_threshold)
            | result["scout_projection_growth"].fillna(-99.0).ge(policy.scout_growth_threshold)
        )
        & (
            result["start_rate_growth"].fillna(-99.0).ge(policy.start_rate_growth_threshold)
            | result["future_start_rate"].fillna(0.0).ge(0.34)
            | result["future_minutes_per_match"].fillna(0.0).ge(38.0)
        )
    )
    progression = result["progression_score"].fillna(-99.0).ge(policy.progression_threshold)
    selected = quality_gate & progression

    result["temporal_target_threshold"] = policy.progression_threshold
    result["temporal_future_score_threshold"] = policy.future_score_threshold
    result["temporal_quality_gate"] = quality_gate
    result["temporal_target_candidate"] = selected
    result["temporal_consolidation_path"] = selected & consolidation
    result["temporal_breakout_path"] = selected & ~result["temporal_consolidation_path"]
    result[TARGET_COLUMN] = selected.astype(bool)
    return result


def _cohort_labels(df: pd.DataFrame) -> pd.Series:
    age_group = pd.cut(
        pd.to_numeric(df["age"], errors="coerce"),
        bins=[-np.inf, 14.0, 16.0, np.inf],
        labels=["12-14", "15-16", "17-18"],
    ).astype(str)
    return df["position"].fillna("Sin posicion").astype(str) + ":" + age_group


def target_independent_split_ids(
    df: pd.DataFrame,
    seed: int,
    test_size: float = TEST_SIZE,
    validation_size_on_train_pool: float = VALIDATION_SIZE_ON_TRAIN_POOL,
) -> Dict[str, list[int]]:
    """Separa por posición/edad antes de crear la etiqueta temporal."""
    if df.empty or df["player_id"].duplicated().any():
        raise ValueError("El dataframe debe contener player_id únicos y al menos una fila.")
    ids = df["player_id"].astype(int).to_numpy()
    cohorts = _cohort_labels(df).to_numpy()
    train_val_ids, test_ids, train_val_cohorts, _ = train_test_split(
        ids,
        cohorts,
        test_size=test_size,
        random_state=seed,
        stratify=cohorts,
    )
    train_ids, validation_ids = train_test_split(
        train_val_ids,
        test_size=validation_size_on_train_pool,
        random_state=seed,
        stratify=train_val_cohorts,
    )
    return {
        "train_player_ids": [int(value) for value in train_ids],
        "validation_player_ids": [int(value) for value in validation_ids],
        "test_player_ids": [int(value) for value in test_ids],
    }


def select_ids(df: pd.DataFrame, player_ids: Iterable[int]) -> pd.DataFrame:
    indexed = df.set_index("player_id", drop=False)
    ordered = [int(value) for value in player_ids]
    missing = sorted(set(ordered).difference(int(value) for value in indexed.index))
    if missing:
        raise ValueError(f"Faltan player_id del split: {missing[:10]}")
    return indexed.loc[ordered].reset_index(drop=True)


def leakage_safe_labeled_splits(
    raw_df: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TemporalTargetPolicy, Dict[str, list[int]]]:
    split_ids = target_independent_split_ids(raw_df, seed=seed)
    raw_train = select_ids(raw_df, split_ids["train_player_ids"])
    policy = fit_temporal_target_policy(raw_train)
    train_df = apply_temporal_target_policy(raw_train, policy)
    validation_df = apply_temporal_target_policy(
        select_ids(raw_df, split_ids["validation_player_ids"]), policy
    )
    test_df = apply_temporal_target_policy(select_ids(raw_df, split_ids["test_player_ids"]), policy)
    return train_df, validation_df, test_df, policy, split_ids
