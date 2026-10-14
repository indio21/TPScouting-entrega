from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

SCOUTING_APP_DIR = Path(__file__).resolve().parents[1] / "scouting_app"
if str(SCOUTING_APP_DIR) not in sys.path:
    sys.path.insert(0, str(SCOUTING_APP_DIR))

from ml.score_evaluation import combined_scores, paired_bootstrap_deltas
from ml.scoring import combined_probability
from ml.target_policy import (
    apply_temporal_target_policy,
    fit_temporal_target_policy,
    target_independent_split_ids,
)
from player_logic import ATTRIBUTE_FIELDS


def target_frame(rows: int = 120) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "player_id": np.arange(1, rows + 1),
            "age": 12 + (np.arange(rows) % 6),
            "position": np.asarray(["Defensa", "Mediocampista", "Delantero"])[
                np.arange(rows) % 3
            ],
            "progression_score": index / rows,
            "future_final_score": 5.0 + index / rows * 3.0,
            "future_scout_projection_score": 5.2 + index / rows * 2.5,
            "observed_weighted_volatility": 0.1 + index / rows * 0.3,
            "weighted_score_growth": -0.1 + index / rows * 0.5,
            "final_score_growth": -0.1 + index / rows * 0.5,
            "future_high_difficulty_score": 5.0 + index / rows * 3.0,
            "pressure_score_growth": -0.1 + index / rows * 0.4,
            "start_rate_growth": -0.1 + index / rows * 0.4,
            "scout_projection_growth": -0.1 + index / rows * 0.4,
            "future_availability_pct": 60.0 + index / rows * 35.0,
            "future_fatigue_pct": 60.0 - index / rows * 25.0,
            "future_injury_rate": 0.30 - index / rows * 0.20,
            "future_match_entry_count": 3.0,
            "future_minutes_per_match": 35.0 + index / rows * 50.0,
            "future_start_rate": 0.25 + index / rows * 0.65,
            "future_natural_position_rate": 0.50 + index / rows * 0.45,
        }
    )


def test_target_policy_uses_train_only_and_is_unchanged_by_test_mutation():
    dataframe = target_frame()
    train = dataframe.iloc[:80].copy()
    policy = fit_temporal_target_policy(train)

    mutated_test = dataframe.iloc[80:].copy()
    mutated_test["progression_score"] = 999.0
    mutated_test["future_final_score"] = 999.0
    policy_after_test_mutation = fit_temporal_target_policy(train)

    assert policy == policy_after_test_mutation
    assert policy.fitted_on == "train_only"
    assert policy.fitted_player_count == 80


def test_target_independent_splits_are_deterministic_disjoint_and_complete():
    dataframe = target_frame(300)
    first = target_independent_split_ids(dataframe, seed=42)
    second = target_independent_split_ids(dataframe, seed=42)

    assert first == second
    groups = [set(first[name]) for name in first]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert set.union(*groups) == set(dataframe["player_id"])


def test_frozen_policy_applies_same_threshold_to_every_partition():
    dataframe = target_frame()
    policy = fit_temporal_target_policy(dataframe.iloc[:80])
    validation = apply_temporal_target_policy(dataframe.iloc[80:100], policy)
    test = apply_temporal_target_policy(dataframe.iloc[100:], policy)

    assert validation["temporal_target_threshold"].nunique() == 1
    assert test["temporal_target_threshold"].nunique() == 1
    assert validation["temporal_target_threshold"].iloc[0] == policy.progression_threshold
    assert test["temporal_target_threshold"].iloc[0] == policy.progression_threshold


def test_combined_probability_is_bounded_and_renormalizes_missing_components():
    assert combined_probability(0.5) == 0.5
    value = combined_probability(0.5, average_final_score=8.0, fit_score=16.0)
    assert round(value, 4) == 0.695
    assert combined_probability(5.0, average_final_score=99.0, fit_score=99.0) == 0.99


def test_combined_scores_preserve_row_alignment():
    rows = []
    for player_id, rating in ((1, 6.0), (2, 8.0)):
        row = {
            "player_id": player_id,
            "position": "Mediocampista",
            "avg_final_score_hist": rating,
        }
        row.update({field: 10 + player_id for field in ATTRIBUTE_FIELDS})
        rows.append(row)
    values = combined_scores(pd.DataFrame(rows), np.asarray([0.2, 0.8]))
    assert len(values) == 2
    assert values[1] > values[0]


def test_paired_bootstrap_reports_candidate_minus_reference():
    labels = np.asarray([0, 0, 0, 1, 1, 1] * 20)
    reference = np.asarray([0.1, 0.2, 0.3, 0.6, 0.7, 0.8] * 20)
    candidate = np.asarray([0.05, 0.1, 0.2, 0.8, 0.9, 0.95] * 20)
    result = paired_bootstrap_deltas(labels, reference, candidate, iterations=100, seed=7)
    assert result["comparison"] == "candidate_minus_reference"
    assert result["metrics"]["brier_score"]["mean_delta"] < 0
