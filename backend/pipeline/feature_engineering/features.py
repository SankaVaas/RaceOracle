"""
Feature engineering pipeline for RaceOracle.

Responsibilities:
  - Accept raw horse/race data (dict or DataFrame row)
  - Compute derived features (form decay, going preference score, etc.)
  - Output clean tensors ready for the surgical model
  - Compute recency weights per horse based on days since last run

All feature logic lives here so it's version-controlled separately
from the model — crucial when updating features without retraining.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional
import math


# ── Feature lists (must match train_model.py exactly) ──────────────────────
CAT_FEATURES = [
    "track_idx", "going_idx", "surface_idx",
    "draw_position", "class_drop_rise", "age_years"
]

CONT_FEATURES = [
    "win_rate_last5", "win_rate_career", "place_rate_last5",
    "days_since_last_run", "going_preference_score", "distance_fit_score",
    "jockey_win_rate", "trainer_win_rate", "jockey_trainer_combo",
    "weight_carried_lbs", "speed_rating_last", "speed_rating_avg3",
    "track_win_rate", "field_size",
    "injury_risk", "travel_risk", "fatigue_risk",
]

TRACK_MAP   = {"Royal Ascot": 0, "Cheltenham": 1, "Newmarket": 2, "Epsom": 3, "Goodwood": 4}
GOING_MAP   = {"Firm": 0, "Good": 1, "Good-Soft": 2, "Soft": 3, "Heavy": 4}
SURFACE_MAP = {"Turf": 0, "Dirt": 1, "Synthetic": 2}


# ── Form string parser ──────────────────────────────────────────────────────

def parse_form_string(form: str) -> dict:
    """
    Parse a form string like '1-2-3-1-4' into derived stats.
    Supports: 1-9 finish positions, P (pulled up), F (fell), U (unseated).
    """
    if not form or form == "-":
        return {"win_rate_last5": 0.0, "place_rate_last5": 0.0,
                "avg_finish_last5": 5.0, "form_momentum": 0.0}

    positions = []
    for token in form.replace(" ", "").split("-"):
        try:
            positions.append(int(token))
        except ValueError:
            positions.append(10)   # DNF-type result penalised as 10th

    recent = positions[-5:]   # last 5 runs only
    n = len(recent)
    if n == 0:
        return {"win_rate_last5": 0.0, "place_rate_last5": 0.0,
                "avg_finish_last5": 5.0, "form_momentum": 0.0}

    win_rate   = sum(1 for p in recent if p == 1) / n
    place_rate = sum(1 for p in recent if p <= 3) / n
    avg_finish = np.mean(recent)

    # Form momentum: compare last 2 runs to previous 3 — positive = improving
    if n >= 4:
        early = np.mean(recent[:-2])
        late  = np.mean(recent[-2:])
        momentum = (early - late) / early if early > 0 else 0.0
    else:
        momentum = 0.0

    return {
        "win_rate_last5":   win_rate,
        "place_rate_last5": place_rate,
        "avg_finish_last5": avg_finish,
        "form_momentum":    momentum,
    }


# ── Going preference score ──────────────────────────────────────────────────

def compute_going_preference(horse_going_history: list, current_going: str) -> float:
    """
    Score how well a horse performs on today's going.
    horse_going_history: list of (going_str, finish_position) tuples
    Returns: float in [-1, 1], positive = suits today's going
    """
    if not horse_going_history:
        return 0.0

    # Group similar going conditions
    going_groups = {
        "fast":  ["Firm", "Good"],
        "soft":  ["Good-Soft", "Soft"],
        "heavy": ["Heavy"],
    }
    def going_group(g):
        for grp, vals in going_groups.items():
            if g in vals:
                return grp
        return "unknown"

    target_grp = going_group(current_going)
    same, other = [], []
    for going, pos in horse_going_history:
        if going_group(going) == target_grp:
            same.append(pos)
        else:
            other.append(pos)

    if not same:
        return 0.0   # no data on this going type

    avg_same  = np.mean(same)
    avg_other = np.mean(other) if other else avg_same

    # Normalise: lower finish position is better
    score = (avg_other - avg_same) / max(avg_other, 1.0)
    return float(np.clip(score, -1.0, 1.0))


# ── Distance fit score ──────────────────────────────────────────────────────

def compute_distance_fit(distance_history: list, target_distance: float) -> float:
    """
    How well does this horse's history align with today's distance?
    distance_history: list of (distance_furlongs, finish_position) tuples
    """
    if not distance_history:
        return 0.0

    distances = np.array([d for d, _ in distance_history])
    positions = np.array([p for _, p in distance_history])

    # Weight by proximity to target distance
    diffs   = np.abs(distances - target_distance)
    weights = np.exp(-diffs / 4.0)   # decay over ~4 furlong difference
    if weights.sum() == 0:
        return 0.0

    weighted_avg_pos = (weights * positions).sum() / weights.sum()
    # Normalise: avg position 1 → score +1, avg position 8+ → score -1
    score = 1.0 - (weighted_avg_pos - 1.0) / 7.0
    return float(np.clip(score, -1.0, 1.0))


# ── Recency weight vector ───────────────────────────────────────────────────

def compute_recency_weights(days_since_last_run: float, seq_len: int = 7) -> np.ndarray:
    """
    Produces a (seq_len,) weight vector fed into FormDecayFFN.
    Horses that ran recently get weights closer to 1.0.
    Horses that haven't run in 60+ days get low weights (ring rust).

    Example:
        7  days → max weight ~0.79  (fresh, sharp)
        21 days → max weight ~0.50  (normal gap)
        60 days → max weight ~0.14  (ring rust)
    """
    base_decay = math.exp(-days_since_last_run / 30.0)   # exponential decay on days
    # Shape across sequence positions (more recent positions → higher weight)
    positions  = np.arange(seq_len)
    pos_decay  = np.exp(-0.1 * (seq_len - 1 - positions))  # positional ramp 0 -> 1
    # Multiply: base_decay sets the ceiling, pos_decay shapes within-sequence ramp
    # Do NOT normalise — that would erase the between-horse difference
    weights    = base_decay * pos_decay
    return weights.astype(np.float32)


# ── Jockey-trainer combo score ──────────────────────────────────────────────

def compute_jockey_trainer_combo(jockey_id: str, trainer_id: str,
                                  historical_pairs: dict) -> float:
    """
    Returns the win rate of this jockey-trainer combination from history.
    historical_pairs: {(jockey_id, trainer_id): (wins, runs)} dict
    """
    key = (jockey_id, trainer_id)
    if key not in historical_pairs or historical_pairs[key][1] < 5:
        return 0.1   # default: small prior if insufficient data
    wins, runs = historical_pairs[key]
    return wins / runs


# ── Master feature builder ──────────────────────────────────────────────────

def build_horse_features(raw: dict, race_context: dict,
                          historical_pairs: Optional[dict] = None) -> dict:
    """
    Main entry point. Converts a raw horse data dict into the full
    feature set expected by the surgical model.

    raw: {
        "name": str,
        "form": str,                          e.g. "1-2-1-3-2"
        "age": int,
        "weight_lbs": int,
        "draw": int,
        "days_since_last_run": float,
        "jockey_win_rate": float,
        "trainer_win_rate": float,
        "speed_ratings": list[float],         last N speed ratings
        "going_history": list[(str, int)],    (going, finish_pos)
        "distance_history": list[(float,int)],
        "track_wins": int,
        "track_runs": int,
        "xgb_win_prob": float,               from XGBoost model
        "injury_risk": float,                from news pipeline
        "travel_risk": float,
        "fatigue_risk": float,
        "jockey_id": str,
        "trainer_id": str,
    }
    race_context: {
        "track": str, "going": str, "surface": str,
        "distance_furlongs": float, "field_size": int
    }
    """
    # Parse form string
    form_stats = parse_form_string(raw.get("form", ""))

    # Going + distance preference
    going_score    = compute_going_preference(
        raw.get("going_history", []), race_context["going"]
    )
    distance_score = compute_distance_fit(
        raw.get("distance_history", []), race_context["distance_furlongs"]
    )

    # Speed ratings
    spd = raw.get("speed_ratings", [])
    speed_last  = float(spd[-1]) if spd else 85.0
    speed_avg3  = float(np.mean(spd[-3:])) if len(spd) >= 3 else speed_last

    # Track win rate
    tw = raw.get("track_wins", 0)
    tr_runs = raw.get("track_runs", 1)
    track_wr = tw / tr_runs if tr_runs > 0 else 0.0

    # Jockey-trainer combo
    j_t_combo = compute_jockey_trainer_combo(
        raw.get("jockey_id", ""), raw.get("trainer_id", ""),
        historical_pairs or {}
    )

    # Career win rate (from form if not provided)
    career_wins = raw.get("career_wins", 0)
    career_runs = raw.get("career_runs", max(1, len(raw.get("form", "").split("-"))))
    career_wr   = career_wins / career_runs if career_runs > 0 else form_stats["win_rate_last5"]

    # Class drop/rise (-2 to +2: -2 = big drop in class, +2 = big rise)
    class_change = int(np.clip(raw.get("class_change", 0), -2, 2))

    # Categorical indices
    cat_values = [
        TRACK_MAP.get(race_context["track"], 0),
        GOING_MAP.get(race_context["going"], 1),
        SURFACE_MAP.get(race_context["surface"], 0),
        int(np.clip(raw.get("draw", 1), 1, 20)),
        class_change + 2,         # shift to 0-4 range for embedding
        int(np.clip(raw.get("age", 4), 2, 10)) - 2,   # 0-8 range
    ]

    # Continuous values (order must match CONT_FEATURES)
    cont_values = [
        form_stats["win_rate_last5"],
        career_wr,
        form_stats["place_rate_last5"],
        float(raw.get("days_since_last_run", 21)),
        going_score,
        distance_score,
        float(raw.get("jockey_win_rate", 0.12)),
        float(raw.get("trainer_win_rate", 0.12)),
        j_t_combo,
        float(raw.get("weight_lbs", 126)),
        speed_last,
        speed_avg3,
        track_wr,
        float(race_context["field_size"]),
        float(raw.get("injury_risk", 0.0)),
        float(raw.get("travel_risk", 0.0)),
        float(raw.get("fatigue_risk", 0.0)),
    ]

    # Recency weights for FormDecayFFN
    recency_w = compute_recency_weights(
        float(raw.get("days_since_last_run", 21)), seq_len=7
    )

    return {
        "name":             raw.get("name", "Unknown"),
        "cat_values":       cat_values,        # list[int] len=6
        "cont_values":      cont_values,        # list[float] len=17
        "recency_weights":  recency_w,          # np.ndarray shape (7,)
        "xgb_win_prob":     float(raw.get("xgb_win_prob", 1.0 / race_context["field_size"])),
        "news_text":        raw.get("news_text", ""),
        # Pass through for response
        "form":             raw.get("form", "-"),
        "jockey":           raw.get("jockey", ""),
        "trainer":          raw.get("trainer", ""),
        "age":              raw.get("age", 4),
        "weight_lbs":       raw.get("weight_lbs", 126),
        "draw":             raw.get("draw", 1),
    }


def build_race_features(horses_raw: list, race_context: dict,
                         historical_pairs: Optional[dict] = None) -> list:
    """Build features for every horse in a race. Returns list of feature dicts."""
    race_context = {**race_context, "field_size": len(horses_raw)}
    return [build_horse_features(h, race_context, historical_pairs) for h in horses_raw]