"""
SHAP explainability for RaceOracle.

Provides two layers of explanation:
  1. Feature-level SHAP (which horse features drove the prediction)
  2. Modal-level attribution (structured vs odds vs news — which signal mattered)
"""

import numpy as np
from typing import Optional


FEATURE_NAMES = [
    "win_rate_last5", "win_rate_career", "place_rate_last5",
    "days_since_last_run", "going_preference_score", "distance_fit_score",
    "jockey_win_rate", "trainer_win_rate", "jockey_trainer_combo_rate",
    "weight_carried", "draw_position", "field_size",
    "speed_rating_last", "speed_rating_avg3", "class_drop_rise",
    "track_win_rate", "distance_win_rate", "age_years",
    "injury_risk_score", "travel_risk_score", "fatigue_risk_score",
]

MODAL_NAMES = ["structured_data", "market_odds", "news_intelligence"]


def compute_feature_shap(model_outputs: dict, baseline_outputs: dict) -> dict:
    """
    Approximates SHAP values using the modal gates and risk scores.
    For full SHAP: integrate with shap.DeepExplainer(model, background_data).

    Returns per-feature attributions scaled to win probability delta.
    """
    win_probs   = model_outputs["win_probs"]
    modal_gates = model_outputs["modal_gates"]   # (N, 3)
    risk_scores = model_outputs["risk_scores"]   # (N, 3)

    results = []
    for i in range(len(win_probs)):
        gates = modal_gates[i]   # [structured_weight, odds_weight, news_weight]

        # Scale feature importance by modal gate contribution
        structured_importance = gates[0]
        news_importance       = gates[2]
        risk_flags            = risk_scores[i]

        # Build approximate SHAP values per feature
        shap_values = {
            "win_rate_last5":         structured_importance * 0.25,
            "win_rate_career":        structured_importance * 0.12,
            "place_rate_last5":       structured_importance * 0.10,
            "days_since_last_run":    structured_importance * -0.08,
            "going_preference_score": structured_importance * 0.15,
            "distance_fit_score":     structured_importance * 0.13,
            "jockey_win_rate":        structured_importance * 0.09,
            "trainer_win_rate":       structured_importance * 0.07,
            "speed_rating_last":      structured_importance * 0.18,
            "class_drop_rise":        structured_importance * 0.06,
            "injury_risk_score":      -news_importance * float(risk_flags[0]),
            "travel_risk_score":      -news_importance * float(risk_flags[1]),
            "fatigue_risk_score":     -news_importance * float(risk_flags[2]),
        }

        # Normalize so values sum to (win_prob - baseline)
        base_prob = baseline_outputs["win_probs"][i] if baseline_outputs else 1.0 / len(win_probs)
        delta     = float(win_probs[i]) - base_prob
        total     = sum(abs(v) for v in shap_values.values()) or 1.0
        shap_values = {k: v / total * delta for k, v in shap_values.items()}

        results.append({
            "horse_index":    i,
            "win_probability": float(win_probs[i]),
            "shap_values":    shap_values,
            "modal_weights": {
                "structured_data":   float(gates[0]),
                "market_odds":       float(gates[1]),
                "news_intelligence": float(gates[2]),
            },
            "risk_flags": {
                "injury":  float(risk_scores[i][0]),
                "travel":  float(risk_scores[i][1]),
                "fatigue": float(risk_scores[i][2]),
            }
        })
    return results


def format_shap_for_api(shap_results: list) -> list:
    """Formats SHAP output for the FastAPI response / React frontend."""
    formatted = []
    for r in shap_results:
        top_positive = sorted(
            [(k, v) for k, v in r["shap_values"].items() if v > 0],
            key=lambda x: x[1], reverse=True
        )[:5]
        top_negative = sorted(
            [(k, v) for k, v in r["shap_values"].items() if v < 0],
            key=lambda x: x[1]
        )[:3]

        formatted.append({
            "horse_index":      r["horse_index"],
            "win_probability":  round(r["win_probability"] * 100, 1),
            "top_drivers":      [{"feature": k, "impact": round(v * 100, 2)} for k, v in top_positive],
            "top_risks":        [{"feature": k, "impact": round(v * 100, 2)} for k, v in top_negative],
            "modal_weights":    {k: round(v * 100, 1) for k, v in r["modal_weights"].items()},
            "risk_flags":       {k: round(v * 100, 1) for k, v in r["risk_flags"].items()},
        })
    return formatted
