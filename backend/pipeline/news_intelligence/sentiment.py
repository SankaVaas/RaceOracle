"""
Sentiment utilities — converts LLM risk scores into model features
and dashboard-ready display objects.

This is the bridge between the news intelligence pipeline
and the surgical model's input feature vector.
"""

from typing import Optional


# ── Risk thresholds for flag display ─────────────────────────────────────────

RISK_THRESHOLDS = {
    "injury": {
        "low":    (0.10, "⚪ No injury concerns"),
        "medium": (0.35, "🟡 Minor fitness concern"),
        "high":   (0.60, "🟠 Injury reported"),
        "critical":(0.80, "🔴 Serious injury risk"),
    },
    "travel": {
        "low":    (0.10, "⚪ No travel concerns"),
        "medium": (0.30, "🟡 Recent travel"),
        "high":   (0.55, "🟠 Long-distance travel"),
        "critical":(0.75, "🔴 Intercontinental travel"),
    },
    "fatigue": {
        "low":    (0.10, "⚪ Fresh"),
        "medium": (0.35, "🟡 Busy campaign"),
        "high":   (0.55, "🟠 Fatigue risk"),
        "critical":(0.75, "🔴 Overraced"),
    },
}


def risk_to_label(risk_type: str, score: float) -> str:
    thresholds = RISK_THRESHOLDS.get(risk_type, {})
    label = thresholds.get("low", (0, "⚪ Unknown"))[1]
    for level in ["medium", "high", "critical"]:
        if score >= thresholds[level][0]:
            label = thresholds[level][1]
    return label


def sentiment_to_label(sentiment: float) -> str:
    if sentiment >= 0.5:  return "🟢 Very positive"
    if sentiment >= 0.2:  return "🟢 Positive"
    if sentiment >= -0.2: return "⚪ Neutral"
    if sentiment >= -0.5: return "🟡 Negative"
    return "🔴 Very negative"


def news_intel_to_model_features(intel: dict) -> dict:
    """
    Convert LLM intelligence output → model input features.
    These values directly replace the injury_risk, travel_risk, fatigue_risk
    columns in the feature vector.
    """
    return {
        "injury_risk":  intel.get("injury_risk", 0.0),
        "travel_risk":  intel.get("travel_risk", 0.0),
        "fatigue_risk": intel.get("fatigue_risk", 0.0),
    }


def news_intel_to_dashboard(intel: dict) -> dict:
    """
    Format intelligence output for the React dashboard display.
    """
    injury  = intel.get("injury_risk", 0.0)
    travel  = intel.get("travel_risk", 0.0)
    fatigue = intel.get("fatigue_risk", 0.0)

    # Overall risk = weighted combination
    overall_risk = (injury * 0.5) + (travel * 0.25) + (fatigue * 0.25)

    risk_flags = []
    for risk_type, score in [("injury", injury), ("travel", travel), ("fatigue", fatigue)]:
        label = risk_to_label(risk_type, score)
        if score >= 0.10:
            risk_flags.append({
                "type":     risk_type,
                "score":    round(score * 100, 1),
                "label":    label,
                "severity": "critical" if score >= 0.75 else
                            "high"     if score >= 0.50 else
                            "medium"   if score >= 0.25 else "low",
            })

    return {
        "summary":          intel.get("summary", ""),
        "overall_risk":     round(overall_risk * 100, 1),
        "sentiment_label":  sentiment_to_label(intel.get("sentiment", 0.0)),
        "sentiment_score":  round(intel.get("sentiment", 0.0) * 100, 1),
        "confidence":       round(intel.get("confidence", 0.1) * 100, 1),
        "risk_flags":       risk_flags,
        "flags":            intel.get("flags", []),
        "positive_signals": intel.get("positive_signals", []),
    }


def batch_news_intel(horses: list[dict]) -> list[dict]:
    """
    Merge news intelligence back into a list of horse dicts.
    Each horse dict should have 'news_intel' key from the pipeline.
    Adds 'news_features' (for model) and 'news_display' (for dashboard).
    """
    enriched = []
    for horse in horses:
        intel = horse.get("news_intel", {})
        horse["news_features"] = news_intel_to_model_features(intel)
        horse["news_display"]  = news_intel_to_dashboard(intel)
        enriched.append(horse)
    return enriched