"""
End-to-end integration test for the full inference pipeline.
Run: python -m pytest tests/integration/test_predictor_pipeline.py -v
  or: python tests/integration/test_predictor_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from backend.pipeline.feature_engineering.features import (
    parse_form_string, compute_going_preference,
    compute_distance_fit, compute_recency_weights, build_race_features
)
from backend.pipeline.feature_engineering.encoders import RaceFeatureEncoder
from backend.model.inference.predictor import RacePredictor


# ── Realistic mock race data ────────────────────────────────────────────────

RACE_CONTEXT = {
    "race_id":           "ascot_r3_20250622",
    "race_name":         "Royal Ascot — Race 3 (1m4f)",
    "track":             "Royal Ascot",
    "going":             "Good",
    "surface":           "Turf",
    "distance_furlongs": 14.0,
}

HORSES_RAW = [
    {
        "name": "Thunderstrike",
        "form": "1-1-2-1-3",
        "age": 5, "weight_lbs": 126, "draw": 1,
        "days_since_last_run": 14,
        "jockey": "F. Dettori", "jockey_id": "dettori",
        "trainer": "J. Gosden",  "trainer_id": "gosden",
        "jockey_win_rate": 0.21, "trainer_win_rate": 0.18,
        "speed_ratings": [112, 115, 111, 118, 116],
        "going_history": [("Good", 1), ("Good", 1), ("Firm", 2), ("Good-Soft", 3)],
        "distance_history": [(14, 1), (12, 1), (14, 2), (16, 3)],
        "track_wins": 3, "track_runs": 5,
        "career_wins": 7, "career_runs": 11,
        "xgb_win_prob": 0.34,
        "class_change": 0,
        "injury_risk": 0.05,
        "travel_risk": 0.02,
        "fatigue_risk": 0.08,
        "news_text": "Thunderstrike looked outstanding in morning gallops. Trainer Gosden confident ahead of Ascot.",
    },
    {
        "name": "Silver Arrow",
        "form": "2-1-3-2-1",
        "age": 4, "weight_lbs": 130, "draw": 2,
        "days_since_last_run": 21,
        "jockey": "R. Moore", "jockey_id": "moore",
        "trainer": "A. O'Brien", "trainer_id": "obrien",
        "jockey_win_rate": 0.23, "trainer_win_rate": 0.20,
        "speed_ratings": [108, 113, 110, 114, 112],
        "going_history": [("Good", 2), ("Firm", 1), ("Good", 3)],
        "distance_history": [(14, 2), (14, 1), (12, 2)],
        "track_wins": 2, "track_runs": 4,
        "career_wins": 5, "career_runs": 12,
        "xgb_win_prob": 0.28,
        "class_change": 1,
        "injury_risk": 0.08,
        "travel_risk": 0.15,
        "fatigue_risk": 0.12,
        "news_text": "Connections happy but concerned about weight allocation. Travelled from Ireland yesterday.",
    },
    {
        "name": "Dark Horizon",
        "form": "3-2-1-4-5",
        "age": 6, "weight_lbs": 124, "draw": 3,
        "days_since_last_run": 42,
        "jockey": "W. Buick", "jockey_id": "buick",
        "trainer": "C. Appleby", "trainer_id": "appleby",
        "jockey_win_rate": 0.19, "trainer_win_rate": 0.17,
        "speed_ratings": [114, 110, 116, 104, 102],
        "going_history": [("Good-Soft", 1), ("Soft", 2), ("Good", 4)],
        "distance_history": [(14, 1), (16, 2), (12, 4)],
        "track_wins": 1, "track_runs": 6,
        "career_wins": 4, "career_runs": 14,
        "xgb_win_prob": 0.18,
        "class_change": -1,
        "injury_risk": 0.62,   # high — tendon issue reported
        "travel_risk": 0.05,
        "fatigue_risk": 0.25,
        "news_text": "CONCERN: Tendon inflammation reported after last workout. Trainer downplayed but vet seen.",
    },
    {
        "name": "Morning Glory",
        "form": "4-5-3-3-4",
        "age": 5, "weight_lbs": 122, "draw": 4,
        "days_since_last_run": 10,
        "jockey": "H. Doyle", "jockey_id": "doyle",
        "trainer": "M. Johnston", "trainer_id": "johnston",
        "jockey_win_rate": 0.14, "trainer_win_rate": 0.13,
        "speed_ratings": [98, 100, 102, 99, 101],
        "going_history": [("Good", 3), ("Good", 4), ("Firm", 5)],
        "distance_history": [(14, 3), (12, 4), (14, 5)],
        "track_wins": 0, "track_runs": 3,
        "career_wins": 3, "career_runs": 18,
        "xgb_win_prob": 0.11,
        "class_change": -2,
        "injury_risk": 0.04,
        "travel_risk": 0.03,
        "fatigue_risk": 0.10,
        "news_text": "Dropped in class today — potential outsider value. Trainer expecting improvement.",
    },
]


def test_form_parser():
    print("\n── test_form_parser ──")
    stats = parse_form_string("1-1-2-1-3")
    assert stats["win_rate_last5"] == 0.6,  f"Expected 0.6, got {stats['win_rate_last5']}"
    assert stats["place_rate_last5"] == 1.0, f"Expected 1.0, got {stats['place_rate_last5']}"
    print(f"  win_rate_last5  : {stats['win_rate_last5']}")
    print(f"  place_rate_last5: {stats['place_rate_last5']}")
    print(f"  form_momentum   : {stats['form_momentum']:.3f}")
    print("  ✅ PASSED")


def test_going_preference():
    print("\n── test_going_preference ──")
    history = [("Good", 1), ("Good", 1), ("Soft", 4), ("Heavy", 5)]
    score = compute_going_preference(history, "Good")
    assert score > 0, f"Good preference expected positive, got {score}"
    bad_score = compute_going_preference(history, "Heavy")
    assert bad_score < 0, f"Heavy preference expected negative, got {bad_score}"
    print(f"  Going=Good  score: {score:.3f}  ✅")
    print(f"  Going=Heavy score: {bad_score:.3f}  ✅")


def test_recency_weights():
    print("\n── test_recency_weights ──")
    fresh  = compute_recency_weights(7,  seq_len=7)
    stale  = compute_recency_weights(60, seq_len=7)
    assert fresh.max() > stale.max(), "Fresh horse should have higher weights"
    assert fresh.shape == (7,)
    print(f"  Fresh horse  (7 days) weights: {fresh.round(3)}")
    print(f"  Stale horse (60 days) weights: {stale.round(3)}")
    print("  ✅ PASSED")


def test_build_race_features():
    print("\n── test_build_race_features ──")
    feats = build_race_features(HORSES_RAW, RACE_CONTEXT)
    assert len(feats) == 4
    for i, f in enumerate(feats):
        assert len(f["cat_values"])  == 6,  f"Horse {i}: expected 6 cat features"
        assert len(f["cont_values"]) == 17, f"Horse {i}: expected 17 cont features"
        assert f["recency_weights"].shape == (7,)
        print(f"  {f['name']:20s} | win_rate_last5={f['cont_values'][0]:.2f} | "
              f"going_score={f['cont_values'][4]:.2f} | injury={f['cont_values'][14]:.2f}")
    print("  ✅ PASSED")


def test_encoder():
    print("\n── test_encoder ──")
    feats   = build_race_features(HORSES_RAW, RACE_CONTEXT)
    encoder = RaceFeatureEncoder(scaler_path="./data/models/scaler_test.pkl")
    encoder.fit(feats)
    encoded = encoder.encode_race(feats)
    assert len(encoded) == 4
    for enc in encoded:
        assert enc["x_cat"].shape  == (1, 6),  f"x_cat shape wrong: {enc['x_cat'].shape}"
        assert enc["x_cont"].shape == (1, 17), f"x_cont shape wrong: {enc['x_cont'].shape}"
        assert enc["recency_weights"].shape == (1, 7, 1)
        print(f"  {enc['name']:20s} x_cat{tuple(enc['x_cat'].shape)}  "
              f"x_cont{tuple(enc['x_cont'].shape)}  ✅")
    print("  ✅ PASSED")


def test_full_predictor():
    print("\n── test_full_predictor (end-to-end) ──")
    predictor = RacePredictor()
    result    = predictor.predict(HORSES_RAW, RACE_CONTEXT, use_cache=False)

    assert "horses" in result
    assert len(result["horses"]) == 4
    total_prob = sum(h["win_prob"] for h in result["horses"])
    assert abs(total_prob - 100.0) < 1.0, f"Win probs should sum to ~100, got {total_prob}"

    print(f"\n  Race     : {result['race_name']}")
    print(f"  Track    : {result['track']} | Going: {result['going']}")
    print(f"  Infer ms : {result['inference_ms']}ms")
    print(f"  Top pick : {result['top_pick']}\n")
    print(f"  {'Rank':<5} {'Horse':<20} {'Win%':>6} {'Top2%':>6} {'Top3%':>6} "
          f"{'Conf%':>6} {'Inj':>6} {'Modal weights (S/O/N)'}")
    print(f"  {'-'*80}")
    for h in result["horses"]:
        mw = h["modal_weights"]
        flags = " ".join(f["label"] for f in h["risk_flags"]) if h["risk_flags"] else "–"
        print(f"  {h['rank']:<5} {h['name']:<20} {h['win_prob']:>5.1f}% "
              f"{h['top2_prob']:>5.1f}% {h['top3_prob']:>5.1f}% "
              f"{h['confidence']:>5.1f}%  "
              f"S:{mw['structured']:>4.0f}% O:{mw['odds']:>4.0f}% N:{mw['news']:>4.0f}%  {flags}")

    print(f"\n  Win prob sum: {total_prob:.1f}% ✅")
    print("  ✅ FULL PIPELINE PASSED")
    return result


if __name__ == "__main__":
    import os
    os.makedirs("./data/models", exist_ok=True)
    os.makedirs("./data/cache",  exist_ok=True)

    test_form_parser()
    test_going_preference()
    test_recency_weights()
    test_build_race_features()
    test_encoder()
    result = test_full_predictor()

    print("\n" + "═"*60)
    print("ALL TESTS PASSED — inference pipeline fully operational")
    print("═"*60)
