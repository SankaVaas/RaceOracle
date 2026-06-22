"""
Generates synthetic horse racing data for training and testing.
Run: python scripts/generate_mock_data.py
"""

import numpy as np
import pandas as pd
import os, json, random
from datetime import datetime, timedelta

TRACKS     = ["Royal Ascot", "Cheltenham", "Newmarket", "Epsom", "Goodwood"]
GOING      = ["Firm", "Good", "Good-Soft", "Soft", "Heavy"]
SURFACES   = ["Turf", "Dirt", "Synthetic"]
DISTANCES  = [5, 6, 7, 8, 10, 12, 14, 16]   # furlongs


def generate_horse_race_dataset(n_races: int = 2000, horses_per_race: int = 8,
                                 output_dir: str = "./data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for race_id in range(n_races):
        track    = random.choice(TRACKS)
        going    = random.choice(GOING)
        surface  = random.choice(SURFACES)
        distance = random.choice(DISTANCES)
        n_horses = random.randint(4, horses_per_race)

        # True latent ability for each horse in this race
        true_abilities = np.random.randn(n_horses)
        winner_idx     = np.argmax(true_abilities + np.random.randn(n_horses) * 0.5)

        for horse_idx in range(n_horses):
            ability = true_abilities[horse_idx]
            won     = (horse_idx == winner_idx)
            finish  = 1 if won else (horse_idx % (n_horses - 1)) + 2

            record = {
                "race_id":               race_id,
                "horse_id":              f"H{race_id:04d}_{horse_idx}",
                "track":                 track,
                "going":                 going,
                "surface":               surface,
                "distance_furlongs":     distance,
                "field_size":            n_horses,
                "draw_position":         horse_idx + 1,
                "finish_position":       finish,
                "won":                   int(won),

                # Structured features (normalised around real distributions)
                "win_rate_last5":        max(0, min(1, (ability + np.random.randn() * 0.3 + 0.5) / 2)),
                "win_rate_career":       max(0, min(1, (ability + np.random.randn() * 0.4 + 0.5) / 2.5)),
                "place_rate_last5":      max(0, min(1, (ability + np.random.randn() * 0.2 + 0.8) / 1.8)),
                "days_since_last_run":   max(7, int(np.random.exponential(28))),
                "going_preference_score": np.clip(ability * 0.3 + np.random.randn() * 0.2, -1, 1),
                "distance_fit_score":    np.clip(ability * 0.25 + np.random.randn() * 0.2, -1, 1),
                "jockey_win_rate":       max(0, min(0.3, np.random.beta(2, 8))),
                "trainer_win_rate":      max(0, min(0.3, np.random.beta(2, 7))),
                "jockey_trainer_combo":  np.random.beta(1, 5),
                "weight_carried_lbs":    int(np.random.normal(126, 8)),
                "speed_rating_last":     max(60, min(130, ability * 15 + 95 + np.random.randn() * 5)),
                "speed_rating_avg3":     max(60, min(130, ability * 12 + 92 + np.random.randn() * 4)),
                "class_drop_rise":       np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.2, 0.4, 0.2, 0.1]),
                "track_win_rate":        max(0, np.random.beta(1, 6)),
                "age_years":             random.randint(2, 9),
                "xgb_win_prob":          max(0.01, min(0.99, 1/n_horses + ability * 0.08 + np.random.randn() * 0.05)),

                # News/risk features (synthetic)
                "injury_risk":           max(0, np.random.exponential(0.1)),
                "travel_risk":           max(0, np.random.exponential(0.05)),
                "fatigue_risk":          max(0, np.random.exponential(0.08)),

                # Categorical indices for embedding
                "track_idx":             TRACKS.index(track),
                "going_idx":             GOING.index(going),
                "surface_idx":           SURFACES.index(surface),
            }
            records.append(record)

    df = pd.DataFrame(records)
    out_path = os.path.join(output_dir, "horse_races.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Generated {len(df)} horse-race records across {n_races} races → {out_path}")
    print(f"   Win rate check: {df['won'].mean():.3f} (expected ~{1/6:.3f})")
    return df


if __name__ == "__main__":
    df = generate_horse_race_dataset(n_races=3000, horses_per_race=8)
    print(df.describe())
