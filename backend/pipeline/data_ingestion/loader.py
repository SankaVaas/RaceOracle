"""
Real data ingestion pipeline for RaceOracle.

Reads the actual horse racing CSV (with columns:
  course, countryCode, marketTime, title, runners, condition,
  prize, rclass, horseName, trainerName, jockeyName,
  RPRc, TRc, OR, weightSt, weightLb, age, decimalPrice)

And produces a clean feature-engineered DataFrame ready for training.

Key derivations:
  - win label        : lowest decimalPrice in each race = winner (favourite proxy)
  - finish position  : rank by decimalPrice within each race
  - form string      : built per horse by looking at their historical races chronologically
  - weight_lbs       : weightSt * 14 + weightLb
  - going category   : mapped from raw condition strings to standard 5-class going
  - win/place rates  : computed per jockey and trainer from historical data
  - speed proxy      : RPRc / TRc ratings used as speed figures
"""

import pandas as pd
import numpy as np
import os
import hashlib
from datetime import datetime
from typing import Optional
from backend.utils.logger import logger


# ── Going condition normaliser ────────────────────────────────────────────────
# Racing Post uses many variations — map all to 5 standard classes

GOING_MAP = {
    # Firm
    "firm": "Firm", "hard": "Firm", "fast": "Firm",

    # Good
    "good": "Good", "standard": "Good", "good to firm": "Good",
    "good to fast": "Good",

    # Good-Soft
    "good to soft": "Good-Soft", "good to yielding": "Good-Soft",
    "yielding to good": "Good-Soft", "soft to good": "Good-Soft",

    # Soft
    "soft": "Soft", "yielding": "Soft",
    "yielding to soft": "Soft", "soft to yielding": "Soft",

    # Heavy
    "heavy": "Heavy", "very soft": "Heavy", "muddy": "Heavy",
    "boggy": "Heavy", "slow": "Heavy",
}

GOING_IDX = {"Firm": 0, "Good": 1, "Good-Soft": 2, "Soft": 3, "Heavy": 4}


def normalise_going(condition: str) -> str:
    if not isinstance(condition, str):
        return "Good"
    key = condition.strip().lower()
    # Exact match first
    if key in GOING_MAP:
        return GOING_MAP[key]
    # Partial match
    for k, v in GOING_MAP.items():
        if k in key:
            return v
    return "Good"  # default


# ── Course → index mapping (built from data) ─────────────────────────────────

def build_course_index(courses: pd.Series) -> dict:
    unique_courses = sorted(courses.dropna().unique())
    return {c: i for i, c in enumerate(unique_courses)}


# ── Weight conversion ─────────────────────────────────────────────────────────

def weight_to_lbs(stones: float, lbs: float) -> float:
    return float(stones) * 14.0 + float(lbs)


# ── Implied probability from decimal odds ─────────────────────────────────────

def implied_prob(decimal_price: float) -> float:
    """Convert decimal odds to implied win probability."""
    if decimal_price <= 1.0:
        return 0.99
    return 1.0 / decimal_price


# ── Form string builder ────────────────────────────────────────────────────────

def build_form_strings(df: pd.DataFrame) -> pd.Series:
    """
    For each row, build the form string from the horse's PREVIOUS races
    (not including the current one — no data leakage).

    Since we don't have actual finish positions, we use rank by decimalPrice
    within each race as a proxy: lowest odds = rank 1 (best).
    """
    logger.info("Building form strings from historical race order...")

    # Sort all races chronologically
    df = df.sort_values("marketTime").reset_index(drop=True)

    # Within each race, rank horses by decimalPrice (lower price = better rank)
    df["finish_pos_proxy"] = df.groupby("race_id")["decimalPrice"].rank(method="min").astype(int)

    # Build form per horse: collect their previous finish positions in order
    horse_history: dict = {}  # horseName -> list of finish positions (chronological)
    form_strings = []

    for _, row in df.iterrows():
        horse = row["horseName"]
        history = horse_history.get(horse, [])

        # Form = last 6 results, most recent last, as "4-2-1-3" style
        if len(history) == 0:
            form_str = "-"
        else:
            recent = history[-6:]
            form_str = "-".join(str(p) for p in recent)

        form_strings.append(form_str)

        # Now add this race's result to history
        history.append(row["finish_pos_proxy"])
        horse_history[horse] = history

    return pd.Series(form_strings, index=df.index)


# ── Win/place rate calculators ─────────────────────────────────────────────────

def compute_entity_rates(df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """
    For each row, compute the entity's win rate and place rate
    from ALL their previous races (strictly before current race date).
    Uses expanding window to avoid leakage.
    """
    df = df.sort_values("marketTime").copy()

    win_rates   = []
    place_rates = []
    entity_stats: dict = {}  # entity -> {"wins": int, "places": int, "runs": int}

    for _, row in df.iterrows():
        entity = row[entity_col]
        stats  = entity_stats.get(entity, {"wins": 0, "places": 0, "runs": 0})

        if stats["runs"] >= 5:
            win_rates.append(stats["wins"] / stats["runs"])
            place_rates.append(stats["places"] / stats["runs"])
        else:
            # Prior: use population average (~12% win rate for jockeys/trainers)
            win_rates.append(0.12)
            place_rates.append(0.35)

        # Update stats with this race
        pos = row.get("finish_pos_proxy", 99)
        stats["runs"]   += 1
        stats["wins"]   += 1 if pos == 1 else 0
        stats["places"] += 1 if pos <= 3 else 0
        entity_stats[entity] = stats

    df[f"{entity_col}_win_rate"]   = win_rates
    df[f"{entity_col}_place_rate"] = place_rates
    return df


# ── Main ingestion function ────────────────────────────────────────────────────

def ingest_real_data(csv_path: str, output_path: str = "./data/processed/horse_races.csv") -> pd.DataFrame:
    """
    Full ingestion pipeline: reads real CSV → outputs model-ready feature CSV.

    Args:
        csv_path   : path to the raw CSV file
        output_path: where to save the processed features

    Returns:
        Processed DataFrame ready for train_model.py
    """
    logger.info(f"Loading raw data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["marketTime"])
    logger.info(f"Loaded {len(df):,} rows, {df['horseName'].nunique():,} unique horses, "
                f"{df['course'].nunique()} courses")

    # ── 1. Basic cleaning ─────────────────────────────────────────────────
    df["horseName"]   = df["horseName"].fillna("Unknown").str.strip()
    df["trainerName"] = df["trainerName"].fillna("Unknown").str.strip()
    df["jockeyName"]  = df["jockeyName"].fillna("Unknown").str.strip()
    df["course"]      = df["course"].fillna("Unknown").str.strip()
    df["age"]         = pd.to_numeric(df["age"], errors="coerce").fillna(4).clip(2, 15)
    df["decimalPrice"]= pd.to_numeric(df["decimalPrice"], errors="coerce").fillna(10.0).clip(1.01, 500)
    df["RPRc"]        = pd.to_numeric(df["RPRc"], errors="coerce")
    df["TRc"]         = pd.to_numeric(df["TRc"],  errors="coerce")
    df["OR"]          = pd.to_numeric(df["OR"],   errors="coerce")
    df["runners"]     = pd.to_numeric(df["runners"], errors="coerce").fillna(8).clip(2, 40)
    df["prize"]       = pd.to_numeric(df["prize"],   errors="coerce").fillna(5000)
    df["weightSt"]    = pd.to_numeric(df["weightSt"], errors="coerce").fillna(9)
    df["weightLb"]    = pd.to_numeric(df["weightLb"], errors="coerce").fillna(0)

    # ── 2. Parse datetime properly ─────────────────────────────────────────
    df["marketTime"] = pd.to_datetime(df["marketTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["marketTime"])
    df = df.sort_values("marketTime").reset_index(drop=True)

    # ── 3. Build race_id (unique per race) ─────────────────────────────────
    df["race_id"] = (df["course"] + "_" + df["marketTime"].astype(str) + "_" + df["title"]).apply(
        lambda x: hashlib.md5(x.encode()).hexdigest()[:10]
    )

    # ── 4. Going normalisation ─────────────────────────────────────────────
    df["going"]     = df["condition"].apply(normalise_going)
    df["going_idx"] = df["going"].map(GOING_IDX).fillna(1).astype(int)

    # ── 5. Course index ────────────────────────────────────────────────────
    course_map      = build_course_index(df["course"])
    df["track_idx"] = df["course"].map(course_map).fillna(0).astype(int)
    n_courses       = len(course_map)
    logger.info(f"Indexed {n_courses} unique courses")

    # ── 6. Surface inference (most UK/IRE races are Turf; AW tracks listed) ─
    all_weather_keywords = ["kempton", "lingfield", "wolverhampton", "chelmsford",
                             "southwell", "dunstall", "newcastle"]
    df["surface"]     = df["course"].str.lower().apply(
        lambda c: "Synthetic" if any(k in c for k in all_weather_keywords) else "Turf"
    )
    df["surface_idx"] = df["surface"].map({"Turf": 0, "Synthetic": 1, "Dirt": 2}).fillna(0).astype(int)

    # ── 7. Weight in lbs ──────────────────────────────────────────────────
    df["weight_lbs"] = df.apply(lambda r: weight_to_lbs(r["weightSt"], r["weightLb"]), axis=1)

    # ── 8. Implied win probability from odds ──────────────────────────────
    df["implied_win_prob"] = df["decimalPrice"].apply(implied_prob)

    # ── 9. Finish position proxy (rank within race by odds) ───────────────
    df["finish_pos_proxy"] = df.groupby("race_id")["decimalPrice"].rank(method="min").astype(int)
    df["won"]              = (df["finish_pos_proxy"] == 1).astype(int)

    # ── 10. Form strings (from historical results, no leakage) ────────────
    df["form"] = build_form_strings(df)

    # ── 11. Win/place rates per jockey and trainer ────────────────────────
    logger.info("Computing jockey win rates (expanding window, no leakage)...")
    df = compute_entity_rates(df, "jockeyName")
    logger.info("Computing trainer win rates (expanding window, no leakage)...")
    df = compute_entity_rates(df, "trainerName")

    # ── 12. Speed ratings — use RPRc as primary, TRc as secondary ─────────
    # Fill missing with within-race median, then global median
    df["speed_rating"] = df["RPRc"].fillna(df["TRc"]).fillna(df["OR"])
    race_median        = df.groupby("race_id")["speed_rating"].transform("median")
    df["speed_rating"] = df["speed_rating"].fillna(race_median)
    global_median      = df["speed_rating"].median()
    df["speed_rating"] = df["speed_rating"].fillna(global_median if not np.isnan(global_median) else 85.0)

    # Rolling average speed per horse (last 3 runs, no leakage)
    df = df.sort_values("marketTime")
    df["speed_avg3"] = (
        df.groupby("horseName")["speed_rating"]
          .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    ).fillna(df["speed_rating"])

    # ── 13. Days since last run per horse ─────────────────────────────────
    df["prev_race_time"] = df.groupby("horseName")["marketTime"].shift(1)
    df["days_since_last_run"] = (
        (df["marketTime"] - df["prev_race_time"])
        .dt.total_seconds() / 86400
    ).clip(0, 365).fillna(30.0)  # default 30 days for first run

    # ── 14. Win rate for horse at this course (historical) ────────────────
    df = df.sort_values("marketTime")
    df["track_win"] = (df["finish_pos_proxy"] == 1).astype(int)
    df["track_win_rate"] = (
        df.groupby(["horseName", "course"])["track_win"]
          .transform(lambda x: x.shift(1).expanding().mean())
    ).fillna(0.1)

    # ── 15. Parse form string into win/place rates ─────────────────────────
    def form_to_rates(form_str: str):
        if not isinstance(form_str, str) or form_str == "-":
            return 0.0, 0.0
        positions = []
        for token in form_str.split("-"):
            try:
                positions.append(int(token))
            except ValueError:
                positions.append(10)
        if not positions:
            return 0.0, 0.0
        n = len(positions)
        return (
            sum(1 for p in positions if p == 1) / n,
            sum(1 for p in positions if p <= 3) / n,
        )

    rates = df["form"].apply(form_to_rates)
    df["win_rate_last5"]   = rates.apply(lambda x: x[0])
    df["place_rate_last5"] = rates.apply(lambda x: x[1])

    # ── 16. Career win rate (all historical races for this horse) ─────────
    df["career_win"] = (df["finish_pos_proxy"] == 1).astype(int)
    df["win_rate_career"] = (
        df.groupby("horseName")["career_win"]
          .transform(lambda x: x.shift(1).expanding().mean())
    ).fillna(df["win_rate_last5"])

    # ── 17. Class drop/rise (prize money comparison) ──────────────────────
    df["prev_prize"] = df.groupby("horseName")["prize"].shift(1).fillna(df["prize"])
    df["class_change_raw"] = np.log1p(df["prize"]) - np.log1p(df["prev_prize"])
    df["class_drop_rise"] = pd.cut(
        df["class_change_raw"],
        bins=[-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2).astype(int)  # 2 = no change

    # ── 18. Going preference score ─────────────────────────────────────────
    # Compare average finish position on today's going vs other going types
    df["going_group"] = df["going"].map({
        "Firm": "fast", "Good": "fast",
        "Good-Soft": "soft", "Soft": "soft", "Heavy": "heavy"
    })
    def going_pref_score(group: pd.DataFrame) -> pd.Series:
        scores = []
        for _, row in group.iterrows():
            horse_hist = df[
                (df["horseName"] == row["horseName"]) &
                (df["marketTime"] < row["marketTime"])
            ]
            if len(horse_hist) < 3:
                scores.append(0.0)
                continue
            same   = horse_hist[horse_hist["going_group"] == row["going_group"]]["finish_pos_proxy"]
            other  = horse_hist[horse_hist["going_group"] != row["going_group"]]["finish_pos_proxy"]
            if len(same) == 0:
                scores.append(0.0)
            elif len(other) == 0:
                scores.append(0.1)
            else:
                score = (other.mean() - same.mean()) / max(other.mean(), 1.0)
                scores.append(float(np.clip(score, -1.0, 1.0)))
        return pd.Series(scores, index=group.index)

    logger.info("Computing going preference scores (this may take a moment)...")
    df["going_preference_score"] = 0.0
    sample_mask = df.groupby("horseName").transform("count")["course"] >= 3
    if sample_mask.sum() > 0:
        df.loc[sample_mask, "going_preference_score"] = going_pref_score(df[sample_mask])

    # ── 19. Distance fit (using race title length as crude proxy) ─────────
    # Since we don't have distance in furlongs, we use prize/class as distance proxy
    # and set distance_fit_score to 0 (neutral) — can be improved with real distance data
    df["distance_fit_score"] = 0.0

    # ── 20. Age index ──────────────────────────────────────────────────────
    df["age_idx"] = (df["age"].clip(2, 9) - 2).astype(int)  # 0-7

    # ── 21. Draw (position in stalls) — not in CSV, use 0 (unknown) ───────
    df["draw_position"] = 0  # neutral embedding index

    # ── 22. Jockey-trainer combo ───────────────────────────────────────────
    df["jt_combo"] = df["jockeyName"] + "___" + df["trainerName"]
    jt_win_rates = (
        df.groupby("jt_combo")["career_win"]
          .transform(lambda x: x.shift(1).expanding().mean())
    ).fillna(0.1)
    df["jockey_trainer_combo"] = jt_win_rates

    # ── 23. Risk scores (set to 0 for historical data — news pipeline fills these live)
    df["injury_risk"]  = 0.0
    df["travel_risk"]  = 0.0
    df["fatigue_risk"] = 0.0

    # ── 24. XGBoost win prob proxy (implied probability normalised within race)
    df["xgb_win_prob"] = df.groupby("race_id")["implied_win_prob"].transform(
        lambda x: x / x.sum()
    ).clip(0.01, 0.99)

    # ── 25. Select and rename final feature columns ────────────────────────
    out = pd.DataFrame({
        # Identifiers
        "race_id":              df["race_id"],
        "horse_name":           df["horseName"],
        "jockey":               df["jockeyName"],
        "trainer":              df["trainerName"],
        "course":               df["course"],
        "race_date":            df["marketTime"].dt.date.astype(str),
        "going":                df["going"],

        # Target
        "won":                  df["won"],
        "finish_pos_proxy":     df["finish_pos_proxy"],

        # Categorical features (model_config CAT_FEATURES order)
        "track_idx":            df["track_idx"].clip(0, 999),
        "going_idx":            df["going_idx"],
        "surface_idx":          df["surface_idx"],
        "draw_position":        df["draw_position"],
        "class_drop_rise":      df["class_drop_rise"],
        "age_years":            df["age_idx"],

        # Continuous features (model_config CONT_FEATURES order)
        "win_rate_last5":       df["win_rate_last5"],
        "win_rate_career":      df["win_rate_career"],
        "place_rate_last5":     df["place_rate_last5"],
        "days_since_last_run":  df["days_since_last_run"],
        "going_preference_score": df["going_preference_score"],
        "distance_fit_score":   df["distance_fit_score"],
        "jockey_win_rate":      df["jockeyName_win_rate"],
        "trainer_win_rate":     df["trainerName_win_rate"],
        "jockey_trainer_combo": df["jockey_trainer_combo"],
        "weight_carried_lbs":   df["weight_lbs"],
        "speed_rating_last":    df["speed_rating"],
        "speed_rating_avg3":    df["speed_avg3"],
        "track_win_rate":       df["track_win_rate"],
        "field_size":           df["runners"],
        "injury_risk":          df["injury_risk"],
        "travel_risk":          df["travel_risk"],
        "fatigue_risk":         df["fatigue_risk"],

        # XGBoost proxy
        "xgb_win_prob":         df["xgb_win_prob"],

        # Extras kept for analysis
        "form":                 df["form"],
        "rpr":                  df["RPRc"],
        "tr":                   df["TRc"],
        "official_rating":      df["OR"],
        "decimal_price":        df["decimalPrice"],
        "prize":                df["prize"],
        "field_size_raw":       df["runners"],
    })

    # Drop rows with any NaN in key features
    key_cols = ["won", "track_idx", "going_idx", "win_rate_last5",
                "speed_rating_last", "days_since_last_run"]
    before = len(out)
    out = out.dropna(subset=key_cols).reset_index(drop=True)
    logger.info(f"Dropped {before - len(out)} rows with missing key features")

    # ── 26. Save ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)

    logger.info(f"✅ Processed {len(out):,} rows → {output_path}")
    logger.info(f"   Races     : {out['race_id'].nunique():,}")
    logger.info(f"   Horses    : {out['horse_name'].nunique():,}")
    logger.info(f"   Courses   : {out['course'].nunique()}")
    logger.info(f"   Win rate  : {out['won'].mean():.3f} (expected ~1/avg_field_size)")
    logger.info(f"   Date range: {out['race_date'].min()} → {out['race_date'].max()}")

    return out, course_map


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "./data/raw/horse_races.csv"
    df, course_map = ingest_real_data(csv_path)
    print("\nFeature preview:")
    print(df[["horse_name", "course", "going", "win_rate_last5",
              "speed_rating_last", "days_since_last_run", "won"]].head(10).to_string())