"""
Single source of truth for RaceOracle model architecture constants.

Import from here in ALL files:
    train_model.py, predictor.py, fusion_model.py, encoders.py

NOTE: CAT_CARDINALITIES[0] (track_idx) is dynamic — it depends on how many
unique courses are in your real dataset. It is saved to
data/models/cat_cardinalities.json by ingest_data.py and loaded at runtime
by both train_model.py and predictor.py. The value here is only a fallback.
"""

import os
import json

# ── Feature column names (order matters — must match DataFrame columns) ───────

CAT_FEATURES = [
    "track_idx",        # which racecourse — dynamic cardinality from real data
    "going_idx",        # going condition: 0=Firm 1=Good 2=Good-Soft 3=Soft 4=Heavy
    "surface_idx",      # 0=Turf 1=Synthetic 2=Dirt
    "draw_position",    # stalls draw (0 = unknown for this dataset)
    "class_drop_rise",  # 0=big drop 1=drop 2=same 3=rise 4=big rise
    "age_years",        # horse age remapped: age2→0, age3→1 ... age9→7
]

CONT_FEATURES = [
    "win_rate_last5",         # win rate from last 5 races (form string derived)
    "win_rate_career",        # career win rate (expanding, no leakage)
    "place_rate_last5",       # top-3 rate from last 5 races
    "days_since_last_run",    # days since previous race
    "going_preference_score", # how well horse performs on today's going [-1,1]
    "distance_fit_score",     # distance suitability score [-1,1]
    "jockey_win_rate",        # jockey historical win rate (expanding, no leakage)
    "trainer_win_rate",       # trainer historical win rate (expanding, no leakage)
    "jockey_trainer_combo",   # jockey+trainer partnership win rate
    "weight_carried_lbs",     # total weight in lbs (stones*14 + lbs)
    "speed_rating_last",      # RPRc/TRc from most recent race
    "speed_rating_avg3",      # rolling 3-race average speed rating
    "track_win_rate",         # horse win rate at this specific course
    "field_size",             # number of runners in this race
    "injury_risk",            # from news intelligence pipeline (0 if historical)
    "travel_risk",            # from news intelligence pipeline (0 if historical)
    "fatigue_risk",           # from news intelligence pipeline (0 if historical)
]

NUM_CONTINUOUS = len(CONT_FEATURES)   # 17


# ── Cardinalities ─────────────────────────────────────────────────────────────

# Fixed cardinalities for all features except track_idx
_FIXED_CARDINALITIES = {
    "going_idx":        5,   # Firm/Good/Good-Soft/Soft/Heavy
    "surface_idx":      3,   # Turf/Synthetic/Dirt
    "draw_position":    1,   # all 0 (unknown in current dataset)
    "class_drop_rise":  5,   # 0-4
    "age_years":        8,   # 0-7  (age 2→9)
}


def load_cat_cardinalities(model_dir: str = "./data/models") -> list:
    """
    Load CAT_CARDINALITIES with the real track_idx cardinality from disk.
    Falls back to 50 courses if the file doesn't exist yet.

    Returns list in CAT_FEATURES order:
        [n_courses, 5, 3, 1, 5, 8]
    """
    card_path = os.path.join(model_dir, "cat_cardinalities.json")
    if os.path.exists(card_path):
        with open(card_path) as f:
            return json.load(f)

    # Fallback: ingest_data.py hasn't been run yet
    return [
        50,  # track_idx  — placeholder; real value set by ingest_data.py
        _FIXED_CARDINALITIES["going_idx"],
        _FIXED_CARDINALITIES["surface_idx"],
        _FIXED_CARDINALITIES["draw_position"],
        _FIXED_CARDINALITIES["class_drop_rise"],
        _FIXED_CARDINALITIES["age_years"],
    ]


# Default export used by files that don't need dynamic loading
# (e.g. quick tests, smoke checks)
CAT_CARDINALITIES = load_cat_cardinalities()


# ── TabTransformer architecture ───────────────────────────────────────────────

TAB_D_MODEL  = 64    # embedding dimension per feature
TAB_N_LAYERS = 8     # total transformer blocks (first 6 frozen, last 2 surgical)
TAB_OUT_DIM  = 128   # output embedding dimension → fusion layer


# ── Fusion model ──────────────────────────────────────────────────────────────

FUSED_DIM    = 256   # cross-modal attention output dimension


# ── Surgery config ────────────────────────────────────────────────────────────

FREEZE_DEPTH      = 6    # freeze layers 0..5, operate on layers 6..7
N_SURGICAL_HEADS  = 4    # race-context attention heads (replaces 8 generic)
N_RISK_HEADS      = 3    # SBERT risk heads: injury / travel / fatigue