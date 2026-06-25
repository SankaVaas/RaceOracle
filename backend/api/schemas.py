"""
Pydantic schemas — request and response contracts for the FastAPI endpoints.
These define exactly what the React frontend sends and receives.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request schemas ───────────────────────────────────────────────────────────

class HorseInput(BaseModel):
    name:               str
    trainer:            str             = ""
    jockey:             str             = ""
    form:               str             = "-"
    age:                int             = Field(default=4, ge=2, le=15)
    weight_lbs:         float           = Field(default=126, ge=80, le=200)
    draw:               int             = Field(default=0, ge=0, le=40)
    days_since_last_run:float           = Field(default=21, ge=0, le=365)
    jockey_win_rate:    float           = Field(default=0.12, ge=0, le=1)
    trainer_win_rate:   float           = Field(default=0.12, ge=0, le=1)
    speed_ratings:      list[float]     = []
    going_history:      list[list]      = []   # [[going_str, finish_pos], ...]
    distance_history:   list[list]      = []   # [[distance_f, finish_pos], ...]
    track_wins:         int             = 0
    track_runs:         int             = 1
    career_wins:        int             = 0
    career_runs:        int             = 1
    xgb_win_prob:       Optional[float] = None
    class_change:       int             = Field(default=0, ge=-2, le=2)
    injury_risk:        float           = Field(default=0.0, ge=0, le=1)
    travel_risk:        float           = Field(default=0.0, ge=0, le=1)
    fatigue_risk:       float           = Field(default=0.0, ge=0, le=1)
    news_text:          str             = ""


class RaceContextInput(BaseModel):
    race_id:            str             = "unknown"
    race_name:          str             = "Unnamed Race"
    track:              str             = "Unknown"
    going:              str             = "Good"
    surface:            str             = "Turf"
    distance_furlongs:  float           = Field(default=10.0, ge=2, le=30)


class PredictRequest(BaseModel):
    horses:             list[HorseInput]
    race:               RaceContextInput
    fetch_news:         bool            = False   # if True, run news intelligence pipeline


class NewsRequest(BaseModel):
    horse_name:         str
    trainer_name:       str             = ""
    race_info:          str             = "upcoming race"
    use_cache:          bool            = True


# ── Response schemas ──────────────────────────────────────────────────────────

class RiskFlag(BaseModel):
    type:       str
    severity:   str
    label:      str
    score:      float


class ShapEntry(BaseModel):
    feature:    str
    impact:     float


class ModalWeights(BaseModel):
    structured: float
    odds:       float
    news:       float


class HorsePrediction(BaseModel):
    rank:           int
    name:           str
    form:           str
    jockey:         str
    trainer:        str
    age:            int
    draw:           int
    win_prob:       float
    top2_prob:      float
    top3_prob:      float
    confidence:     float
    modal_weights:  ModalWeights
    risk_flags:     list[RiskFlag]
    top_drivers:    list[ShapEntry]
    top_risks:      list[ShapEntry]
    news_summary:   str             = ""


class PredictResponse(BaseModel):
    race_id:            str
    race_name:          str
    track:              str
    going:              str
    surface:            str
    distance:           float
    field_size:         int
    model_version:      str
    inference_ms:       float
    top_pick:           str
    model_confidence:   float
    horses:             list[HorsePrediction]


class NewsIntelResponse(BaseModel):
    horse_name:         str
    summary:            str
    injury_risk:        float
    travel_risk:        float
    fatigue_risk:       float
    sentiment:          float
    confidence:         float
    flags:              list[str]
    positive_signals:   list[str]
    overall_risk:       float
    sentiment_label:    str


class HealthResponse(BaseModel):
    status:             str
    model_loaded:       bool
    scaler_fitted:      bool
    model_version:      str
    device:             str
    trainable_params:   dict