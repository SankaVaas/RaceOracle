"""
FastAPI routes for RaceOracle.

Endpoints:
  GET  /health                  — model status check
  POST /predict                 — predict race outcome
  POST /news                    — fetch + analyse news for one horse
  POST /predict/with-news       — predict + fetch news in one call
  GET  /predict/demo            — demo race with mock horses (no auth needed)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.api.schemas import (
    PredictRequest, PredictResponse, NewsRequest, NewsIntelResponse, ArticleItem,
    HealthResponse, HorsePrediction, RiskFlag, ShapEntry, ModalWeights
)
from backend.model.inference.predictor import RacePredictor
from backend.pipeline.news_intelligence import enrich_race_with_news
from backend.pipeline.news_intelligence.news_fetcher import fetch_horse_news, format_articles_for_llm
from backend.pipeline.news_intelligence.llm_summarizer import summarise_horse_news
from backend.pipeline.news_intelligence.sentiment import news_intel_to_dashboard
from backend.utils.logger import logger

router    = APIRouter()
predictor = RacePredictor()   # loaded once at startup


# ── /health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Check model load status. Use before calling /predict."""
    info = predictor.health_check()
    return HealthResponse(
        status          = "ok" if info["model_loaded"] else "degraded",
        model_loaded    = info["model_loaded"],
        scaler_fitted   = info["scaler_fitted"],
        model_version   = info["model_version"],
        device          = info["device"],
        trainable_params= info["trainable_params"],
    )


# ── /predict ──────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest):
    """
    Predict win probabilities for a race.

    Send a list of horses with their stats and race context.
    Returns ranked predictions with SHAP explanations and risk flags.

    Set fetch_news=true to also run the news intelligence pipeline
    (adds ~2-5s per horse but enriches injury/travel/fatigue scores).
    """
    if len(req.horses) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 horses to predict a race.")
    if len(req.horses) > 40:
        raise HTTPException(status_code=400, detail="Maximum 40 runners per race.")

    horses_raw = [h.model_dump() for h in req.horses]
    race_ctx   = req.race.model_dump()

    # Optionally enrich with live news intelligence
    if req.fetch_news:
        race_info  = f"{race_ctx['track']}, {race_ctx['distance_furlongs']}f, {race_ctx['going']}"
        horses_raw = enrich_race_with_news(horses_raw, race_info, use_cache=True)
        for i, h in enumerate(horses_raw):
            if "news_features" in h:
                horses_raw[i].update(h["news_features"])

    try:
        result = predictor.predict(horses_raw, race_ctx, use_cache=False)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return _format_response(result)


# ── /predict/with-news ────────────────────────────────────────────────────────

@router.post("/predict/with-news", response_model=PredictResponse, tags=["prediction"])
def predict_with_news(req: PredictRequest):
    """
    Convenience endpoint: predict + fetch news in one call.
    Equivalent to POST /predict with fetch_news=true.
    """
    req.fetch_news = True
    return predict(req)


# ── /predict/demo ─────────────────────────────────────────────────────────────

@router.get("/predict/demo", response_model=PredictResponse, tags=["prediction"])
def predict_demo():
    """
    Demo prediction with a sample Royal Ascot race.
    No request body needed — great for frontend development and casino demos.
    """
    demo_horses = [
        {"name": "Thunderstrike", "trainer": "J. Gosden",   "jockey": "F. Dettori",
         "form": "1-1-2-1-3", "age": 5, "weight_lbs": 126, "draw": 1,
         "days_since_last_run": 14, "jockey_win_rate": 0.21, "trainer_win_rate": 0.18,
         "speed_ratings": [112, 115, 111, 118, 116], "career_wins": 7, "career_runs": 11,
         "going_history": [["Good", 1], ["Good", 1], ["Firm", 2]],
         "distance_history": [[14, 1], [12, 1], [14, 2]],
         "track_wins": 3, "track_runs": 5, "xgb_win_prob": 0.34,
         "injury_risk": 0.05, "travel_risk": 0.02, "fatigue_risk": 0.08,
         "news_text": "Strong gallop reported. Trainer very confident."},
        {"name": "Silver Arrow",  "trainer": "A. O'Brien",  "jockey": "R. Moore",
         "form": "2-1-3-2-1", "age": 4, "weight_lbs": 130, "draw": 2,
         "days_since_last_run": 21, "jockey_win_rate": 0.23, "trainer_win_rate": 0.20,
         "speed_ratings": [108, 113, 110, 114, 112], "career_wins": 5, "career_runs": 12,
         "going_history": [["Good", 2], ["Firm", 1]], "distance_history": [[14, 2], [14, 1]],
         "track_wins": 2, "track_runs": 4, "xgb_win_prob": 0.28,
         "injury_risk": 0.08, "travel_risk": 0.15, "fatigue_risk": 0.12,
         "news_text": "Travelled from Ireland. Connections happy despite weight."},
        {"name": "Dark Horizon",  "trainer": "C. Appleby",  "jockey": "W. Buick",
         "form": "3-2-1-4-5", "age": 6, "weight_lbs": 124, "draw": 3,
         "days_since_last_run": 42, "jockey_win_rate": 0.19, "trainer_win_rate": 0.17,
         "speed_ratings": [114, 110, 116, 104, 102], "career_wins": 4, "career_runs": 14,
         "going_history": [["Soft", 1], ["Good", 4]], "distance_history": [[14, 1], [16, 2]],
         "track_wins": 1, "track_runs": 6, "xgb_win_prob": 0.18,
         "injury_risk": 0.62, "travel_risk": 0.05, "fatigue_risk": 0.25,
         "news_text": "Tendon inflammation reported after last workout."},
        {"name": "Morning Glory", "trainer": "M. Johnston", "jockey": "H. Doyle",
         "form": "4-5-3-3-4", "age": 5, "weight_lbs": 122, "draw": 4,
         "days_since_last_run": 10, "jockey_win_rate": 0.14, "trainer_win_rate": 0.13,
         "speed_ratings": [98, 100, 102, 99, 101], "career_wins": 3, "career_runs": 18,
         "going_history": [["Good", 3], ["Good", 4]], "distance_history": [[14, 3], [12, 4]],
         "track_wins": 0, "track_runs": 3, "xgb_win_prob": 0.11,
         "injury_risk": 0.04, "travel_risk": 0.03, "fatigue_risk": 0.10,
         "news_text": "Dropped in class. Trainer expecting improvement."},
    ]
    demo_race = {
        "race_id": "demo_ascot_r3", "race_name": "Royal Ascot — Race 3 (1m4f)",
        "track": "Royal Ascot", "going": "Good",
        "surface": "Turf", "distance_furlongs": 14.0,
    }

    try:
        result = predictor.predict(demo_horses, demo_race, use_cache=True)
    except Exception as e:
        logger.error(f"Demo prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return _format_response(result)


# ── /news ─────────────────────────────────────────────────────────────────────

@router.post("/news", response_model=NewsIntelResponse, tags=["news"])
def get_news_intel(req: NewsRequest):
    """
    Fetch and analyse news for a single horse.
    Returns structured risk scores + LLM summary.
    """
    try:
        articles      = fetch_horse_news(req.horse_name, req.trainer_name,
                                         use_cache=req.use_cache)
        articles_text = format_articles_for_llm(articles)
        intel         = summarise_horse_news(req.horse_name, articles_text,
                                             req.race_info, use_cache=req.use_cache)
        display       = news_intel_to_dashboard(intel)

        return NewsIntelResponse(
            horse_name       = req.horse_name,
            summary          = intel.get("summary", ""),
            injury_risk      = intel.get("injury_risk", 0.0),
            travel_risk      = intel.get("travel_risk", 0.0),
            fatigue_risk     = intel.get("fatigue_risk", 0.0),
            sentiment        = intel.get("sentiment", 0.0),
            confidence       = intel.get("confidence", 0.1),
            flags            = intel.get("flags", []),
            positive_signals = intel.get("positive_signals", []),
            overall_risk     = display["overall_risk"],
            sentiment_label  = display["sentiment_label"],
            articles         = [
                ArticleItem(
                    title     = a.get("title", ""),
                    source    = a.get("source", ""),
                    published = a.get("published", ""),
                    summary   = a.get("summary", ""),
                    url       = a.get("url", ""),
                ) for a in articles
            ],
        )
    except Exception as e:
        logger.error(f"News intelligence error for {req.horse_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_response(result: dict) -> PredictResponse:
    """Convert raw predictor output dict to Pydantic response model."""
    horses_out = []
    for h in result["horses"]:
        shap      = h.get("shap", {})
        mw        = h.get("modal_weights", {"structured": 33, "odds": 33, "news": 33})
        risk_flags= [RiskFlag(**f) for f in h.get("risk_flags", [])
                     if all(k in f for k in ["type", "severity", "label", "score"])]

        horses_out.append(HorsePrediction(
            rank          = h["rank"],
            name          = h["name"],
            form          = h.get("form", "-"),
            jockey        = h.get("jockey", ""),
            trainer       = h.get("trainer", ""),
            age           = h.get("age", 4),
            draw          = h.get("draw", 0),
            win_prob      = h["win_prob"],
            top2_prob     = h["top2_prob"],
            top3_prob     = h["top3_prob"],
            confidence    = h["confidence"],
            modal_weights = ModalWeights(**mw),
            risk_flags    = risk_flags,
            top_drivers   = [ShapEntry(**d) for d in shap.get("top_drivers", [])],
            top_risks     = [ShapEntry(**r) for r in shap.get("top_risks", [])],
            news_summary  = h.get("news_text", ""),
        ))

    return PredictResponse(
        race_id          = result["race_id"],
        race_name        = result["race_name"],
        track            = result["track"],
        going            = result["going"],
        surface          = result["surface"],
        distance         = result["distance"],
        field_size       = result["field_size"],
        model_version    = result["model_version"],
        inference_ms     = result["inference_ms"],
        top_pick         = result["top_pick"],
        model_confidence = result["model_confidence"],
        horses           = horses_out,
    )