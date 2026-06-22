# RaceOracle — AI Horse Racing Prediction Platform

> B2B casino intelligence platform using surgical deep learning for race outcome prediction.

## What makes it unique

Unlike competitors that treat models as black boxes, RaceOracle performs **model surgery**:

| Model | Surgery applied |
|---|---|
| TabTransformer | Freeze layers 1-6 · Replace 8 generic heads with 4 race-context heads · Add FormDecayFFN |
| Sentence-BERT | Freeze layers 1-8 · Remove CLS pooling · Replace with 3 risk-detection heads (injury/travel/fatigue) |
| XGBoost | Freeze tree structure · Unfreeze + recalibrate leaf weights · Add Platt scaling calibration layer |

These are fused via a **cross-modal attention fusion layer** that learns which signal to trust per context.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic training data
python scripts/generate_mock_data.py

# 3. Train the model (CPU-friendly, ~20 min)
python scripts/train_model.py

# 4. Start API
uvicorn backend.api.main:app --reload --port 8000

# 5. Start frontend (in another terminal)
cd frontend && npm install && npm start
```

## Project structure

```
raceoracle/
├── backend/
│   ├── api/               FastAPI routes + schemas
│   ├── model/
│   │   ├── training/
│   │   │   ├── tab_transformer.py   Surgical TabTransformer
│   │   │   ├── news_encoder.py      Surgical Sentence-BERT
│   │   │   ├── fusion_model.py      Cross-modal fusion + meta-learner
│   │   │   └── dataset.py           Data loaders
│   │   ├── inference/     Predictor wrapper
│   │   └── explainability/ SHAP attribution
│   └── pipeline/
│       ├── data_ingestion/ CSV + scraper
│       ├── feature_engineering/ Feature transforms
│       └── news_intelligence/  News fetch + Claude summarizer
├── frontend/              React dashboard
├── scripts/               Train + backtest + generate data
└── data/                  Raw / processed / models / cache
```

## Environment variables

Copy `.env.example` to `.env` and fill in:
- `ANTHROPIC_API_KEY` — for news intelligence summarization
- `NEWS_API_KEY` — for article fetching (newsapi.org free tier)
