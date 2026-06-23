"""
RaceOracle Inference Predictor.
"""

import os
import time
import torch
import numpy as np
from typing import Optional

from backend.model.training.fusion_model import RaceOracleModel
import json
from backend.model.training.model_config import NUM_CONTINUOUS, TAB_D_MODEL, FUSED_DIM
from backend.pipeline.feature_engineering.features import build_race_features
from backend.pipeline.feature_engineering.encoders import RaceFeatureEncoder
from backend.model.explainability.shap_explainer import compute_feature_shap, format_shap_for_api
from backend.utils.config import config
from backend.utils.logger import logger
from backend.utils.cache import cache_get, cache_set


class RacePredictor:
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device     = device
        self.model_path = model_path or os.path.join(config.MODEL_DIR, "raceoracle_full.pt")
        self.encoder    = RaceFeatureEncoder()
        self.model      = None
        self._load_model()

    def _load_cardinalities(self) -> list:
        """Load from saved JSON — set dynamically by ingest_data.py."""
        card_path = os.path.join(config.MODEL_DIR, "cat_cardinalities.json")
        if os.path.exists(card_path):
            with open(card_path) as f:
                return json.load(f)
        # Fallback default (5 courses) — only used before first real ingestion
        from backend.model.training.model_config import CAT_CARDINALITIES
        return CAT_CARDINALITIES

    def _load_model(self):
        cat_cardinalities = self._load_cardinalities()
        self.model = RaceOracleModel(
            cat_cardinalities=cat_cardinalities,
            num_continuous=NUM_CONTINUOUS,
            tab_dim=TAB_D_MODEL,
            fused_dim=FUSED_DIM,
            dropout=0.0,
        ).to(self.device)
        self.model.eval()

        # Try full fused model first
        if os.path.exists(self.model_path):
            state = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            logger.info(f"Loaded full model from {self.model_path}")
            return

        # Fall back to surgical tab weights
        tab_path = os.path.join(config.MODEL_DIR, "tab_transformer_surgical.pt")
        if os.path.exists(tab_path):
            state = torch.load(tab_path, map_location=self.device, weights_only=True)
            result = self.model.tab_model.load_state_dict(state, strict=True)
            logger.info(f"Loaded surgical TabTransformer weights from {tab_path}")
            if result.missing_keys:
                logger.warning(f"Missing keys: {result.missing_keys}")
            return

        logger.warning("No trained model found. Using untrained weights — run scripts/train_model.py first.")

    def predict(self, horses_raw: list, race_context: dict, use_cache: bool = True) -> dict:
        race_id   = race_context.get("race_id", "unknown")
        cache_key = f"predict:{race_id}"

        if use_cache:
            cached = cache_get(cache_key, ttl_seconds=600)
            if cached:
                logger.info(f"Cache hit for race {race_id}")
                return cached

        t0 = time.time()

        horse_features = build_race_features(horses_raw, race_context)

        if not self.encoder.is_fitted:
            logger.warning("Scaler not found — fitting on current race (dev mode).")
            self.encoder.fit(horse_features)

        encoded_horses = self.encoder.encode_race(horse_features)

        with torch.no_grad():
            raw_outputs = self.model.predict_race(encoded_horses, device=self.device)

        n        = len(horses_raw)
        baseline = {"win_probs": np.full(n, 1.0 / n)}
        shap_raw = compute_feature_shap(raw_outputs, baseline)
        shap_fmt = format_shap_for_api(shap_raw)

        win_probs = raw_outputs["win_probs"]
        risks     = raw_outputs["risk_scores"]
        gates     = raw_outputs["modal_gates"]
        confs     = raw_outputs["confidence"]

        horses_out = []
        for i, enc in enumerate(encoded_horses):
            risk_flags = []
            if risks[i][0] > 0.35:
                risk_flags.append({"type": "injury",  "severity": float(risks[i][0]), "label": "🔴 Injury risk"})
            if risks[i][1] > 0.30:
                risk_flags.append({"type": "travel",  "severity": float(risks[i][1]), "label": "🟡 Travel stress"})
            if risks[i][2] > 0.35:
                risk_flags.append({"type": "fatigue", "severity": float(risks[i][2]), "label": "🟠 Fatigue flag"})

            horses_out.append({
                "rank":          0,
                "name":          enc["name"],
                "form":          enc["form"],
                "jockey":        enc["jockey"],
                "trainer":       enc["trainer"],
                "age":           enc["age"],
                "draw":          enc["draw"],
                "win_prob":      round(float(win_probs[i]) * 100, 1),
                "top2_prob":     round(min(99.0, float(win_probs[i]) * 100 * 1.9), 1),
                "top3_prob":     round(min(99.0, float(win_probs[i]) * 100 * 2.6), 1),
                "confidence":    round(float(confs[i]) * 100, 1),
                "modal_weights": {
                    "structured": round(float(gates[i][0]) * 100, 1),
                    "odds":       round(float(gates[i][1]) * 100, 1),
                    "news":       round(float(gates[i][2]) * 100, 1),
                },
                "risk_flags":    risk_flags,
                "shap":          shap_fmt[i],
                "news_text":     enc.get("news_text", ""),
            })

        horses_out.sort(key=lambda h: h["win_prob"], reverse=True)
        for rank_i, h in enumerate(horses_out):
            h["rank"] = rank_i + 1

        result = {
            "race_id":          race_id,
            "race_name":        race_context.get("race_name", "Unknown Race"),
            "track":            race_context.get("track", ""),
            "going":            race_context.get("going", ""),
            "surface":          race_context.get("surface", "Turf"),
            "distance":         race_context.get("distance_furlongs", 0),
            "field_size":       n,
            "model_version":    "raceoracle-v1-surgical",
            "inference_ms":     round((time.time() - t0) * 1000, 1),
            "top_pick":         horses_out[0]["name"] if horses_out else None,
            "model_confidence": round(float(np.mean(confs)) * 100, 1),
            "horses":           horses_out,
        }

        if use_cache:
            cache_set(cache_key, result)

        logger.info(f"Race {race_id} predicted in {result['inference_ms']}ms | "
                    f"top pick: {result['top_pick']} ({horses_out[0]['win_prob']}%)")
        return result

    def health_check(self) -> dict:
        return {
            "model_loaded":     self.model is not None,
            "scaler_fitted":    self.encoder.is_fitted,
            "device":           self.device,
            "model_version":    "raceoracle-v1-surgical",
            "trainable_params": self.model.count_trainable_params() if self.model else {},
        }