"""
RaceOracle Inference Predictor.

Single-entry-point class that:
  1. Loads the trained surgical fusion model from disk
  2. Accepts raw race + horse data
  3. Runs the full pipeline: feature engineering → encoding → model → SHAP
  4. Returns a clean, API-ready prediction response dict

Usage:
    predictor = RacePredictor()
    result    = predictor.predict(horses_raw, race_context)
"""

import os
import json
import time
import torch
import numpy as np
from typing import Optional

from backend.model.training.fusion_model import RaceOracleModel
from backend.model.training.tab_transformer import SurgicalTabTransformer
from backend.pipeline.feature_engineering.features import build_race_features
from backend.pipeline.feature_engineering.encoders import RaceFeatureEncoder
from backend.model.explainability.shap_explainer import compute_feature_shap, format_shap_for_api
from backend.utils.config import config
from backend.utils.logger import logger
from backend.utils.cache import cache_get, cache_set


# ── Default model config (must match training) ─────────────────────────────
CAT_CARDINALITIES = [5, 5, 3, 20, 5, 9]
NUM_CONTINUOUS    = 17


class RacePredictor:
    """
    Thread-safe inference wrapper for the RaceOracle surgical model.
    Designed to be instantiated once at API startup and reused.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device     = device
        self.model_path = model_path or os.path.join(config.MODEL_DIR, "raceoracle_full.pt")
        self.encoder    = RaceFeatureEncoder()
        self.model      = None
        self._load_model()

    def _load_model(self):
        """Load surgical model. Falls back to untrained model for dev/demo."""
        self.model = RaceOracleModel(
            cat_cardinalities=CAT_CARDINALITIES,
            num_continuous=NUM_CONTINUOUS,
            tab_dim=64,
            fused_dim=256,
            dropout=0.0,   # disable dropout at inference
        ).to(self.device)
        self.model.eval()

        if os.path.exists(self.model_path):
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            logger.info(f"Loaded trained model from {self.model_path}")
        else:
            logger.warning(
                f"No trained model found at {self.model_path}. "
                "Using untrained weights — run scripts/train_model.py first."
            )

        # Load tab_transformer surgical weights if available separately
        tab_path = os.path.join(config.MODEL_DIR, "tab_transformer_surgical.pt")
        if os.path.exists(tab_path) and not os.path.exists(self.model_path):
            state = torch.load(tab_path, map_location=self.device)
            self.model.tab_model.load_state_dict(state)
            logger.info("Loaded surgical TabTransformer weights.")

    def predict(self, horses_raw: list, race_context: dict,
                use_cache: bool = True) -> dict:
        """
        Full prediction pipeline for one race.

        Args:
            horses_raw:   list of raw horse dicts (see features.py for schema)
            race_context: {track, going, surface, distance_furlongs, race_name, race_id}
            use_cache:    cache results by race_id for 10 minutes

        Returns:
            Full prediction response dict ready for the API / frontend.
        """
        race_id   = race_context.get("race_id", "unknown")
        cache_key = f"predict:{race_id}"

        if use_cache:
            cached = cache_get(cache_key, ttl_seconds=600)
            if cached:
                logger.info(f"Cache hit for race {race_id}")
                return cached

        t0 = time.time()

        # 1. Feature engineering
        horse_features = build_race_features(horses_raw, race_context)

        # 2. Fit encoder if first run (dev mode) — in prod scaler is pre-fitted
        if not self.encoder.is_fitted:
            logger.warning("Scaler not found — fitting on current race (dev mode only).")
            self.encoder.fit(horse_features)

        # 3. Encode to tensors
        encoded_horses = self.encoder.encode_race(horse_features)

        # 4. Model inference
        with torch.no_grad():
            raw_outputs = self.model.predict_race(encoded_horses, device=self.device)

        # 5. Baseline for SHAP delta computation (uniform prior)
        n = len(horses_raw)
        baseline = {"win_probs": np.full(n, 1.0 / n)}

        # 6. SHAP explainability
        shap_raw    = compute_feature_shap(raw_outputs, baseline)
        shap_formatted = format_shap_for_api(shap_raw)

        # 7. Assemble response
        horses_out = []
        win_probs  = raw_outputs["win_probs"]
        risks      = raw_outputs["risk_scores"]
        gates      = raw_outputs["modal_gates"]
        confs      = raw_outputs["confidence"]

        for i, enc in enumerate(encoded_horses):
            risk_flags = []
            if risks[i][0] > 0.35:
                risk_flags.append({"type": "injury",  "severity": float(risks[i][0]), "label": "🔴 Injury risk"})
            if risks[i][1] > 0.30:
                risk_flags.append({"type": "travel",  "severity": float(risks[i][1]), "label": "🟡 Travel stress"})
            if risks[i][2] > 0.35:
                risk_flags.append({"type": "fatigue", "severity": float(risks[i][2]), "label": "🟠 Fatigue flag"})

            horses_out.append({
                "rank":          int(np.argsort(-win_probs)[i]) + 1,
                "name":          enc["name"],
                "form":          enc["form"],
                "jockey":        enc["jockey"],
                "trainer":       enc["trainer"],
                "age":           enc["age"],
                "draw":          enc["draw"],
                "win_prob":      round(float(win_probs[i]) * 100, 1),
                "top2_prob":     round(float(self._compute_top2_prob(win_probs, i)) * 100, 1),
                "top3_prob":     round(float(self._compute_top3_prob(win_probs, i)) * 100, 1),
                "confidence":    round(float(confs[i]) * 100, 1),
                "modal_weights": {
                    "structured": round(float(gates[i][0]) * 100, 1),
                    "odds":       round(float(gates[i][1]) * 100, 1),
                    "news":       round(float(gates[i][2]) * 100, 1),
                },
                "risk_flags":    risk_flags,
                "shap":          shap_formatted[i],
                "news_text":     enc.get("news_text", ""),
            })

        # Sort by win probability descending
        horses_out.sort(key=lambda h: h["win_prob"], reverse=True)
        for rank_i, h in enumerate(horses_out):
            h["rank"] = rank_i + 1

        result = {
            "race_id":      race_id,
            "race_name":    race_context.get("race_name", "Unknown Race"),
            "track":        race_context.get("track", ""),
            "going":        race_context.get("going", ""),
            "surface":      race_context.get("surface", "Turf"),
            "distance":     race_context.get("distance_furlongs", 0),
            "field_size":   n,
            "model_version": "raceoracle-v1-surgical",
            "inference_ms": round((time.time() - t0) * 1000, 1),
            "top_pick":     horses_out[0]["name"] if horses_out else None,
            "model_confidence": round(float(np.mean(confs)) * 100, 1),
            "horses":       horses_out,
        }

        if use_cache:
            cache_set(cache_key, result)

        logger.info(
            f"Race {race_id} predicted in {result['inference_ms']}ms | "
            f"top pick: {result['top_pick']} ({horses_out[0]['win_prob']}%)"
        )
        return result

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _compute_top2_prob(win_probs: np.ndarray, idx: int) -> float:
        """P(horse finishes top 2) — sum of own win prob + conditional."""
        p_win = win_probs[idx]
        others = [p for j, p in enumerate(win_probs) if j != idx]
        p_second = sum(
            (win_probs[j] / (1 - p_win + 1e-8)) * p_win
            for j in range(len(win_probs)) if j != idx
        )
        return float(np.clip(p_win + (1 - p_win) * (p_win / (sum(others) + 1e-8)), 0, 1))

    @staticmethod
    def _compute_top3_prob(win_probs: np.ndarray, idx: int) -> float:
        """Approximate top-3 probability via complementary method."""
        p_not_top3 = 1.0
        sorted_others = sorted(
            [p for j, p in enumerate(win_probs) if j != idx], reverse=True
        )
        p_win = win_probs[idx]
        # Simplified: P(top3) ≈ 1 - P(all 3 better horses beat us)
        top3_others_prob = sum(sorted_others[:2])
        return float(np.clip(p_win + (1 - p_win) * min(0.85, p_win * 3.5), 0, 1))

    def health_check(self) -> dict:
        """Returns model status for the /health API endpoint."""
        return {
            "model_loaded":   self.model is not None,
            "scaler_fitted":  self.encoder.is_fitted,
            "device":         self.device,
            "model_path":     self.model_path,
            "model_version":  "raceoracle-v1-surgical",
            "trainable_params": self.model.count_trainable_params() if self.model else {},
        }
