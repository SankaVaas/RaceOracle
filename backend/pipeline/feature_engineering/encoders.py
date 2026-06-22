"""
Encoders — convert feature dicts into PyTorch tensors.
Also handles loading/saving the fitted StandardScaler.
"""

import numpy as np
import torch
import joblib
import os
from typing import Optional
from sklearn.preprocessing import StandardScaler

from backend.pipeline.feature_engineering.features import CONT_FEATURES


class RaceFeatureEncoder:
    """
    Stateful encoder that:
      1. Fits a StandardScaler on training data (call fit())
      2. Transforms feature dicts → tensors at inference time (call encode())

    Always save/load the scaler so train/inference scaling is identical.
    """

    def __init__(self, scaler_path: str = "./data/models/scaler.pkl"):
        self.scaler_path = scaler_path
        self.scaler: Optional[StandardScaler] = None
        self._try_load_scaler()

    def _try_load_scaler(self):
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)

    def fit(self, horse_feature_dicts: list):
        """Fit scaler on a list of feature dicts (from training data)."""
        cont_matrix = np.array([h["cont_values"] for h in horse_feature_dicts])
        self.scaler = StandardScaler()
        self.scaler.fit(cont_matrix)
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)

    def encode_horse(self, feat: dict) -> dict:
        """
        Convert a single horse feature dict → tensors.
        Returns dict with x_cat, x_cont, xgb_prob, recency_weights tensors.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Run fit() or load a saved scaler first.")

        cont_raw = np.array(feat["cont_values"], dtype=np.float32).reshape(1, -1)
        cont_scaled = self.scaler.transform(cont_raw).squeeze(0)

        x_cat = torch.tensor(feat["cat_values"], dtype=torch.long).unsqueeze(0)   # (1, 6)
        x_cont = torch.tensor(cont_scaled, dtype=torch.float32).unsqueeze(0)      # (1, 17)
        xgb_prob = torch.tensor([feat["xgb_win_prob"]], dtype=torch.float32)      # (1,)
        recency_w = torch.tensor(
            feat["recency_weights"], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(-1)                                               # (1, 7, 1)

        return {
            "x_cat":            x_cat,
            "x_cont":           x_cont,
            "xgb_prob":         xgb_prob,
            "recency_weights":  recency_w,
            "news_text":        feat.get("news_text", ""),
            # Passthrough metadata
            "name":    feat["name"],
            "form":    feat["form"],
            "jockey":  feat["jockey"],
            "trainer": feat["trainer"],
            "age":     feat["age"],
            "draw":    feat["draw"],
        }

    def encode_race(self, horse_feature_dicts: list) -> list:
        """Encode all horses in a race. Returns list of tensor dicts."""
        return [self.encode_horse(h) for h in horse_feature_dicts]

    @property
    def is_fitted(self) -> bool:
        return self.scaler is not None
