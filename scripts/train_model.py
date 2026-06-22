"""
Training script for RaceOracle.

Trains the full surgical fusion model end-to-end on CPU.
Run: python scripts/train_model.py

Phase 1: Pretrain TabTransformer on structured data only
Phase 2: Freeze layers 0-5, train surgical heads
Phase 3: Train full fusion model (all unfrozen components jointly)
"""

import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from backend.model.training.tab_transformer import SurgicalTabTransformer
from backend.model.training.fusion_model import RaceOracleModel, XGBoostEmbedder
from backend.utils.logger import logger


CAT_FEATURES  = ["track_idx", "going_idx", "surface_idx", "draw_position", "class_drop_rise", "age_years"]
CONT_FEATURES = [
    "win_rate_last5", "win_rate_career", "place_rate_last5",
    "days_since_last_run", "going_preference_score", "distance_fit_score",
    "jockey_win_rate", "trainer_win_rate", "jockey_trainer_combo",
    "weight_carried_lbs", "speed_rating_last", "speed_rating_avg3",
    "track_win_rate", "field_size",
    "injury_risk", "travel_risk", "fatigue_risk",
]
CAT_CARDINALITIES = [len(["Royal Ascot","Cheltenham","Newmarket","Epsom","Goodwood"]),
                     5, 3, 20, 5, 10]


def load_and_preprocess(data_path: str):
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")

    scaler = StandardScaler()
    df[CONT_FEATURES] = scaler.fit_transform(df[CONT_FEATURES].fillna(0))
    joblib.dump(scaler, "./data/models/scaler.pkl")

    X_cat  = torch.tensor(df[CAT_FEATURES].fillna(0).values.astype(int), dtype=torch.long)
    X_cont = torch.tensor(df[CONT_FEATURES].values.astype(float), dtype=torch.float32)
    xgb_p  = torch.tensor(df["xgb_win_prob"].values, dtype=torch.float32)
    y      = torch.tensor(df["won"].values, dtype=torch.float32)

    # Simple recency weights (proxy: 1 / (1 + days_since_last_run / 30))
    days   = df["days_since_last_run"].values
    rw     = 1.0 / (1.0 + days / 30.0)
    rw_t   = torch.tensor(rw, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

    idx = np.arange(len(df))
    tr, va = train_test_split(idx, test_size=0.15, random_state=42)

    def make_loader(i, shuffle=True):
        ds = TensorDataset(X_cat[i], X_cont[i], xgb_p[i], rw_t[i].expand(-1, 10, 1), y[i])
        return DataLoader(ds, batch_size=256, shuffle=shuffle)

    return make_loader(tr), make_loader(va, shuffle=False)


def train_phase1_tab(train_loader, val_loader, epochs: int = 30):
    """Phase 1: pretrain TabTransformer on structured race data."""
    logger.info("=== Phase 1: Pretraining TabTransformer ===")
    model  = SurgicalTabTransformer(CAT_CARDINALITIES, len(CONT_FEATURES), d_model=64, out_dim=128)
    head   = nn.Linear(128, 1)
    optim_ = optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    for epoch in range(epochs):
        model.train(); head.train()
        train_loss = 0.0
        for x_cat, x_cont, _, rw, y in train_loader:
            optim_.zero_grad()
            emb  = model(x_cat, x_cont, rw)
            loss = loss_fn(head(emb).squeeze(-1), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_.step()
            train_loss += loss.item()
        sched.step()

        model.eval(); head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_cont, _, rw, y in val_loader:
                emb  = model(x_cat, x_cont, rw)
                val_loss += loss_fn(head(emb).squeeze(-1), y).item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "./data/models/tab_transformer_pretrained.pt")

        if (epoch + 1) % 5 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} | train: {train_loss:.4f} | val: {val_loss:.4f}")

    logger.info(f"Phase 1 complete. Best val loss: {best_val:.4f}")
    return model


def train_phase2_surgical(pretrained_tab, train_loader, val_loader, epochs: int = 20):
    """Phase 2: freeze pretrained layers, train only surgical heads."""
    logger.info("=== Phase 2: Training surgical heads (frozen backbone) ===")
    pretrained_tab.freeze_pretrained_layers()
    trainable = pretrained_tab.count_trainable_params()
    logger.info(f"Trainable params after freeze: {trainable:,}")

    head    = nn.Linear(128, 1)
    params  = [p for p in pretrained_tab.parameters() if p.requires_grad] + list(head.parameters())
    optim_  = optim.AdamW(params, lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        pretrained_tab.train(); head.train()
        for x_cat, x_cont, _, rw, y in train_loader:
            optim_.zero_grad()
            emb  = pretrained_tab(x_cat, x_cont, rw)
            loss = loss_fn(head(emb).squeeze(-1), y)
            loss.backward()
            optim_.step()

        if (epoch + 1) % 5 == 0:
            logger.info(f"  Surgical epoch {epoch+1}/{epochs}")

    torch.save(pretrained_tab.state_dict(), "./data/models/tab_transformer_surgical.pt")
    logger.info("Phase 2 complete.")
    return pretrained_tab


if __name__ == "__main__":
    os.makedirs("./data/models", exist_ok=True)
    data_path = "./data/processed/horse_races.csv"

    if not os.path.exists(data_path):
        logger.info("Mock data not found — generating...")
        os.system("python scripts/generate_mock_data.py")

    train_loader, val_loader = load_and_preprocess(data_path)

    tab_model  = train_phase1_tab(train_loader, val_loader, epochs=20)
    tab_model  = train_phase2_surgical(tab_model, train_loader, val_loader, epochs=15)

    logger.info("Training complete. Models saved to ./data/models/")
