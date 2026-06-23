"""
Training script for RaceOracle — uses REAL ingested data.

Run in order:
    1. python scripts/ingest_data.py data/raw/horse_races.csv
    2. python scripts/train_model.py
"""

import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import joblib

from backend.model.training.tab_transformer import SurgicalTabTransformer
from backend.model.training.model_config import (
    CAT_FEATURES, CONT_FEATURES,
    TAB_D_MODEL, TAB_N_LAYERS, TAB_OUT_DIM
)
from backend.utils.logger import logger

META_PATH = "./data/processed/dataset_meta.json"
DATA_PATH = "./data/processed/horse_races.csv"


def load_cardinalities() -> list:
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(
            "dataset_meta.json not found. "
            "Run: python scripts/ingest_data.py <your_csv> first."
        )
    with open(META_PATH) as f:
        meta = json.load(f)
    logger.info(f"Dataset: {meta['n_rows']:,} rows | {meta['n_races']:,} races | "
                f"{meta['n_courses']} courses | {meta['date_min']} → {meta['date_max']}")
    return [meta["n_courses"], 5, 3, 1, 5, 8]


def clamp_cat(df: pd.DataFrame, n_courses: int) -> pd.DataFrame:
    df = df.copy()
    df["track_idx"]       = df["track_idx"].clip(0, n_courses - 1)
    df["going_idx"]       = df["going_idx"].clip(0, 4)
    df["surface_idx"]     = df["surface_idx"].clip(0, 2)
    df["draw_position"]   = df["draw_position"].clip(0, 0)
    df["class_drop_rise"] = df["class_drop_rise"].clip(0, 4)
    df["age_years"]       = df["age_years"].clip(0, 7)
    return df


def top_pick_accuracy(model, head, df_val: pd.DataFrame,
                       X_cat_val, X_cont_val, rw_val) -> float:
    """
    The metric that matters for casino clients:
    In what % of races does the model's top-ranked horse actually win?
    (Here 'win' = lowest decimalPrice = our proxy label.)
    """
    model.eval(); head.eval()
    with torch.no_grad():
        logits = head(model(X_cat_val, X_cont_val, rw_val)).squeeze(-1)
        probs  = torch.sigmoid(logits).numpy()

    df_val = df_val.copy()
    df_val["pred_prob"] = probs
    df_val["won"]       = df_val["won"].astype(int)

    correct = 0
    total   = 0
    for _, race in df_val.groupby("race_id"):
        if len(race) < 2:
            continue
        top_pick_idx = race["pred_prob"].idxmax()
        if race.loc[top_pick_idx, "won"] == 1:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def load_and_preprocess():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. "
            "Run: python scripts/ingest_data.py <your_csv> first."
        )

    cat_cardinalities = load_cardinalities()
    n_courses = cat_cardinalities[0]

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df):,} rows for training")

    df = clamp_cat(df, n_courses)

    for col, card in zip(CAT_FEATURES, cat_cardinalities):
        bad = (df[col] < 0) | (df[col] >= card)
        if bad.any():
            logger.warning(f"Clamping {bad.sum()} out-of-range in {col}")
            df[col] = df[col].clip(0, card - 1)

    # ── Split by RACE not by row — prevents data leakage ──────────────────
    # Horses in the same race must all be in the same split
    race_ids     = df["race_id"].unique()
    n_val        = max(1, int(len(race_ids) * 0.15))
    val_races    = set(race_ids[-n_val:])   # use last 15% of races (chronological)
    train_mask   = ~df["race_id"].isin(val_races)
    val_mask     = df["race_id"].isin(val_races)

    df_train = df[train_mask].reset_index(drop=True)
    df_val   = df[val_mask].reset_index(drop=True)
    logger.info(f"Train: {len(df_train):,} rows ({df_train['race_id'].nunique():,} races) | "
                f"Val: {len(df_val):,} rows ({df_val['race_id'].nunique():,} races)")

    # ── Scale continuous features ──────────────────────────────────────────
    scaler = StandardScaler()
    df_train[CONT_FEATURES] = scaler.fit_transform(df_train[CONT_FEATURES].fillna(0))
    df_val[CONT_FEATURES]   = scaler.transform(df_val[CONT_FEATURES].fillna(0))
    os.makedirs("./data/models", exist_ok=True)
    joblib.dump(scaler, "./data/models/scaler.pkl")

    # Save cardinalities
    with open("./data/models/cat_cardinalities.json", "w") as f:
        json.dump(cat_cardinalities, f)

    def to_tensors(d):
        days = d["days_since_last_run"].clip(0, 365).values
        rw   = np.outer(np.exp(-days / 30.0), np.exp(-0.1 * (6 - np.arange(7))))
        return (
            torch.tensor(d[CAT_FEATURES].values.astype(int),    dtype=torch.long),
            torch.tensor(d[CONT_FEATURES].values.astype(float),  dtype=torch.float32),
            torch.tensor(d["xgb_win_prob"].values,               dtype=torch.float32),
            torch.tensor(rw, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(d["won"].values,                         dtype=torch.float32),
        )

    tr = to_tensors(df_train)
    va = to_tensors(df_val)

    # Compute pos_weight from actual positive rate
    pos_rate   = df_train["won"].mean()
    pos_weight = torch.tensor([(1 - pos_rate) / pos_rate])
    logger.info(f"Positive rate: {pos_rate:.3f} | pos_weight: {pos_weight.item():.1f}")

    def make_loader(tensors, shuffle=True):
        ds = TensorDataset(*tensors)
        return DataLoader(ds, batch_size=256, shuffle=shuffle, drop_last=False)

    return (make_loader(tr), make_loader(va, shuffle=False),
            tr, va, df_val, cat_cardinalities, pos_weight)


def evaluate(model, head, loader, loss_fn):
    model.eval(); head.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x_cat, x_cont, _, rw, y in loader:
            logit = head(model(x_cat, x_cont, rw)).squeeze(-1)
            total_loss += loss_fn(logit, y).item()
            correct    += ((torch.sigmoid(logit) > 0.5) == y.bool()).sum().item()
            n          += len(y)
    return total_loss, correct / n


def train_phase1(train_loader, val_loader, val_tensors, df_val,
                  cat_cardinalities, pos_weight, epochs=30):
    logger.info("=== Phase 1: Pretraining TabTransformer on real data ===")

    model  = SurgicalTabTransformer(cat_cardinalities, len(CONT_FEATURES),
                                    d_model=TAB_D_MODEL, n_layers=TAB_N_LAYERS,
                                    out_dim=TAB_OUT_DIM, dropout=0.2)
    head   = nn.Linear(TAB_OUT_DIM, 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim_  = optim.AdamW(list(model.parameters()) + list(head.parameters()),
                          lr=5e-4, weight_decay=1e-3)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(optim_, patience=3, factor=0.5)

    X_cat_v, X_cont_v, _, rw_v, _ = val_tensors
    best_val, best_tpa, patience_count = float("inf"), 0.0, 0
    PATIENCE = 6  # early stopping

    for epoch in range(epochs):
        model.train(); head.train()
        train_loss = 0.0
        for x_cat, x_cont, _, rw, y in train_loader:
            optim_.zero_grad()
            loss = loss_fn(head(model(x_cat, x_cont, rw)).squeeze(-1), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_.step()
            train_loss += loss.item()

        val_loss, val_acc = evaluate(model, head, val_loader, loss_fn)
        tpa = top_pick_accuracy(model, head, df_val, X_cat_v, X_cont_v, rw_v)
        sched.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_tpa = tpa
            patience_count = 0
            torch.save(model.state_dict(), "./data/models/tab_transformer_pretrained.pt")
        else:
            patience_count += 1

        if (epoch + 1) % 5 == 0 or patience_count == PATIENCE:
            logger.info(f"  Epoch {epoch+1:>3}/{epochs} | "
                        f"train={train_loss:.2f} | val={val_loss:.2f} | "
                        f"acc={val_acc:.3f} | top-pick-acc={tpa:.3f}")

        if patience_count >= PATIENCE:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    logger.info(f"Phase 1 done. Best val={best_val:.4f} | top-pick-acc={best_tpa:.3f}")
    return model


def train_phase2(model, train_loader, val_loader, val_tensors, df_val,
                  pos_weight, epochs=15):
    logger.info("=== Phase 2: Surgical heads only (backbone frozen) ===")
    model.load_state_dict(torch.load("./data/models/tab_transformer_pretrained.pt",
                                     map_location="cpu", weights_only=True))
    model.freeze_pretrained_layers()
    trainable = model.count_trainable_params()
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    head    = nn.Linear(TAB_OUT_DIM, 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    params  = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optim_  = optim.AdamW(params, lr=2e-4, weight_decay=1e-3)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(optim_, patience=3, factor=0.5, verbose=False)

    X_cat_v, X_cont_v, _, rw_v, _ = val_tensors
    best_val, best_tpa, patience_count = float("inf"), 0.0, 0
    PATIENCE = 5

    for epoch in range(epochs):
        model.train(); head.train()
        train_loss = 0.0
        for x_cat, x_cont, _, rw, y in train_loader:
            optim_.zero_grad()
            loss = loss_fn(head(model(x_cat, x_cont, rw)).squeeze(-1), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_.step()
            train_loss += loss.item()

        val_loss, val_acc = evaluate(model, head, val_loader, loss_fn)
        tpa = top_pick_accuracy(model, head, df_val, X_cat_v, X_cont_v, rw_v)
        sched.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_tpa = tpa
            patience_count = 0
            torch.save(model.state_dict(), "./data/models/tab_transformer_surgical.pt")
        else:
            patience_count += 1

        if (epoch + 1) % 5 == 0 or patience_count == PATIENCE:
            logger.info(f"  Epoch {epoch+1:>3}/{epochs} | "
                        f"train={train_loss:.2f} | val={val_loss:.2f} | "
                        f"acc={val_acc:.3f} | top-pick-acc={tpa:.3f}")

        if patience_count >= PATIENCE:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    logger.info(f"Phase 2 done. Best val={best_val:.4f} | top-pick-acc={best_tpa:.3f}")
    logger.info("Saved → ./data/models/tab_transformer_surgical.pt")
    return model


if __name__ == "__main__":
    (train_loader, val_loader,
     tr_tensors, va_tensors,
     df_val, cat_cardinalities, pos_weight) = load_and_preprocess()

    model = train_phase1(train_loader, val_loader, va_tensors, df_val,
                          cat_cardinalities, pos_weight, epochs=30)
    model = train_phase2(model, train_loader, val_loader, va_tensors, df_val,
                          pos_weight, epochs=15)

    logger.info("=" * 50)
    logger.info("Training complete. Models saved to ./data/models/")
    logger.info("Next step: python scripts/backtest.py")