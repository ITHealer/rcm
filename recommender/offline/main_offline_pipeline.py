

# scripts/offline/main_offline_pipeline.py
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine

from recommender.common.data_loading import DataLoader
from recommender.common.feature_engineer import build_training_matrices
from recommender.offline.training_state import TrainingStateManager
from recommender.offline.model_trainer import ModelTrainer
from recommender.offline.artifact_manager import save_artifacts


# ----------------------------- helpers ---------------------------------------
def load_yaml_if_exists(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def make_engine(cfg: Dict[str, Any]):
    url = (cfg.get("database", {}) or {}).get("url", "")
    if not url:
        return None
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=int(cfg.get("database", {}).get("pool_size", 5)),
        max_overflow=int(cfg.get("database", {}).get("max_overflow", 5)),
        future=True,
    )

def tcfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "window_days": int(cfg.get("training", {}).get("window_days", 14)),
        "half_life_days": int(cfg.get("training", {}).get("half_life_days", 7)),
        "pretrain_full_export": bool(cfg.get("training", {}).get("pretrain_full_export", True)),
        "overlap_days": int(cfg.get("training", {}).get("overlap_days", 2)),
        "chunk_size": int(cfg.get("training", {}).get("chunk_size", 200_000)),
        "test_days": int(cfg.get("training", {}).get("test_days", 3)),
        "val_days": int(cfg.get("training", {}).get("val_days", 3)),
        # optional override multipliers
        "action_multipliers": (cfg.get("training", {}).get("action_multipliers") or {
            "view": 0.5,
            "like": 1.0, "love": 1.3, "care": 1.25, "laugh": 1.2, "wow": 1.1, "sad": 0.9, "angry": 0.9,
            "comment": 1.5, "share": 2.0, "save": 1.2,
        }),
    }

def acfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "base_dir": cfg.get("artifacts", {}).get("base_dir", "models"),
        "state_file": cfg.get("artifacts", {}).get("state_file", "models/training_state.json"),
    }

def temporal_split(df: pd.DataFrame, test_days: int = 3, val_days: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "created_at" not in df.columns:
        raise ValueError("interactions must include 'created_at' for temporal split")
    d = df.copy()
    d["created_at"] = pd.to_datetime(d["created_at"], utc=True, errors="coerce")
    d = d.sort_values("created_at")
    maxd = d["created_at"].max()
    test_start = maxd - timedelta(days=test_days)
    val_start = test_start - timedelta(days=val_days)
    return d[d["created_at"] < val_start], d[(d["created_at"] >= val_start) & (d["created_at"] < test_start)], d[d["created_at"] >= test_start]

def compute_time_decay_weights(
    inter_df: pd.DataFrame,
    half_life_days: int,
    action_multipliers: Dict[str, float],
    now_ts: datetime,
) -> pd.Series:
    """
    weight = action_multiplier * 0.5 ** (age_days / half_life)
    """
    ts = pd.to_datetime(inter_df["created_at"], utc=True, errors="coerce")
    age_days = (now_ts - ts).dt.total_seconds() / 86400.0
    decay = np.power(0.5, np.clip(age_days, 0, None) / max(half_life_days, 1))
    mult = inter_df["action"].map(action_multipliers).fillna(1.0).astype(float)
    w = np.maximum(decay * mult.values, 1e-3)
    return pd.Series(w, index=inter_df.index)


# ------------------------------ main -----------------------------------------
def main():
    # 1) Load config
    cfg = load_yaml_if_exists("configs/config_offline.yaml")
    cfg2 = load_yaml_if_exists("configs/config.yaml")
    for k, v in cfg2.items():
        cfg.setdefault(k, v)
    train_cfg = tcfg(cfg)
    art_cfg = acfg(cfg)

    # 2) Engine
    engine = make_engine(cfg)

    # 3) State
    state_mgr = TrainingStateManager(art_cfg["state_file"])
    state = state_mgr.load()

    # 4) DataLoader
    dl = DataLoader(
        db_connection=engine,
        config={
            "lookback_days": train_cfg["window_days"],
            "chunk_size": train_cfg["chunk_size"],
            "tables": (cfg.get("database", {}).get("tables", {}) or {}),
            "reaction_code_map": (cfg.get("database", {}).get("reaction_code_map", {}) or {
                "like": "like", "love": "love", "laugh": "laugh", "wow": "wow", "sad": "sad", "angry": "angry", "care": "care"
            }),
            "reaction_name_map": (cfg.get("database", {}).get("reaction_name_map", {}) or {
                "Like": "like", "Love": "love", "Laugh": "laugh", "Wow": "wow", "Sad": "sad", "Angry": "angry", "Care": "care"
            }),
            "csv_dir": (cfg.get("dataset", {}) or {}).get("dir", "dataset"),
            "csv_files": (cfg.get("dataset", {}) or {}),
        },
    )

    # 5) Load bundle (pretrain vs incremental)
    use_csv = engine is None
    now = now_utc()
    if state.last_train_end is None:
        # PATCH #1: lần đầu -> lấy toàn bộ lịch sử (data bạn mới ~2 tháng)
        if train_cfg.get("pretrain_full_export", True):
            since = datetime(2024, 1, 1, tzinfo=timezone.utc)  # lấy hết
        else:
            since = now - timedelta(days=train_cfg["window_days"])
        bundle = dl.load_training_bundle(since=since, until=now, use_csv=use_csv)
    else:
        last = pd.to_datetime(state.last_train_end, utc=True)
        since = last - timedelta(days=train_cfg["overlap_days"])  # overlap để an toàn
        bundle = dl.load_training_bundle(since=since, until=now, use_csv=use_csv)

    interactions = bundle["interactions"]
    users = bundle["users"]
    posts = bundle["posts"]
    friendships = bundle["friendships"]
    post_hashtags = bundle["post_hashtags"]

    if interactions.empty:
        raise RuntimeError("No interactions loaded for training.")

    # 6) Vector FE (đầy đủ features)
    X, y, interactions_df, meta = build_training_matrices(
        interactions=interactions,
        users_df=users,
        posts_df=posts,
        friendships_df=friendships,
        post_hashtags_df=post_hashtags,
        embeddings=None,
        now_ts=now,
    )

    # 7) Temporal split trên interactions_df (đã bao gồm 'action' + 'created_at')
    tmp = interactions_df.copy()
    tmp["label"] = y.values
    tmp["__row_id__"] = np.arange(len(tmp))
    tr_i, va_i, te_i = temporal_split(tmp, test_days=train_cfg["test_days"], val_days=train_cfg["val_days"])

    tr_rows = tr_i["__row_id__"].to_numpy()
    va_rows = va_i["__row_id__"].to_numpy()
    te_rows = te_i["__row_id__"].to_numpy()

    X_train, y_train, inter_train = X.iloc[tr_rows], y.iloc[tr_rows], interactions_df.iloc[tr_rows]
    X_val, y_val, inter_val = X.iloc[va_rows], y.iloc[va_rows], interactions_df.iloc[va_rows]
    X_test, y_test, inter_test = X.iloc[te_rows], y.iloc[te_rows], interactions_df.iloc[te_rows]

    # 8) Time-decay weights + multipliers (PATCH #2)
    w_train = compute_time_decay_weights(inter_train, train_cfg["half_life_days"], train_cfg["action_multipliers"], now)
    w_val   = compute_time_decay_weights(inter_val,   train_cfg["half_life_days"], train_cfg["action_multipliers"], now)

    train_df = X_train.copy(); train_df["label"] = y_train.values; train_df["weight"] = w_train.values
    val_df   = X_val.copy();   val_df["label"] = y_val.values;     val_df["weight"]   = w_val.values
    test_df  = X_test.copy();  test_df["label"] = y_test.values  # test không cần weight

    # 9) Train + Evaluate
    trainer = ModelTrainer(
        config={
            "model": {
                "params": {
                    "objective": "binary",
                    "metric": "auc",
                    "learning_rate": 0.05,
                    "num_leaves": 64,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "verbose": -1,
                },
                "num_boost_round": 300,
                "early_stopping_rounds": 50,
                "log_every_n": 50,
            },
            "training": {
                "half_life_days": train_cfg["half_life_days"],
                "min_weight": 1e-3,
                "use_action_multiplier": True,
                "action_multipliers": train_cfg["action_multipliers"],
            },
        }
    )
    model, scaler, feature_cols = trainer.train(train_df, val_df)
    metrics = trainer.evaluate(test_df)

    # 10) Save artifacts
    version = now.strftime("v%Y%m%d_%H%M%S")
    meta_out = {
        "version": version,
        "trained_at": now.isoformat(),
        "rows": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "features": feature_cols,
        "metrics": metrics,
        "window_days": train_cfg["window_days"],
        "half_life_days": train_cfg["half_life_days"],
    }
    save_artifacts(version_name=version, model=model, meta=meta_out, artifacts_base_dir=art_cfg["base_dir"])

    # Lưu scaler/feature_cols theo API cũ
    out_base = str(Path(art_cfg["base_dir"]) / version / "ranker")
    trainer.save_model(out_base)

    # 11) Update state
    state.last_train_end = TrainingStateManager.now_iso()
    state_mgr.save(state)

    print(f"[OK] Offline pipeline finished. Version: {version}")
    print(f"[OK] Metrics: {json.dumps(metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
