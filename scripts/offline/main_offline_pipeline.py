# """
# MAIN OFFLINE TRAINING PIPELINE WITH TIME DECAY
# ===============================================
# Complete end-to-end offline training

# Architecture:
# - Raw interactions → Embeddings & CF Model (NO time decay)
# - Weighted interactions → Ranking Model Training (WITH time decay)

# Run: python scripts/offline/main_offline_pipeline.py
# """

# import os
# import sys
# import yaml
# from datetime import datetime
# from pathlib import Path

# # Add project root to path
# sys.path.append(str(Path(__file__).parent.parent.parent))

# from recommender.common.data_loading import DataLoader
# from recommender.common.data_loader import load_data, compute_statistics
# from recommender.offline.embedding_generator import EmbeddingGenerator
# from recommender.offline.cf_builder import CFBuilder
# from recommender.offline.model_trainer import ModelTrainer
# from recommender.offline.artifact_manager import ArtifactManager


# def load_config(config_path='scripts/offline/config_offline.yaml'):
#     """Load configuration"""
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)


# def main():
#     """
#     Complete offline training pipeline
#     """
#     print("=" * 80)
#     print("🚀 OFFLINE TRAINING PIPELINE (WITH TIME DECAY)")
#     print("=" * 80)
#     print(f"Started at: {datetime.now()}")
    
#     # ═════════════════════════════════════════════════════════════════════
#     # STEP 1: LOAD DATA - DUAL LOADING STRATEGY
#     # ═════════════════════════════════════════════════════════════════════
    
#     print("\n" + "="*80)
#     print("STEP 1: DATA LOADING (DUAL STRATEGY)")
#     print("="*80)
    
#     config = load_config()
    
#     # ─────────────────────────────────────────────────────────────────────
#     # 1A. Load RAW data (NO time decay) - For Embeddings & CF
#     # ─────────────────────────────────────────────────────────────────────
    
#     print("\n📦 Loading RAW data (for Embeddings & CF)...")
    
#     data = load_data(config['data']['dir'])
    
#     # Compute statistics
#     user_stats, author_stats, following_dict = compute_statistics(data)
    
#     print(f"✅ Raw data loaded:")
#     print(f"   Users: {len(data['user']):,}")
#     print(f"   Posts: {len(data['post']):,}")
#     print(f"   Interactions: {len(data['postreaction']):,}")
#     print(f"   Friendships: {len(data['friendship']):,}")
    
#     # ─────────────────────────────────────────────────────────────────────
#     # 1B. Load WEIGHTED data (WITH time decay) - For Ranking Model Training
#     # ─────────────────────────────────────────────────────────────────────
    
#     print("\n📦 Loading WEIGHTED data (for Ranking Model)...")
    
#     data_loader_config = {
#         'lookback_days': config['data']['lookback_days'],
#         'half_life_days': config.get('time_decay', {}).get('half_life_days', 7.0),
#         'min_weight': config.get('time_decay', {}).get('min_weight', 0.01),
#         'chunk_size': config.get('data', {}).get('chunk_size', 100000)
#     }
    
#     data_loader = DataLoader(
#         db_connection=None,  # Use CSV for now
#         config=data_loader_config,
#         data_dir=config['data']['dir']
#     )
    
#     # Load interactions with time decay applied
#     interactions_weighted = data_loader.load_and_prepare_training_data(use_csv=True)
    
#     print(f"✅ Weighted data loaded: {len(interactions_weighted):,} interactions")
#     print(f"   Weight range: [{interactions_weighted['weight'].min():.4f}, {interactions_weighted['weight'].max():.4f}]")
#     print(f"   Mean weight: {interactions_weighted['weight'].mean():.4f}")
    
#     # Create temporal splits for weighted data
#     train_interactions, val_interactions, test_interactions = data_loader.create_train_test_split(
#         interactions_weighted,
#         test_days=config['data']['train_test_split']['test_days'],
#         val_days=config['data']['train_test_split']['val_days']
#     )
    
#     print(f"\n📊 Temporal splits:")
#     print(f"   Train: {len(train_interactions):,} ({len(train_interactions)/len(interactions_weighted)*100:.1f}%)")
#     print(f"   Val: {len(val_interactions):,} ({len(val_interactions)/len(interactions_weighted)*100:.1f}%)")
#     print(f"   Test: {len(test_interactions):,} ({len(test_interactions)/len(interactions_weighted)*100:.1f}%)")
    
#     # ═════════════════════════════════════════════════════════════════════
#     # STEP 2: GENERATE EMBEDDINGS (using RAW data)
#     # ═════════════════════════════════════════════════════════════════════
    
#     print("\n" + "="*80)
#     print("STEP 2: EMBEDDING GENERATION (using RAW interactions)")
#     print("="*80)
    
#     embedding_gen = EmbeddingGenerator(config)
    
#     # Use RAW interactions (no time decay) for embeddings
#     post_embeddings = embedding_gen.generate_post_embeddings(data['post'])
#     user_embeddings = embedding_gen.generate_user_embeddings(
#         data['postreaction'],  # ← RAW data, not weighted
#         post_embeddings
#     )
    
#     embeddings = {
#         'post': post_embeddings,
#         'user': user_embeddings
#     }
    
#     print(f"✅ Embeddings generated:")
#     print(f"   Posts: {len(post_embeddings):,}")
#     print(f"   Users: {len(user_embeddings):,}")
    
#     # ═════════════════════════════════════════════════════════════════════
#     # STEP 3: BUILD COLLABORATIVE FILTERING (using RAW data)
#     # ═════════════════════════════════════════════════════════════════════
    
#     print("\n" + "="*80)
#     print("STEP 3: COLLABORATIVE FILTERING (using RAW interactions)")
#     print("="*80)
    
#     cf_builder = CFBuilder(config)
    
#     # Use RAW interactions (no time decay) for CF
#     cf_model = cf_builder.build_cf_model(data['postreaction'])  # ← RAW data
    
#     print(f"✅ CF model built:")
#     print(f"   Users: {len(cf_model['user_ids']):,}")
#     print(f"   Posts: {len(cf_model['post_ids']):,}")
#     print(f"   User similarities computed: {len(cf_model['user_similarities']):,}")
#     print(f"   Item similarities computed: {len(cf_model['item_similarities']):,}")
    
#     # ═════════════════════════════════════════════════════════════════════
#     # STEP 4: TRAIN RANKING MODEL (using WEIGHTED data)
#     # ═════════════════════════════════════════════════════════════════════
    
#     print("\n" + "="*80)
#     print("STEP 4: RANKING MODEL TRAINING (using WEIGHTED interactions)")
#     print("="*80)
    
#     trainer = ModelTrainer(config)
    
#     # Prepare training data using WEIGHTED interactions
#     print("\n🔧 Preparing training data with time decay weights...")
    
#     train_df = trainer.prepare_training_data(
#         train_interactions,  # ← WEIGHTED data with 'weight' column
#         data, 
#         user_stats, 
#         author_stats, 
#         following_dict, 
#         embeddings
#     )
    
#     val_df = trainer.prepare_training_data(
#         val_interactions,  # ← WEIGHTED data
#         data, 
#         user_stats, 
#         author_stats, 
#         following_dict, 
#         embeddings
#     )
    
#     test_df = trainer.prepare_training_data(
#         test_interactions,  # ← WEIGHTED data
#         data, 
#         user_stats, 
#         author_stats, 
#         following_dict, 
#         embeddings
#     )
    
#     print(f"\n✅ Training data prepared:")
#     print(f"   Train samples: {len(train_df):,}")
#     print(f"   Val samples: {len(val_df):,}")
#     print(f"   Test samples: {len(test_df):,}")
    
#     # Check if 'weight' column exists in training data
#     if 'weight' in train_df.columns:
#         print(f"\n📊 Time decay weights in training data:")
#         print(f"   Mean: {train_df['weight'].mean():.4f}")
#         print(f"   Median: {train_df['weight'].median():.4f}")
#         print(f"   Min: {train_df['weight'].min():.4f}")
#         print(f"   Max: {train_df['weight'].max():.4f}")
    
#     # Train
#     print("\n🔧 Training LightGBM ranking model...")
#     ranking_model, ranking_scaler, ranking_feature_cols = trainer.train(train_df, val_df)
    
#     # Evaluate
#     print("\n📊 Evaluating on test set...")
#     test_metrics = trainer.evaluate(test_df)
    
#     # ═════════════════════════════════════════════════════════════════════
#     # STEP 5: SAVE ARTIFACTS
#     # ═════════════════════════════════════════════════════════════════════
    
#     print("\n" + "="*80)
#     print("STEP 5: SAVING ARTIFACTS")
#     print("="*80)
    
#     artifact_mgr = ArtifactManager(config['output']['models_dir'])
    
#     version = artifact_mgr.create_version()
    
#     # Prepare metadata
#     metadata = {
#         'test_metrics': test_metrics,
#         'n_train_samples': len(train_df),
#         'n_val_samples': len(val_df),
#         'n_test_samples': len(test_df),
#         'config': config,
#         'time_decay': {
#             'enabled': True,
#             'half_life_days': data_loader_config['half_life_days'],
#             'min_weight': data_loader_config['min_weight'],
#             'weight_stats': {
#                 'mean': float(interactions_weighted['weight'].mean()),
#                 'median': float(interactions_weighted['weight'].median()),
#                 'min': float(interactions_weighted['weight'].min()),
#                 'max': float(interactions_weighted['weight'].max())
#             }
#         },
#         'data_stats': {
#             'n_users': len(data['user']),
#             'n_posts': len(data['post']),
#             'n_raw_interactions': len(data['postreaction']),
#             'n_weighted_interactions': len(interactions_weighted),
#             'n_embeddings_post': len(post_embeddings),
#             'n_embeddings_user': len(user_embeddings),
#             'n_cf_users': len(cf_model['user_ids']),
#             'n_cf_posts': len(cf_model['post_ids'])
#         }
#     }
    
#     artifact_mgr.save_artifacts(
#         version=version,
#         embeddings=embeddings,
#         cf_model=cf_model,
#         ranking_model=ranking_model,
#         ranking_scaler=ranking_scaler,
#         ranking_feature_cols=ranking_feature_cols,
#         user_stats=user_stats,
#         author_stats=author_stats,
#         following_dict=following_dict,
#         metadata=metadata
#     )
    
#     # Cleanup old versions
#     artifact_mgr.cleanup_old_versions(
#         keep_n=config['output']['keep_last_n_versions']
#     )
    
#     # ═════════════════════════════════════════════════════════════════════
#     # COMPLETE
#     # ═════════════════════════════════════════════════════════════════════
    
#     print("\n" + "="*80)
#     print("✅ OFFLINE TRAINING COMPLETE!")
#     print("="*80)
#     print(f"Version: {version}")
#     print(f"\n📊 Final Metrics:")
#     print(f"   Test AUC: {test_metrics['auc']:.4f}")
#     print(f"   Test Precision@10: {test_metrics['precision@10']:.4f}")
#     print(f"   Test Precision@20: {test_metrics['precision@20']:.4f}")
#     print(f"   Test Precision@50: {test_metrics['precision@50']:.4f}")
#     print(f"\n💾 Artifacts saved to: models/{version}/")
#     print(f"   - Embeddings (RAW data)")
#     print(f"   - CF Model (RAW data)")
#     print(f"   - Ranking Model (WEIGHTED data)")
#     print(f"   - Statistics & Metadata")
#     print(f"\n🕐 Completed at: {datetime.now()}")
#     print("="*80)


# if __name__ == "__main__":
#     main()


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

import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ============================ Small helpers ==================================
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _make_engine(cfg: Dict[str, Any]):
    db_url = (cfg.get("database", {}) or {}).get("url", "")
    if not db_url:
        return None
    # requires: pip install sqlalchemy pymysql
    return create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=int(cfg.get("database", {}).get("pool_size", 5)),
        max_overflow=int(cfg.get("database", {}).get("max_overflow", 5)),
        future=True,
    )

def _temporal_split(df: pd.DataFrame, test_days: int, val_days: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "created_at" not in df.columns:
        raise ValueError("interactions must include 'created_at' for temporal split")
    d = df.copy()
    d["created_at"] = pd.to_datetime(d["created_at"], utc=True, errors="coerce")
    d = d.sort_values("created_at")
    maxd = d["created_at"].max()
    test_start = maxd - timedelta(days=test_days)
    val_start = test_start - timedelta(days=val_days)
    train_df = d[d["created_at"] < val_start]
    val_df = d[(d["created_at"] >= val_start) & (d["created_at"] < test_start)]
    test_df = d[d["created_at"] >= test_start]
    return train_df, val_df, test_df

def _compute_time_decay_weights(
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


# ============================ Main ===========================================
def main():
    print("=" * 80)
    print("🚀 OFFLINE TRAINING PIPELINE (WITH TIME DECAY)")
    print("=" * 80)
    start_ts = _now_utc()
    print(f"Started at: {start_ts}")

    # 0) Load config
    cfg = _load_yaml("configs/config_offline.yaml")
    # optional override by configs/config.yaml
    cfg2 = _load_yaml("configs/config.yaml")
    for k, v in cfg2.items():
        cfg.setdefault(k, v)

    train_cfg = {
        "window_days": int(cfg.get("training", {}).get("window_days", 90)),
        "half_life_days": int(cfg.get("training", {}).get("half_life_days", 7)),
        "pretrain_full_export": bool(cfg.get("training", {}).get("pretrain_full_export", True)),
        "overlap_days": int(cfg.get("training", {}).get("overlap_days", 2)),
        "chunk_size": int(cfg.get("training", {}).get("chunk_size", 100_000)),
        "test_days": int(cfg.get("training", {}).get("test_days", 3)),
        "val_days": int(cfg.get("training", {}).get("val_days", 3)),
        "action_multipliers": (cfg.get("training", {}).get("action_multipliers") or {
            "view": 0.5,
            "like": 1.0, "love": 1.3, "care": 1.25, "laugh": 1.2, "wow": 1.1, "sad": 0.9, "angry": 0.9,
            "comment": 1.5, "share": 2.0, "save": 1.2,
        }),
    }
    artifacts_cfg = {
        "base_dir": cfg.get("artifacts", {}).get("base_dir", "models"),
        "state_file": cfg.get("artifacts", {}).get("state_file", "models/training_state.json"),
    }

    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING (DUAL STRATEGY)")
    print("=" * 80)

    # 1) MySQL engine
    engine = _make_engine(cfg)
    if engine is None:
        raise RuntimeError(
            "❌ Database URL is missing. Please set configs/config_offline.yaml:\n"
            "database:\n"
            "url: mysql+pymysql://USER:PASS@HOST:3306/DBNAME?charset=utf8mb4"
        )

    # 2) Training state (incremental)
    state_mgr = TrainingStateManager(artifacts_cfg["state_file"])
    state = state_mgr.load()
    now = _now_utc()

    if state.last_train_end is None:
        # First run => lấy full history (data ~2 tháng)
        if train_cfg["pretrain_full_export"]:
            since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        else:
            since = now - timedelta(days=train_cfg["window_days"])
    else:
        last = pd.to_datetime(state.last_train_end, utc=True)
        since = last - timedelta(days=train_cfg["overlap_days"])  # overlap để an toàn

    until = now

    # 3) DataLoader (FORCE DB mode — không ép CSV)
    data_loader = DataLoader(
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
            # CSV settings kept for fallback only; not used if db_connection is provided.
            "csv_dir": (cfg.get("dataset", {}) or {}).get("dir", "dataset"),
            "csv_files": (cfg.get("dataset", {}) or {}),
        },
    )

    print("\n📦 Loading RAW data (for Embeddings & CF)...")
    raw_bundle = data_loader.load_training_bundle(since=since, until=until, use_csv=False)

    # (Optional) thống kê raw để debug như log cũ
    print("\nComputing Statistics.")
    print("Computing user stats.")
    users_df = raw_bundle["users"]
    posts_df = raw_bundle["posts"]
    inter_df = raw_bundle["interactions"]
    friendships_df = raw_bundle["friendships"]

    n_users = 0 if users_df is None or users_df.empty else users_df["Id"].nunique()
    print(f"User stats for {n_users} users")

    print("Computing author stats.")
    n_authors = 0 if posts_df is None or posts_df.empty else posts_df["UserId"].nunique()
    print(f"Author stats for {n_authors} authors")

    print("Building following dictionary.")
    n_following_users = 0 if friendships_df is None or friendships_df.empty else friendships_df["UserId"].nunique()
    print(f"Following dict for {n_following_users} users")

    print("✅ Raw data loaded:")
    print(f"   Users: {n_users:,}")
    n_posts = 0 if posts_df is None or posts_df.empty else posts_df["Id"].nunique()
    print(f"   Posts: {n_posts:,}")
    n_inter = 0 if inter_df is None or inter_df.empty else len(inter_df)
    print(f"   Interactions: {n_inter:,}")
    n_friends = 0 if friendships_df is None or friendships_df.empty else len(friendships_df)
    print(f"   Friendships: {n_friends:,}")

    print("\n📦 Loading WEIGHTED data (for Ranking Model)...")
    # Ở thiết kế mới: dùng ngay raw_bundle (vì đã là DB), ranking sẽ tính time-decay và weight ở dưới.

    # ======================= FE (vector) + temporal split =====================
    X, y, interactions_for_split, meta = build_training_matrices(
        interactions=inter_df,
        users_df=users_df,
        posts_df=posts_df,
        friendships_df=friendships_df,
        post_hashtags_df=raw_bundle["post_hashtags"],
        embeddings=None,
        now_ts=now,
    )

    tmp = interactions_for_split.copy()
    tmp["label"] = y.values
    tmp["__rid__"] = np.arange(len(tmp))
    tr_i, va_i, te_i = _temporal_split(tmp, test_days=train_cfg["test_days"], val_days=train_cfg["val_days"])

    rid_tr = tr_i["__rid__"].to_numpy()
    rid_va = va_i["__rid__"].to_numpy()
    rid_te = te_i["__rid__"].to_numpy()

    X_train, y_train, inter_train = X.iloc[rid_tr], y.iloc[rid_tr], interactions_for_split.iloc[rid_tr]
    X_val,   y_val,   inter_val   = X.iloc[rid_va], y.iloc[rid_va], interactions_for_split.iloc[rid_va]
    X_test,  y_test,  inter_test  = X.iloc[rid_te], y.iloc[rid_te], interactions_for_split.iloc[rid_te]

    # ======================= Time-decay weights ===============================
    w_train = _compute_time_decay_weights(inter_train, train_cfg["half_life_days"], train_cfg["action_multipliers"], now)
    w_val   = _compute_time_decay_weights(inter_val,   train_cfg["half_life_days"], train_cfg["action_multipliers"], now)

    train_df = X_train.copy(); train_df["label"] = y_train.values; train_df["weight"] = w_train.values
    val_df   = X_val.copy();   val_df["label"]   = y_val.values;   val_df["weight"]   = w_val.values
    test_df  = X_test.copy();  test_df["label"]  = y_test.values   # no weight for test

    # ======================= Train/Eval =======================================
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING")
    print("=" * 80)

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

    # ======================= Save artifacts ===================================
    print("\n" + "=" * 80)
    print("STEP 3: SAVE ARTIFACTS")
    print("=" * 80)

    version = now.strftime("v%Y%m%d_%H%M%S")
    meta_out = {
        "version": version,
        "trained_at": now.isoformat(),
        "rows": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "features": feature_cols,
        "metrics": metrics,
        "window_days": train_cfg["window_days"],
        "half_life_days": train_cfg["half_life_days"],
        "split": {"test_days": train_cfg["test_days"], "val_days": train_cfg["val_days"]},
    }
    save_artifacts(version_name=version, model=model, meta=meta_out, artifacts_base_dir=artifacts_cfg["base_dir"])

    # (Optional) keep compatibility with your trainer.save_model(...)
    out_base = str(Path(artifacts_cfg["base_dir"]) / version / "ranker")
    trainer.save_model(out_base)

    # ======================= Update training state ============================
    state.last_train_end = TrainingStateManager.now_iso()
    state_mgr.save(state)

    print("\n✅ Done.")
    print(f"Version: {version}")
    print(f"Metrics: {json.dumps(metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
