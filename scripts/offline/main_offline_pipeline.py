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


# # scripts/offline/main_offline_pipeline.py
# from __future__ import annotations

# import json
# from datetime import datetime, timedelta, timezone
# from pathlib import Path
# from typing import Dict, Any, Tuple

# import numpy as np
# import pandas as pd
# import yaml
# from sqlalchemy import create_engine

# from recommender.common.data_loading import DataLoader
# from recommender.common.feature_engineer import build_training_matrices
# from recommender.offline.training_state import TrainingStateManager
# from recommender.offline.model_trainer import ModelTrainer
# from recommender.offline.artifact_manager import save_artifacts

# import os, sys
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
# if ROOT_DIR not in sys.path:
#     sys.path.insert(0, ROOT_DIR)

# # ============================ Small helpers ==================================
# def _now_utc() -> datetime:
#     return datetime.now(timezone.utc)

# def _load_yaml(path: str) -> Dict[str, Any]:
#     p = Path(path)
#     if not p.exists():
#         return {}
#     with p.open("r", encoding="utf-8") as f:
#         return yaml.safe_load(f) or {}

# def _make_engine(cfg: Dict[str, Any]):
#     db_url = (cfg.get("database", {}) or {}).get("url", "")
#     if not db_url:
#         return None
#     # requires: pip install sqlalchemy pymysql
#     return create_engine(
#         db_url,
#         pool_pre_ping=True,
#         pool_size=int(cfg.get("database", {}).get("pool_size", 5)),
#         max_overflow=int(cfg.get("database", {}).get("max_overflow", 5)),
#         future=True,
#     )

# def _temporal_split(df: pd.DataFrame, test_days: int, val_days: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     if "created_at" not in df.columns:
#         raise ValueError("interactions must include 'created_at' for temporal split")
#     d = df.copy()
#     d["created_at"] = pd.to_datetime(d["created_at"], utc=True, errors="coerce")
#     d = d.sort_values("created_at")
#     maxd = d["created_at"].max()
#     test_start = maxd - timedelta(days=test_days)
#     val_start = test_start - timedelta(days=val_days)
#     train_df = d[d["created_at"] < val_start]
#     val_df = d[(d["created_at"] >= val_start) & (d["created_at"] < test_start)]
#     test_df = d[d["created_at"] >= test_start]
#     return train_df, val_df, test_df

# def _compute_time_decay_weights(
#     inter_df: pd.DataFrame,
#     half_life_days: int,
#     action_multipliers: Dict[str, float],
#     now_ts: datetime,
# ) -> pd.Series:
#     """
#     weight = action_multiplier * 0.5 ** (age_days / half_life)
#     """
#     ts = pd.to_datetime(inter_df["created_at"], utc=True, errors="coerce")
#     age_days = (now_ts - ts).dt.total_seconds() / 86400.0
#     decay = np.power(0.5, np.clip(age_days, 0, None) / max(half_life_days, 1))
#     mult = inter_df["action"].map(action_multipliers).fillna(1.0).astype(float)
#     w = np.maximum(decay * mult.values, 1e-3)
#     return pd.Series(w, index=inter_df.index)


# # ============================ Main ===========================================
# def main():
#     print("=" * 80)
#     print("🚀 OFFLINE TRAINING PIPELINE (WITH TIME DECAY)")
#     print("=" * 80)
#     start_ts = _now_utc()
#     print(f"Started at: {start_ts}")

#     # 0) Load config
#     cfg = _load_yaml("configs/config_offline.yaml")
#     # optional override by configs/config.yaml
#     cfg2 = _load_yaml("configs/config.yaml")
#     for k, v in cfg2.items():
#         cfg.setdefault(k, v)

#     train_cfg = {
#         "window_days": int(cfg.get("training", {}).get("window_days", 90)),
#         "half_life_days": int(cfg.get("training", {}).get("half_life_days", 7)),
#         "pretrain_full_export": bool(cfg.get("training", {}).get("pretrain_full_export", True)),
#         "overlap_days": int(cfg.get("training", {}).get("overlap_days", 2)),
#         "chunk_size": int(cfg.get("training", {}).get("chunk_size", 100_000)),
#         "test_days": int(cfg.get("training", {}).get("test_days", 3)),
#         "val_days": int(cfg.get("training", {}).get("val_days", 3)),
#         "action_multipliers": (cfg.get("training", {}).get("action_multipliers") or {
#             "view": 0.5,
#             "like": 1.0, "love": 1.3, "care": 1.25, "laugh": 1.2, "wow": 1.1, "sad": 0.9, "angry": 0.9,
#             "comment": 1.5, "share": 2.0, "save": 1.2,
#         }),
#     }
#     artifacts_cfg = {
#         "base_dir": cfg.get("artifacts", {}).get("base_dir", "models"),
#         "state_file": cfg.get("artifacts", {}).get("state_file", "models/training_state.json"),
#     }

#     print("\n" + "=" * 80)
#     print("STEP 1: DATA LOADING (DUAL STRATEGY)")
#     print("=" * 80)

#     # 1) MySQL engine
#     engine = _make_engine(cfg)
#     if engine is None:
#         raise RuntimeError(
#             "❌ Database URL is missing. Please set configs/config_offline.yaml:\n"
#             "database:\n"
#             "url: mysql+pymysql://USER:PASS@HOST:3306/DBNAME?charset=utf8mb4"
#         )

#     # 2) Training state (incremental)
#     state_mgr = TrainingStateManager(artifacts_cfg["state_file"])
#     state = state_mgr.load()
#     now = _now_utc()

#     if state.last_train_end is None:
#         # First run => lấy full history (data ~2 tháng)
#         if train_cfg["pretrain_full_export"]:
#             since = datetime(2024, 1, 1, tzinfo=timezone.utc)
#         else:
#             since = now - timedelta(days=train_cfg["window_days"])
#     else:
#         last = pd.to_datetime(state.last_train_end, utc=True)
#         since = last - timedelta(days=train_cfg["overlap_days"])  # overlap để an toàn

#     until = now

#     # 3) DataLoader (FORCE DB mode — không ép CSV)
#     data_loader = DataLoader(
#         db_connection=engine,
#         config={
#             "lookback_days": train_cfg["window_days"],
#             "chunk_size": train_cfg["chunk_size"],
#             "tables": (cfg.get("database", {}).get("tables", {}) or {}),
#             "reaction_code_map": (cfg.get("database", {}).get("reaction_code_map", {}) or {
#                 "like": "like", "love": "love", "laugh": "laugh", "wow": "wow", "sad": "sad", "angry": "angry", "care": "care"
#             }),
#             "reaction_name_map": (cfg.get("database", {}).get("reaction_name_map", {}) or {
#                 "Like": "like", "Love": "love", "Laugh": "laugh", "Wow": "wow", "Sad": "sad", "Angry": "angry", "Care": "care"
#             }),
#             # CSV settings kept for fallback only; not used if db_connection is provided.
#             "csv_dir": (cfg.get("dataset", {}) or {}).get("dir", "dataset"),
#             "csv_files": (cfg.get("dataset", {}) or {}),
#         },
#     )

#     print("\n📦 Loading RAW data (for Embeddings & CF)...")
#     raw_bundle = data_loader.load_training_bundle(since=since, until=until, use_csv=False)

#     # (Optional) thống kê raw để debug như log cũ
#     print("\nComputing Statistics.")
#     print("Computing user stats.")
#     users_df = raw_bundle["users"]
#     posts_df = raw_bundle["posts"]
#     inter_df = raw_bundle["interactions"]
#     friendships_df = raw_bundle["friendships"]

#     n_users = 0 if users_df is None or users_df.empty else users_df["Id"].nunique()
#     print(f"User stats for {n_users} users")

#     print("Computing author stats.")
#     n_authors = 0 if posts_df is None or posts_df.empty else posts_df["UserId"].nunique()
#     print(f"Author stats for {n_authors} authors")

#     print("Building following dictionary.")
#     n_following_users = 0 if friendships_df is None or friendships_df.empty else friendships_df["UserId"].nunique()
#     print(f"Following dict for {n_following_users} users")

#     print("✅ Raw data loaded:")
#     print(f"   Users: {n_users:,}")
#     n_posts = 0 if posts_df is None or posts_df.empty else posts_df["Id"].nunique()
#     print(f"   Posts: {n_posts:,}")
#     n_inter = 0 if inter_df is None or inter_df.empty else len(inter_df)
#     print(f"   Interactions: {n_inter:,}")
#     n_friends = 0 if friendships_df is None or friendships_df.empty else len(friendships_df)
#     print(f"   Friendships: {n_friends:,}")

#     print("\n📦 Loading WEIGHTED data (for Ranking Model)...")
#     # Ở thiết kế mới: dùng ngay raw_bundle (vì đã là DB), ranking sẽ tính time-decay và weight ở dưới.

#     # ======================= FE (vector) + temporal split =====================
#     X, y, interactions_for_split, meta = build_training_matrices(
#         interactions=inter_df,
#         users_df=users_df,
#         posts_df=posts_df,
#         friendships_df=friendships_df,
#         post_hashtags_df=raw_bundle["post_hashtags"],
#         embeddings=None,
#         now_ts=now,
#     )

#     tmp = interactions_for_split.copy()
#     tmp["label"] = y.values
#     tmp["__rid__"] = np.arange(len(tmp))
#     tr_i, va_i, te_i = _temporal_split(tmp, test_days=train_cfg["test_days"], val_days=train_cfg["val_days"])

#     rid_tr = tr_i["__rid__"].to_numpy()
#     rid_va = va_i["__rid__"].to_numpy()
#     rid_te = te_i["__rid__"].to_numpy()

#     X_train, y_train, inter_train = X.iloc[rid_tr], y.iloc[rid_tr], interactions_for_split.iloc[rid_tr]
#     X_val,   y_val,   inter_val   = X.iloc[rid_va], y.iloc[rid_va], interactions_for_split.iloc[rid_va]
#     X_test,  y_test,  inter_test  = X.iloc[rid_te], y.iloc[rid_te], interactions_for_split.iloc[rid_te]

#     # ======================= Time-decay weights ===============================
#     w_train = _compute_time_decay_weights(inter_train, train_cfg["half_life_days"], train_cfg["action_multipliers"], now)
#     w_val   = _compute_time_decay_weights(inter_val,   train_cfg["half_life_days"], train_cfg["action_multipliers"], now)

#     train_df = X_train.copy(); train_df["label"] = y_train.values; train_df["weight"] = w_train.values
#     val_df   = X_val.copy();   val_df["label"]   = y_val.values;   val_df["weight"]   = w_val.values
#     test_df  = X_test.copy();  test_df["label"]  = y_test.values   # no weight for test

#     # ======================= Train/Eval =======================================
#     print("\n" + "=" * 80)
#     print("STEP 2: TRAINING")
#     print("=" * 80)

#     trainer = ModelTrainer(
#         config={
#             "model": {
#                 "params": {
#                     "objective": "binary",
#                     "metric": "auc",
#                     "learning_rate": 0.05,
#                     "num_leaves": 64,
#                     "feature_fraction": 0.8,
#                     "bagging_fraction": 0.8,
#                     "bagging_freq": 1,
#                     "verbose": -1,
#                 },
#                 "num_boost_round": 300,
#                 "early_stopping_rounds": 50,
#                 "log_every_n": 50,
#             },
#             "training": {
#                 "half_life_days": train_cfg["half_life_days"],
#                 "min_weight": 1e-3,
#                 "use_action_multiplier": True,
#                 "action_multipliers": train_cfg["action_multipliers"],
#             },
#         }
#     )

#     model, scaler, feature_cols = trainer.train(train_df, val_df)
#     metrics = trainer.evaluate(test_df)

#     # ======================= Save artifacts ===================================
#     print("\n" + "=" * 80)
#     print("STEP 3: SAVE ARTIFACTS")
#     print("=" * 80)

#     version = now.strftime("v%Y%m%d_%H%M%S")
#     meta_out = {
#         "version": version,
#         "trained_at": now.isoformat(),
#         "rows": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
#         "features": feature_cols,
#         "metrics": metrics,
#         "window_days": train_cfg["window_days"],
#         "half_life_days": train_cfg["half_life_days"],
#         "split": {"test_days": train_cfg["test_days"], "val_days": train_cfg["val_days"]},
#     }
#     save_artifacts(version_name=version, model=model, meta=meta_out, artifacts_base_dir=artifacts_cfg["base_dir"])

#     # (Optional) keep compatibility with your trainer.save_model(...)
#     out_base = str(Path(artifacts_cfg["base_dir"]) / version / "ranker")
#     trainer.save_model(out_base)

#     # ======================= Update training state ============================
#     state.last_train_end = TrainingStateManager.now_iso()
#     state_mgr.save(state)

#     print("\n✅ Done.")
#     print(f"Version: {version}")
#     print(f"Metrics: {json.dumps(metrics, ensure_ascii=False)}")


# if __name__ == "__main__":
#     main()



# scripts/offline/main_offline_pipeline.py
# =============================================================================
# OFFLINE TRAINING PIPELINE (FULL/WINDOW/INCREMENTAL + TIME-DECAY)
# =============================================================================
# - Chế độ: full | window | incremental (CLI override)
# - Enforce TRAIN_WINDOW_DAYS=14 khi mode=window (Priority Fix #1)
# - Overlap cho incremental/window để tránh "đứt quãng"
# - Watermark lưu tại artifacts.state_file
# - Split theo thời gian: train / val_days / test_days
# - Robust khi dữ liệu mỏng (kết hợp guard trong ModelTrainer.train)
#
# Yêu cầu:
#   pip install pyyaml sqlalchemy python-dateutil pandas numpy lightgbm scikit-learn
#
# Chạy ví dụ:
#   python -m scripts.offline.main_offline_pipeline --config configs/offline.yaml --mode full
#   python -m scripts.offline.main_offline_pipeline --mode window --window-days 14
#   python -m scripts.offline.main_offline_pipeline --mode incremental
#   python -m scripts.offline.main_offline_pipeline --since 2025-10-01T00:00:00Z
# =============================================================================

from __future__ import annotations

import os
import sys
import json
import math
import time
import yaml
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from dateutil import parser as dparser
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# allow imports from project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from recommender.offline.model_trainer import ModelTrainer
from recommender.offline.artifact_manager import (
    ArtifactManager,
    write_latest_pointer,
)
from recommender.common.feature_engineer import POSITIVE_ACTIONS as POS_ACT


# ----------------------------- Logging ----------------------------------------
logger = logging.getLogger("offline_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --------------------------- Helpers (time/window) ----------------------------
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _load_state(state_file: str) -> dict:
    if not state_file or not os.path.exists(state_file):
        return {}
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_state(state_file: str, state: dict):
    if not state_file:
        return
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def resolve_time_range(cfg: dict, args) -> tuple[Optional[datetime], datetime, str, int]:
    """
    Return (start_ts, end_ts, mode_used, window_days_for_log)
      full: start=None, end=now
      window: start=now-window_days, end=now (enforce 14 by default)
      incremental: start=watermark-overlap_days, end=now (fallback window if no watermark)
      since: if args.since provided → override all.
    """
    tr = (cfg.get("training") or {})
    default_mode = (tr.get("mode") or "window").lower()
    mode = (args.mode or default_mode).lower()

    now = _utcnow()
    # CLI override since/until
    if args.since:
        start = dparser.parse(args.since)
        start = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
        end = dparser.parse(args.until).astimezone(timezone.utc) if args.until else now
        return start, end, "since", 0

    state_file = (cfg.get("artifacts") or {}).get("state_file")
    state = _load_state(state_file)
    window_days_cfg = int(args.window_days) if args.window_days else int(tr.get("window_days", 14))

    if mode == "full":
        return None, now, "full", 0

    if mode == "incremental":
        overlap_days = int(tr.get("overlap_days", 2))
        wm = state.get("watermark_ts")
        if wm:
            start = dparser.parse(wm)
            start = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
            start = start - timedelta(days=overlap_days)
            return start, now, "incremental", window_days_cfg
        mode = "window"  # fallback khi chưa có watermark

    # window (default) — enforce 14 ngày theo Priority Fix #1 (trừ khi CLI override)
    window_days = window_days_cfg if args.window_days else 14
    start = now - timedelta(days=window_days)
    return start, now, "window", window_days

def update_watermark_after_load(cfg: dict, max_ts: Optional[datetime]):
    state_file = (cfg.get("artifacts") or {}).get("state_file")
    if not state_file or max_ts is None:
        return
    state = _load_state(state_file)
    iso = max_ts.astimezone(timezone.utc).isoformat()
    state["watermark_ts"] = iso
    _save_state(state_file, state)

# --------------------------- DB / CSV Loaders ---------------------------------
def _build_engine(db_url: str | None) -> Optional[Engine]:
    if not db_url:
        return None
    try:
        engine = create_engine(db_url, future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        logger.warning(f"DB connect failed: {e}")
        return None

def _ts_clause(col: str, start_ts: Optional[datetime], end_ts: datetime) -> str:
    if start_ts is None:
        return f"{col} <= :end_ts"
    return f"{col} BETWEEN :start_ts AND :end_ts"

def _exec_df(conn, sql: str, params: dict) -> pd.DataFrame:
    return pd.read_sql(text(sql), conn, params=params)

def load_bundle_from_db(
    db_url: str,
    tables_cfg: dict,
    start_ts: Optional[datetime],
    end_ts: datetime,
    chunk_size: int = 100000,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], int, datetime, datetime]:
    """
    Load interactions (PostReaction + Comment + PostView) + entities (User/Post/Friendship/PostHashtag)
    Return:
      interactions_df, data_dict, total_interactions, min_created_at, max_created_at
    interactions_df columns:
      user_id, post_id, action, created_at, reaction_type_id (optional)
    """
    engine = _build_engine(db_url)
    if engine is None:
        raise RuntimeError("No DB engine. Please check database.url in config.")

    t_post_view = tables_cfg.get("post_view", "PostView")
    t_post_reaction = tables_cfg.get("post_reaction", "PostReaction")
    t_reaction_type = tables_cfg.get("reaction_type", "ReactionType")
    t_user = tables_cfg.get("user", "User")
    t_post = tables_cfg.get("post", "Post")
    t_friendship = tables_cfg.get("friendship", "Friendship")
    t_post_hashtag = tables_cfg.get("post_hashtag", "PostHashtag")
    t_comment = tables_cfg.get("comment", "Comment")

    with engine.connect() as conn:
        logger.info("DB: loading PostView ...")
        sql_view = f"""
            SELECT v.Id AS id, v.UserId AS user_id, v.PostId AS post_id, v.CreateDate AS created_at,
                   p.UserId AS author_id, v.Status AS v_status, p.Status AS p_status
            FROM {t_post_view} v
            LEFT JOIN {t_post} p ON p.Id = v.PostId
            WHERE { _ts_clause("v.CreateDate", start_ts, end_ts) }
            ORDER BY v.Id ASC
        """
        df_view = _exec_df(conn, sql_view, {"start_ts": start_ts, "end_ts": end_ts})
        if not df_view.empty:
            df_view = df_view[(df_view["v_status"].isna()) | (df_view["v_status"] == 10)]
            df_view = df_view[(df_view["p_status"].isna()) | (df_view["p_status"] == 10)]
            df_view["action"] = "view"
            df_view["reaction_type_id"] = np.nan
        logger.info(f"  -> loaded {len(df_view)} rows")

        logger.info("DB: loading PostReaction ...")
        sql_react = f"""
            SELECT pr.Id AS id, pr.UserId AS user_id, pr.PostId AS post_id, pr.CreateDate AS created_at,
                   pr.ReactionTypeId AS reaction_type_id,
                   p.UserId AS author_id, pr.Status AS pr_status, p.Status AS p_status
            FROM {t_post_reaction} pr
            LEFT JOIN {t_post} p ON p.Id = pr.PostId
            WHERE { _ts_clause("pr.CreateDate", start_ts, end_ts) }
            ORDER BY pr.Id ASC
        """
        df_react = _exec_df(conn, sql_react, {"start_ts": start_ts, "end_ts": end_ts})
        if not df_react.empty:
            df_react = df_react[(df_react["pr_status"].isna()) | (df_react["pr_status"] == 10)]
            df_react = df_react[(df_react["p_status"].isna()) | (df_react["p_status"] == 10)]

        # Map ReactionTypeId -> action
        react_map = {
            1: "like", 2: "love", 3: "laugh", 4: "wow", 5: "sad",
            6: "angry", 7: "care", 8: "save", 9: "share"
        }
        if not df_react.empty:
            df_react["action"] = df_react["reaction_type_id"].map(lambda x: react_map.get(int(x), "like") if pd.notna(x) else "like")

        logger.info("DB: loading Comment ...")
        sql_cmt = f"""
            SELECT c.Id AS id, c.UserId AS user_id, c.PostId AS post_id, c.CreateDate AS created_at,
                   p.UserId AS author_id, c.Status AS c_status, p.Status AS p_status
            FROM {t_comment} c
            LEFT JOIN {t_post} p ON p.Id = c.PostId
            WHERE { _ts_clause("c.CreateDate", start_ts, end_ts) }
            ORDER BY c.Id ASC
        """
        df_cmt = _exec_df(conn, sql_cmt, {"start_ts": start_ts, "end_ts": end_ts})
        if not df_cmt.empty:
            df_cmt = df_cmt[(df_cmt["c_status"].isna()) | (df_cmt["c_status"] == 10)]
            df_cmt = df_cmt[(df_cmt["p_status"].isna()) | (df_cmt["p_status"] == 10)]
            df_cmt["action"] = "comment"
            df_cmt["reaction_type_id"] = np.nan
        logger.info(f"  -> loaded {len(df_cmt)} rows")

        # entities
        logger.info("DB: loading User ...")
        df_user = _exec_df(conn, f"SELECT * FROM {t_user}", {})
        logger.info("DB: loading Post ...")
        df_post = _exec_df(conn, f"SELECT * FROM {t_post}", {})
        logger.info("DB: loading Friendship ...")
        df_friend = _exec_df(conn, f"SELECT * FROM {t_friendship}", {})
        logger.info("DB: loading PostHashtag ...")
        df_ph = _exec_df(conn, f"SELECT * FROM {t_post_hashtag}", {})

    # normalize types
    for df in [df_view, df_react, df_cmt]:
        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
            df["user_id"] = df["user_id"].astype("Int64")
            df["post_id"] = df["post_id"].astype("Int64")

    # concat
    interactions = []
    if not df_view.empty:
        interactions.append(df_view[["user_id", "post_id", "action", "created_at", "reaction_type_id"]])
    if not df_react.empty:
        interactions.append(df_react[["user_id", "post_id", "action", "created_at", "reaction_type_id"]])
    if not df_cmt.empty:
        interactions.append(df_cmt[["user_id", "post_id", "action", "created_at", "reaction_type_id"]])

    if not interactions:
        interactions_df = pd.DataFrame(columns=["user_id", "post_id", "action", "created_at", "reaction_type_id"])
    else:
        interactions_df = pd.concat(interactions, axis=0, ignore_index=True)

    if interactions_df.empty:
        logger.warning("No interactions loaded in selected time range.")
        min_ts = start_ts
        max_ts = end_ts
    else:
        min_ts = interactions_df["created_at"].min()
        max_ts = interactions_df["created_at"].max()

    data_dict = {
        "users": df_user,
        "posts": df_post,
        "friendships": df_friend,
        "post_hashtags": df_ph
    }
    total = len(interactions_df)
    return interactions_df, data_dict, total, min_ts, max_ts


def load_bundle_from_csv(
    data_dir: str,
    start_ts: Optional[datetime],
    end_ts: datetime,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], int, datetime, datetime]:
    """
    Simple CSV loader fallback (expects User.csv, Post.csv, PostReaction.csv, Friendship.csv).
    """
    p = Path(data_dir)
    users = pd.read_csv(p / "User.csv")
    posts = pd.read_csv(p / "Post.csv")
    fr = pd.read_csv(p / "Friendship.csv")
    pr = pd.read_csv(p / "PostReaction.csv")
    # optional
    try:
        pv = pd.read_csv(p / "PostView.csv")
    except Exception:
        pv = pd.DataFrame(columns=["Id", "UserId", "PostId", "CreateDate", "Status"])
    try:
        cm = pd.read_csv(p / "Comment.csv")
    except Exception:
        cm = pd.DataFrame(columns=["Id", "UserId", "PostId", "CreateDate", "Status"])
    try:
        ph = pd.read_csv(p / "PostHashtag.csv")
    except Exception:
        ph = pd.DataFrame(columns=["Id", "PostId", "HashtagId"])

    # normalize
    for df, col in [(users, "CreateDate"), (posts, "CreateDate"), (pr, "CreateDate"), (pv, "CreateDate"), (cm, "CreateDate")]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # ReactionTypeId mapping → action
    react_map = {
        1: "like", 2: "love", 3: "laugh", 4: "wow", 5: "sad",
        6: "angry", 7: "care", 8: "save", 9: "share"
    }
    pr["action"] = pr.get("ReactionTypeId", np.nan).map(lambda x: react_map.get(int(x), "like") if pd.notna(x) else "like")
    pv["action"] = "view"
    cm["action"] = "comment"

    # filter by range
    def rng(df):
        if df.empty or "CreateDate" not in df.columns:
            return df
        if start_ts is None:
            return df[df["CreateDate"] <= end_ts]
        return df[(df["CreateDate"] >= start_ts) & (df["CreateDate"] <= end_ts)]

    pr = rng(pr)
    pv = rng(pv)
    cm = rng(cm)

    parts = []
    if not pv.empty:
        parts.append(pv.rename(columns={"UserId": "user_id", "PostId": "post_id", "CreateDate": "created_at"})[["user_id", "post_id", "action", "created_at"]])
    if not pr.empty:
        parts.append(pr.rename(columns={"UserId": "user_id", "PostId": "post_id", "CreateDate": "created_at", "ReactionTypeId": "reaction_type_id"})[["user_id", "post_id", "action", "created_at", "reaction_type_id"]])
    if not cm.empty:
        parts.append(cm.rename(columns={"UserId": "user_id", "PostId": "post_id", "CreateDate": "created_at"})[["user_id", "post_id", "action", "created_at"]])

    interactions_df = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame(columns=["user_id","post_id","action","created_at","reaction_type_id"])
    min_ts = interactions_df["created_at"].min() if not interactions_df.empty else start_ts
    max_ts = interactions_df["created_at"].max() if not interactions_df.empty else end_ts

    data_dict = {
        "users": users,
        "posts": posts,
        "friendships": fr,
        "post_hashtags": ph
    }
    return interactions_df, data_dict, len(interactions_df), min_ts, max_ts

# --------------------------- Stats / Following --------------------------------
def build_stats_and_following(
    interactions_df: pd.DataFrame,
    data_dict: Dict[str, pd.DataFrame],
) -> tuple[Dict[int, Dict], Dict[int, Dict], Dict[int, set]]:
    """
    user_stats: {user_id: {total_interactions, positive_rate}}
    author_stats: {author_id: {total_interactions, positive_rate}}
    following_dict: {user_id: set(author_ids_followed)}
    """
    # user stats
    df = interactions_df.copy()
    if df.empty:
        return {}, {}, {}

    df["label"] = df["action"].isin(POS_ACT).astype(int)
    user_stats = (
        df.groupby("user_id")["label"]
        .agg(total_interactions="count", positive_rate="mean")
        .reset_index()
    )
    user_stats_dict = {
        int(r["user_id"]): {"total_interactions": float(r["total_interactions"]), "positive_rate": float(r["positive_rate"])}
        for _, r in user_stats.iterrows()
    }

    # author mapping
    posts = data_dict.get("posts")
    post_to_author = {}
    if isinstance(posts, pd.DataFrame) and {"Id", "UserId"}.issubset(posts.columns):
        post_to_author = dict(zip(posts["Id"].astype("Int64"), posts["UserId"].astype("Int64")))

    df["author_id"] = df["post_id"].map(lambda x: int(post_to_author.get(int(x), -1)) if pd.notna(x) else -1)
    df_a = df[df["author_id"] >= 0]
    if df_a.empty:
        author_stats_dict = {}
    else:
        astats = (
            df_a.groupby("author_id")["label"]
            .agg(total_interactions="count", positive_rate="mean")
            .reset_index()
        )
        author_stats_dict = {
            int(r["author_id"]): {"total_interactions": float(r["total_interactions"]), "positive_rate": float(r["positive_rate"])}
            for _, r in astats.iterrows()
        }

    # following_dict from Friendship (UserId -> FriendId)
    follow = {}
    fr = data_dict.get("friendships")
    if isinstance(fr, pd.DataFrame) and {"UserId", "FriendId"}.issubset(fr.columns):
        for _, r in fr[["UserId", "FriendId"]].dropna().iterrows():
            u, v = int(r["UserId"]), int(r["FriendId"])
            follow.setdefault(u, set()).add(v)

    return user_stats_dict, author_stats_dict, follow

# --------------------------- Split train/val/test ------------------------------
def split_by_time(
    interactions_df: pd.DataFrame,
    val_days: int,
    test_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split using 'created_at'
    """
    if interactions_df.empty:
        return interactions_df, interactions_df, interactions_df

    df = interactions_df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.sort_values("created_at").reset_index(drop=True)

    max_ts = df["created_at"].max()
    test_cut = max_ts - timedelta(days=test_days) if test_days > 0 else max_ts + timedelta(seconds=1)
    val_cut = test_cut - timedelta(days=val_days) if val_days > 0 else test_cut

    test_df = df[df["created_at"] > test_cut]
    val_df = df[(df["created_at"] > val_cut) & (df["created_at"] <= test_cut)]
    train_df = df[df["created_at"] <= val_cut]

    return train_df, val_df, test_df

# --------------------------- Save artifacts / meta -----------------------------
def _make_version_name() -> str:
    return datetime.utcnow().strftime("v%Y%m%d_%H%M%S")

def _write_meta(version_dir: Path, meta: dict):
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

# --------------------------------- MAIN ---------------------------------------
def main():
    print("=" * 80)
    print("🚀 OFFLINE TRAINING PIPELINE (WITH TIME DECAY)")
    print("=" * 80)
    print(f"Started at: {_utcnow()}\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_offline.yaml")
    parser.add_argument("--mode", choices=["full", "window", "incremental"], default=None)
    parser.add_argument("--window-days", type=int, default=None)
    parser.add_argument("--since", type=str, default=None)  # ISO 8601
    parser.add_argument("--until", type=str, default=None)  # ISO 8601
    parser.add_argument("--version", type=str, default=None)
    args = parser.parse_args()

    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    # Resolve time window
    start_ts, end_ts, mode_used, window_days_for_log = resolve_time_range(cfg, args)

    print("=" * 80)
    print("STEP 1: DATA LOADING (DUAL STRATEGY)")
    print("=" * 80)
    source = (cfg.get("data") or {}).get("source", "database").lower()
    lookback_days = (cfg.get("data") or {}).get("lookback_days", 14)
    chunk_size = int((cfg.get("data") or {}).get("chunk_size", 100000))

    if mode_used == "window":
        logger.info("Enforce TRAIN_WINDOW_DAYS=14 (or CLI override). window_days=%s", window_days_for_log)

    logger.info("DataLoader initialized:")
    logger.info("  mode = %s", "DB" if source == "database" else "CSV")
    logger.info("  lookback_days = %s | chunk_size = %s", lookback_days, f"{chunk_size:,}")

    # Load interactions + entities
    if source == "database":
        db_url = (cfg.get("database") or cfg.get("data", {}).get("database") or {}).get("url")
        tables_cfg = (cfg.get("database") or {}).get("tables", {})
        print("\n📦 Loading RAW data (for Embeddings & CF)...")
        logger.info("\n" + "=" * 70 + "\nLOADING TRAINING BUNDLE\n" + "=" * 70)
        logger.info("Source: DB; Window: %s -> %s", start_ts, end_ts)
        interactions_df, data_dict, n_inter, min_ts, max_ts = load_bundle_from_db(
            db_url=db_url, tables_cfg=tables_cfg, start_ts=start_ts, end_ts=end_ts, chunk_size=chunk_size
        )
    else:
        data_dir = (cfg.get("data") or {}).get("dir", "dataset")
        print("\n📦 Loading RAW data (for Embeddings & CF)...")
        logger.info("\n" + "=" * 70 + "\nLOADING TRAINING BUNDLE\n" + "=" * 70)
        logger.info("Source: CSV; Window: %s -> %s", start_ts, end_ts)
        interactions_df, data_dict, n_inter, min_ts, max_ts = load_bundle_from_csv(
            data_dir=data_dir, start_ts=start_ts, end_ts=end_ts
        )

    logger.info("✅ Interactions: %s | users=%s | posts=%s",
                f"{len(interactions_df):,}",
                f"{len(data_dict.get('users', [])):,}" if isinstance(data_dict.get("users"), pd.DataFrame) else 0,
                f"{len(data_dict.get('posts', [])):,}" if isinstance(data_dict.get("posts"), pd.DataFrame) else 0)
    if not interactions_df.empty:
        logger.info("   Range: %s -> %s", min_ts, max_ts)

    # Update watermark
    update_watermark_after_load(cfg, max_ts)

    # Simple stats
    print("\nComputing Statistics.")
    print("Computing user stats.")
    user_stats, author_stats, following_dict = build_stats_and_following(interactions_df, data_dict)
    print(f"User stats for {len(user_stats)} users")
    print("Computing author stats.")
    print(f"Author stats for {len(author_stats)} authors")
    print("Building following dictionary.")
    print(f"Following dict for {len(following_dict)} users")

    print("✅ Raw data loaded:")
    print(f"   Users: {len(data_dict.get('users', [])) if isinstance(data_dict.get('users'), pd.DataFrame) else 0:,}")
    print(f"   Posts: {len(data_dict.get('posts', [])) if isinstance(data_dict.get('posts'), pd.DataFrame) else 0:,}")
    print(f"   Interactions: {len(interactions_df):,}")
    print(f"   Friendships: {len(data_dict.get('friendships', [])) if isinstance(data_dict.get('friendships'), pd.DataFrame) else 0:,}")

    print("\n📦 Loading WEIGHTED data (for Ranking Model)...")

    # Time split
    tr_cfg = cfg.get("training", {}) or {}
    val_days = int(tr_cfg.get("val_days", (cfg.get("data", {}).get("train_test_split", {}).get("val_days", 3))))
    test_days = int(tr_cfg.get("test_days", (cfg.get("data", {}).get("train_test_split", {}).get("test_days", 3))))

    train_int, val_int, test_int = split_by_time(interactions_df, val_days=val_days, test_days=test_days)

    # Initialize trainer
    trainer = ModelTrainer(cfg)

    # Prepare feature datasets
    # Embeddings ở offline có thể build ở chỗ khác; nếu không có sẽ default cosine=0
    embeddings = {}

    # Prepare train
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    if not train_int.empty:
        train_df = trainer.prepare_training_data(
            interactions_df=train_int,
            data=data_dict,
            user_stats=user_stats,
            author_stats=author_stats,
            following_dict=following_dict,
            embeddings=embeddings,
        )
    if not val_int.empty:
        val_df = trainer.prepare_training_data(
            interactions_df=val_int,
            data=data_dict,
            user_stats=user_stats,
            author_stats=author_stats,
            following_dict=following_dict,
            embeddings=embeddings,
        )
    if not test_int.empty:
        test_df = trainer.prepare_training_data(
            interactions_df=test_int,
            data=data_dict,
            user_stats=user_stats,
            author_stats=author_stats,
            following_dict=following_dict,
            embeddings=embeddings,
        )

    # ============================== TRAIN ======================================
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING")
    print("=" * 80)

    model, scaler, feature_cols = trainer.train(train_df=train_df, val_df=val_df)

    # ============================= EVALUATE ====================================
    metrics = {}
    try:
        metrics = trainer.evaluate(test_df=test_df)
    except Exception as e:
        logger.warning(f"Evaluate failed: {e}")

    # ============================== SAVE =======================================
    version_name = args.version or _make_version_name()
    models_dir = Path((cfg.get("artifacts") or {}).get("base_dir", "models"))
    version_dir = models_dir / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save ranker artifacts with unified naming (online loader expects this)
    out_base = str(version_dir / "ranker")
    trainer.save_model(output_path_base=out_base)

    # Save meta
    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "mode": mode_used,
        "window_days": window_days_for_log if mode_used == "window" else None,
        "val_days": val_days,
        "test_days": test_days,
        "counts": {
            "interactions_total": int(len(interactions_df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "users": int(len(data_dict.get("users"))) if isinstance(data_dict.get("users"), pd.DataFrame) else 0,
            "posts": int(len(data_dict.get("posts"))) if isinstance(data_dict.get("posts"), pd.DataFrame) else 0,
        },
        "features": feature_cols,
        "metrics": metrics,
    }
    _write_meta(version_dir, meta)

    # Update latest pointer (Windows-safe)
    write_latest_pointer(str(models_dir), version_name)

    print(f"\n✅ Artifacts saved under: {version_dir}")
    print(f"   -> latest.version -> {version_name}")
    print("\n🎉 Done!\n")


if __name__ == "__main__":
    main()
