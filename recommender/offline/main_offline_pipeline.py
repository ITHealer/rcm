# # """
# # MAIN OFFLINE TRAINING PIPELINE WITH TIME DECAY
# # ===============================================
# # Complete end-to-end offline training

# # Architecture:
# # - Raw interactions → Embeddings & CF Model (NO time decay)
# # - Weighted interactions → Ranking Model Training (WITH time decay)

# # Run: python scripts/offline/main_offline_pipeline.py
# # """

# # import os
# # import sys
# # import yaml
# # from datetime import datetime
# # from pathlib import Path

# # # Add project root to path
# # sys.path.append(str(Path(__file__).parent.parent.parent))

# # from recommender.common.data_loading import DataLoader
# # from recommender.common.data_loader import load_data, compute_statistics
# # from recommender.offline.embedding_generator import EmbeddingGenerator
# # from recommender.offline.cf_builder import CFBuilder
# # from recommender.offline.model_trainer import ModelTrainer
# # from recommender.offline.artifact_manager import ArtifactManager


# # def load_config(config_path='scripts/offline/config_offline.yaml'):
# #     """Load configuration"""
# #     with open(config_path, 'r') as f:
# #         return yaml.safe_load(f)


# # def main():
# #     """
# #     Complete offline training pipeline
# #     """
# #     print("=" * 80)
# #     print("🚀 OFFLINE TRAINING PIPELINE (WITH TIME DECAY)")
# #     print("=" * 80)
# #     print(f"Started at: {datetime.now()}")
    
# #     # ═════════════════════════════════════════════════════════════════════
# #     # STEP 1: LOAD DATA - DUAL LOADING STRATEGY
# #     # ═════════════════════════════════════════════════════════════════════
    
# #     print("\n" + "="*80)
# #     print("STEP 1: DATA LOADING (DUAL STRATEGY)")
# #     print("="*80)
    
# #     config = load_config()
    
# #     # ─────────────────────────────────────────────────────────────────────
# #     # 1A. Load RAW data (NO time decay) - For Embeddings & CF
# #     # ─────────────────────────────────────────────────────────────────────
    
# #     print("\n📦 Loading RAW data (for Embeddings & CF)...")
    
# #     data = load_data(config['data']['dir'])
    
# #     # Compute statistics
# #     user_stats, author_stats, following_dict = compute_statistics(data)
    
# #     print(f"✅ Raw data loaded:")
# #     print(f"   Users: {len(data['user']):,}")
# #     print(f"   Posts: {len(data['post']):,}")
# #     print(f"   Interactions: {len(data['postreaction']):,}")
# #     print(f"   Friendships: {len(data['friendship']):,}")
    
# #     # ─────────────────────────────────────────────────────────────────────
# #     # 1B. Load WEIGHTED data (WITH time decay) - For Ranking Model Training
# #     # ─────────────────────────────────────────────────────────────────────
    
# #     print("\n📦 Loading WEIGHTED data (for Ranking Model)...")
    
# #     data_loader_config = {
# #         'lookback_days': config['data']['lookback_days'],
# #         'half_life_days': config.get('time_decay', {}).get('half_life_days', 7.0),
# #         'min_weight': config.get('time_decay', {}).get('min_weight', 0.01),
# #         'chunk_size': config.get('data', {}).get('chunk_size', 100000)
# #     }
    
# #     data_loader = DataLoader(
# #         db_connection=None,  # Use CSV for now
# #         config=data_loader_config,
# #         data_dir=config['data']['dir']
# #     )
    
# #     # Load interactions with time decay applied
# #     interactions_weighted = data_loader.load_and_prepare_training_data(use_csv=True)
    
# #     print(f"✅ Weighted data loaded: {len(interactions_weighted):,} interactions")
# #     print(f"   Weight range: [{interactions_weighted['weight'].min():.4f}, {interactions_weighted['weight'].max():.4f}]")
# #     print(f"   Mean weight: {interactions_weighted['weight'].mean():.4f}")
    
# #     # Create temporal splits for weighted data
# #     train_interactions, val_interactions, test_interactions = data_loader.create_train_test_split(
# #         interactions_weighted,
# #         test_days=config['data']['train_test_split']['test_days'],
# #         val_days=config['data']['train_test_split']['val_days']
# #     )
    
# #     print(f"\n📊 Temporal splits:")
# #     print(f"   Train: {len(train_interactions):,} ({len(train_interactions)/len(interactions_weighted)*100:.1f}%)")
# #     print(f"   Val: {len(val_interactions):,} ({len(val_interactions)/len(interactions_weighted)*100:.1f}%)")
# #     print(f"   Test: {len(test_interactions):,} ({len(test_interactions)/len(interactions_weighted)*100:.1f}%)")
    
# #     # ═════════════════════════════════════════════════════════════════════
# #     # STEP 2: GENERATE EMBEDDINGS (using RAW data)
# #     # ═════════════════════════════════════════════════════════════════════
    
# #     print("\n" + "="*80)
# #     print("STEP 2: EMBEDDING GENERATION (using RAW interactions)")
# #     print("="*80)
    
# #     embedding_gen = EmbeddingGenerator(config)
    
# #     # Use RAW interactions (no time decay) for embeddings
# #     post_embeddings = embedding_gen.generate_post_embeddings(data['post'])
# #     user_embeddings = embedding_gen.generate_user_embeddings(
# #         data['postreaction'],  # ← RAW data, not weighted
# #         post_embeddings
# #     )
    
# #     embeddings = {
# #         'post': post_embeddings,
# #         'user': user_embeddings
# #     }
    
# #     print(f"✅ Embeddings generated:")
# #     print(f"   Posts: {len(post_embeddings):,}")
# #     print(f"   Users: {len(user_embeddings):,}")
    
# #     # ═════════════════════════════════════════════════════════════════════
# #     # STEP 3: BUILD COLLABORATIVE FILTERING (using RAW data)
# #     # ═════════════════════════════════════════════════════════════════════
    
# #     print("\n" + "="*80)
# #     print("STEP 3: COLLABORATIVE FILTERING (using RAW interactions)")
# #     print("="*80)
    
# #     cf_builder = CFBuilder(config)
    
# #     # Use RAW interactions (no time decay) for CF
# #     cf_model = cf_builder.build_cf_model(data['postreaction'])  # ← RAW data
    
# #     print(f"✅ CF model built:")
# #     print(f"   Users: {len(cf_model['user_ids']):,}")
# #     print(f"   Posts: {len(cf_model['post_ids']):,}")
# #     print(f"   User similarities computed: {len(cf_model['user_similarities']):,}")
# #     print(f"   Item similarities computed: {len(cf_model['item_similarities']):,}")
    
# #     # ═════════════════════════════════════════════════════════════════════
# #     # STEP 4: TRAIN RANKING MODEL (using WEIGHTED data)
# #     # ═════════════════════════════════════════════════════════════════════
    
# #     print("\n" + "="*80)
# #     print("STEP 4: RANKING MODEL TRAINING (using WEIGHTED interactions)")
# #     print("="*80)
    
# #     trainer = ModelTrainer(config)
    
# #     # Prepare training data using WEIGHTED interactions
# #     print("\n🔧 Preparing training data with time decay weights...")
    
# #     train_df = trainer.prepare_training_data(
# #         train_interactions,  # ← WEIGHTED data with 'weight' column
# #         data, 
# #         user_stats, 
# #         author_stats, 
# #         following_dict, 
# #         embeddings
# #     )
    
# #     val_df = trainer.prepare_training_data(
# #         val_interactions,  # ← WEIGHTED data
# #         data, 
# #         user_stats, 
# #         author_stats, 
# #         following_dict, 
# #         embeddings
# #     )
    
# #     test_df = trainer.prepare_training_data(
# #         test_interactions,  # ← WEIGHTED data
# #         data, 
# #         user_stats, 
# #         author_stats, 
# #         following_dict, 
# #         embeddings
# #     )
    
# #     print(f"\n✅ Training data prepared:")
# #     print(f"   Train samples: {len(train_df):,}")
# #     print(f"   Val samples: {len(val_df):,}")
# #     print(f"   Test samples: {len(test_df):,}")
    
# #     # Check if 'weight' column exists in training data
# #     if 'weight' in train_df.columns:
# #         print(f"\n📊 Time decay weights in training data:")
# #         print(f"   Mean: {train_df['weight'].mean():.4f}")
# #         print(f"   Median: {train_df['weight'].median():.4f}")
# #         print(f"   Min: {train_df['weight'].min():.4f}")
# #         print(f"   Max: {train_df['weight'].max():.4f}")
    
# #     # Train
# #     print("\n🔧 Training LightGBM ranking model...")
# #     ranking_model, ranking_scaler, ranking_feature_cols = trainer.train(train_df, val_df)
    
# #     # Evaluate
# #     print("\n📊 Evaluating on test set...")
# #     test_metrics = trainer.evaluate(test_df)
    
# #     # ═════════════════════════════════════════════════════════════════════
# #     # STEP 5: SAVE ARTIFACTS
# #     # ═════════════════════════════════════════════════════════════════════
    
# #     print("\n" + "="*80)
# #     print("STEP 5: SAVING ARTIFACTS")
# #     print("="*80)
    
# #     artifact_mgr = ArtifactManager(config['output']['models_dir'])
    
# #     version = artifact_mgr.create_version()
    
# #     # Prepare metadata
# #     metadata = {
# #         'test_metrics': test_metrics,
# #         'n_train_samples': len(train_df),
# #         'n_val_samples': len(val_df),
# #         'n_test_samples': len(test_df),
# #         'config': config,
# #         'time_decay': {
# #             'enabled': True,
# #             'half_life_days': data_loader_config['half_life_days'],
# #             'min_weight': data_loader_config['min_weight'],
# #             'weight_stats': {
# #                 'mean': float(interactions_weighted['weight'].mean()),
# #                 'median': float(interactions_weighted['weight'].median()),
# #                 'min': float(interactions_weighted['weight'].min()),
# #                 'max': float(interactions_weighted['weight'].max())
# #             }
# #         },
# #         'data_stats': {
# #             'n_users': len(data['user']),
# #             'n_posts': len(data['post']),
# #             'n_raw_interactions': len(data['postreaction']),
# #             'n_weighted_interactions': len(interactions_weighted),
# #             'n_embeddings_post': len(post_embeddings),
# #             'n_embeddings_user': len(user_embeddings),
# #             'n_cf_users': len(cf_model['user_ids']),
# #             'n_cf_posts': len(cf_model['post_ids'])
# #         }
# #     }
    
# #     artifact_mgr.save_artifacts(
# #         version=version,
# #         embeddings=embeddings,
# #         cf_model=cf_model,
# #         ranking_model=ranking_model,
# #         ranking_scaler=ranking_scaler,
# #         ranking_feature_cols=ranking_feature_cols,
# #         user_stats=user_stats,
# #         author_stats=author_stats,
# #         following_dict=following_dict,
# #         metadata=metadata
# #     )
    
# #     # Cleanup old versions
# #     artifact_mgr.cleanup_old_versions(
# #         keep_n=config['output']['keep_last_n_versions']
# #     )
    
# #     # ═════════════════════════════════════════════════════════════════════
# #     # COMPLETE
# #     # ═════════════════════════════════════════════════════════════════════
    
# #     print("\n" + "="*80)
# #     print("✅ OFFLINE TRAINING COMPLETE!")
# #     print("="*80)
# #     print(f"Version: {version}")
# #     print(f"\n📊 Final Metrics:")
# #     print(f"   Test AUC: {test_metrics['auc']:.4f}")
# #     print(f"   Test Precision@10: {test_metrics['precision@10']:.4f}")
# #     print(f"   Test Precision@20: {test_metrics['precision@20']:.4f}")
# #     print(f"   Test Precision@50: {test_metrics['precision@50']:.4f}")
# #     print(f"\n💾 Artifacts saved to: models/{version}/")
# #     print(f"   - Embeddings (RAW data)")
# #     print(f"   - CF Model (RAW data)")
# #     print(f"   - Ranking Model (WEIGHTED data)")
# #     print(f"   - Statistics & Metadata")
# #     print(f"\n🕐 Completed at: {datetime.now()}")
# #     print("="*80)


# # if __name__ == "__main__":
# #     main()


# # scripts/offline/main_offline_pipeline.py
# from __future__ import annotations

# import json
# from datetime import datetime, timedelta, timezone
# from pathlib import Path
# from typing import Optional, Dict, Any

# import pandas as pd
# import yaml
# from sqlalchemy import create_engine

# from recommender.common.data_loading import DataLoader
# from recommender.offline.training_state import TrainingStateManager
# from recommender.offline.model_trainer import train_lightgbm
# from recommender.offline.artifact_manager import save_artifacts


# # ----------------------------- Config loader ---------------------------------
# def load_yaml_if_exists(path: str) -> Dict[str, Any]:
#     p = Path(path)
#     if not p.exists():
#         return {}
#     with p.open("r", encoding="utf-8") as f:
#         return yaml.safe_load(f) or {}


# def now_utc() -> datetime:
#     return datetime.now(timezone.utc)


# def create_db_engine_from_cfg(cfg: Dict[str, Any]):
#     db_url = (
#         cfg.get("database", {}) or {}
#     ).get("url", "")
#     if not db_url:
#         return None
#     # SQLAlchemy engine (sync)
#     return create_engine(
#         db_url,
#         pool_pre_ping=True,
#         pool_size=int(cfg.get("database", {}).get("pool_size", 5)),
#         max_overflow=int(cfg.get("database", {}).get("max_overflow", 5)),
#         future=True,
#     )


# def get_training_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
#     # Defaults
#     return {
#         "lookback_days": int(cfg.get("training", {}).get("window_days", 14)),
#         "half_life_days": int(cfg.get("training", {}).get("half_life_days", 7)),
#         "pretrain_full_export": bool(cfg.get("training", {}).get("pretrain_full_export", True)),
#         "overlap_days": int(cfg.get("training", {}).get("overlap_days", 2)),
#         "chunk_size": int(cfg.get("training", {}).get("chunk_size", 200_000)),
#     }


# def get_artifacts_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
#     return {
#         "base_dir": cfg.get("artifacts", {}).get("base_dir", "models"),
#         "state_file": cfg.get("artifacts", {}).get("state_file", "models/training_state.json"),
#     }


# # ----------------------------- Main pipeline ---------------------------------
# def main():
#     # 1) Load config (non-blocking if missing)
#     cfg = {}
#     cfg.update(load_yaml_if_exists("configs/config_offline.yaml"))
#     # allow override from configs/config.yaml if you want
#     cfg_base = load_yaml_if_exists("configs/config.yaml")
#     for k, v in cfg_base.items():
#         cfg.setdefault(k, v)

#     train_cfg = get_training_cfg(cfg)
#     artifacts_cfg = get_artifacts_cfg(cfg)

#     # 2) DB engine (optional)
#     engine = create_db_engine_from_cfg(cfg)

#     # 3) Training state
#     state_mgr = TrainingStateManager(artifacts_cfg["state_file"])
#     state = state_mgr.load()

#     # 4) DataLoader init
#     dl_config = {
#         "lookback_days": train_cfg["lookback_days"],
#         "half_life_days": train_cfg["half_life_days"],
#         "chunk_size": train_cfg["chunk_size"],
#         # DB tables/columns — adjust if your BE schema differs
#         "tables": {"interactions": (cfg.get("database", {}).get("tables", {}) or {}).get("interactions", "interactions")},
#         "columns": {
#             "user_id": (cfg.get("database", {}).get("columns", {}) or {}).get("user_id", "user_id"),
#             "post_id": (cfg.get("database", {}).get("columns", {}) or {}).get("post_id", "post_id"),
#             "action": (cfg.get("database", {}).get("columns", {}) or {}).get("action", "action"),
#             "created_at": (cfg.get("database", {}).get("columns", {}) or {}).get("created_at", "created_at"),
#         },
#         # CSV fallback directory
#         "csv_dir": (cfg.get("dataset", {}) or {}).get("dir", "dataset"),
#         "csv_reaction_file": (cfg.get("dataset", {}) or {}).get("reactions", "PostReaction.csv"),
#         "csv_user_col": (cfg.get("dataset", {}) or {}).get("user_col", "UserId"),
#         "csv_post_col": (cfg.get("dataset", {}) or {}).get("post_col", "PostId"),
#         "csv_time_col": (cfg.get("dataset", {}) or {}).get("time_col", "CreateDate"),
#         "csv_reaction_type_col": (cfg.get("dataset", {}) or {}).get("reaction_type_col", "ReactionTypeId"),
#     }
#     loader = DataLoader(db_connection=engine, config=dl_config, data_dir=dl_config["csv_dir"])

#     # 5) Choose window: pretrain or incremental
#     use_csv = engine is None  # if no DB url -> use CSV fallback
#     now = now_utc()
#     if state.last_train_end is None:
#         # PRETRAIN
#         if train_cfg["pretrain_full_export"] and not use_csv:
#             interactions = loader.load_initial_training_data(full_export=True, use_csv=False)
#         else:
#             interactions = loader.load_training_data(
#                 lookback_days=train_cfg["lookback_days"],
#                 use_csv=use_csv,
#                 since=None,
#                 until=now,
#             )
#     else:
#         # INCREMENTAL
#         last_end = pd.to_datetime(state.last_train_end, utc=True)
#         interactions = loader.load_incremental_training_data(
#             last_train_end=last_end,
#             overlap_days=train_cfg["overlap_days"],
#             until=now,
#             use_csv=use_csv,
#         )

#     # 6) Feature engineering (use your project’s function)
#     #    build_training_matrices must return: X (DataFrame), y (Series), interactions_df (DataFrame), meta (dict)
#     from recommender.common.feature_engineer import build_training_matrices  # keep your existing code
#     X, y, interactions_df, meta = build_training_matrices(
#         interactions=interactions,
#         now_ts=now,
#     )

#     # 7) Train model with time-decay sample weights
#     params = {
#         "objective": "binary",
#         "metric": "auc",
#         "learning_rate": 0.05,
#         "num_leaves": 64,
#         "feature_fraction": 0.8,
#         "bagging_fraction": 0.8,
#         "bagging_freq": 1,
#         "verbose": -1,
#     }
#     result = train_lightgbm(
#         X=X,
#         y=y,
#         interactions_df=interactions_df,
#         params=params,
#         half_life_days=train_cfg["half_life_days"],
#         ref_time=now,
#         created_at_col="created_at",
#         base_weight_col=None,  # set to e.g. 'base_weight' if you precompute action multipliers
#     )

#     # 8) Save artifacts (model + meta) with version
#     version_name = now.strftime("v%Y%m%d_%H%M%S")
#     meta = meta or {}
#     meta.update(
#         {
#             "version": version_name,
#             "trained_at": now.isoformat(),
#             "lookback_days": train_cfg["lookback_days"],
#             "half_life_days": train_cfg["half_life_days"],
#             "rows": len(X),
#         }
#     )
#     if result.feature_importances_ is not None:
#         meta["feature_importances_top10"] = (
#             result.feature_importances_.head(10).to_dict(orient="records")
#         )

#     save_artifacts(
#         version_name=version_name,
#         model=result.model,
#         meta=meta,
#         artifacts_base_dir=get_artifacts_cfg(cfg)["base_dir"],
#     )

#     # 9) Update training state
#     state.last_train_end = TrainingStateManager.now_iso()
#     state_mgr.save(state)

#     print(f"[OK] Offline pipeline finished. Version: {version_name}")


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
