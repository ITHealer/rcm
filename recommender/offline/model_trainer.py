
# # recommender/offline/model_trainer.py
# """
# Ranking Model Training with LightGBM (time-decay + evaluate)
# ------------------------------------------------------------

# - POSITIVE_ACTIONS cập nhật: like, love, laugh, wow, sad, angry, care, comment, share, save
# - Tự tính sample_weight nếu chưa có: weight = time_decay(age_days, half_life) * action_multiplier[action]
#   * time_decay = 0.5 ** (age_days / half_life_days)
#   * action_multipliers (mặc định):
#       view: 0.5
#       like: 1.0, love: 1.3, care: 1.25, laugh: 1.2, wow: 1.1, sad: 0.9, angry: 0.9
#       comment: 1.5, share: 2.0, save: 1.2
# - Lưu ý: để online loader đọc được, khi gọi save_model hãy truyền:
#     output_path_base = "models/<version>/ranker"
#   sẽ sinh ra:
#     models/<version>/ranker_model.txt
#     models/<version>/ranker_scaler.pkl
#     models/<version>/ranker_feature_cols.pkl
# """

# from __future__ import annotations

# import pickle
# from dataclasses import dataclass
# from datetime import datetime, timezone
# from typing import Dict, Tuple, Optional, Any, List

# import numpy as np
# import pandas as pd
# import lightgbm as lgb
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score

# from recommender.common.feature_engineer import FeatureEngineer

# # >>> updated positive actions <<<
# POSITIVE_ACTIONS = {"like", "love", "laugh", "wow", "sad", "angry", "care", "comment", "share", "save"}

# def _ensure_datetime_utc_any(v) -> Optional[pd.Timestamp]:
#     if v is None or (isinstance(v, float) and np.isnan(v)):
#         return None
#     try:
#         ts = pd.to_datetime(v, utc=True, errors="coerce")
#         return ts
#     except Exception:
#         return None

# def _ensure_datetime_series_utc(series: pd.Series) -> pd.Series:
#     if not pd.api.types.is_datetime64_any_dtype(series):
#         series = pd.to_datetime(series, utc=True, errors="coerce")
#     else:
#         if series.dt.tz is None:
#             series = series.dt.tz_localize("UTC")
#         else:
#             series = series.dt.tz_convert("UTC")
#     return series

# def _now_utc() -> datetime:
#     return datetime.now(timezone.utc)


# class ModelTrainer:
#     """
#     Train ranking model với LightGBM (time-decay weights + evaluation)
#     """

#     def __init__(self, config: dict):
#         self.config = config or {}

#         # ---- model params ----
#         model_cfg = (
#             self.config.get("ranking_model", {})
#             or self.config.get("model", {})
#             or {}
#         )
#         self.model_params = model_cfg.get("params", {}) or {
#             "boosting_type": "gbdt",
#             "objective": "binary",
#             "metric": "auc",
#             "learning_rate": 0.05,
#             "num_leaves": 64,
#             "feature_fraction": 0.8,
#             "bagging_fraction": 0.8,
#             "bagging_freq": 1,
#             "max_depth": -1,
#             "verbose": -1,
#             "n_jobs": -1,
#             "seed": 42,
#         }
#         self.num_boost_round = int(model_cfg.get("num_boost_round", 500))
#         self.early_stopping_rounds = int(model_cfg.get("early_stopping_rounds", 50))
#         self.log_every_n = int(model_cfg.get("log_every_n", 50))

#         # ---- time-decay & multipliers ----
#         td_cfg = self.config.get("time_decay", {}) or {}
#         tr_cfg = self.config.get("training", {}) or {}

#         # ưu tiên time_decay.half_life_days, fallback training.half_life_days
#         self.half_life_days = float(td_cfg.get("half_life_days", tr_cfg.get("half_life_days", 7)))
#         self.min_weight = float(td_cfg.get("min_weight", tr_cfg.get("min_weight", 1e-3)))
#         self.use_action_multiplier = bool(tr_cfg.get("use_action_multiplier", True))
#         self.action_multipliers = (
#             td_cfg.get("action_multipliers")
#             or tr_cfg.get("action_multipliers")
#             or {
#                 "view": 0.5,
#                 "like": 1.0, "love": 1.3, "care": 1.25, "laugh": 1.2, "wow": 1.1, "sad": 0.9, "angry": 0.9,
#                 "comment": 1.5, "share": 2.0, "save": 1.2,
#             }
#         )

#         self.model: Optional[lgb.Booster] = None
#         self.scaler: Optional[StandardScaler] = None
#         self.feature_cols: Optional[List[str]] = None

#     # ----------------------------------------------------------------------
#     # 1) PREPARE TRAINING DATA (row-by-row FE cho tương thích code cũ)
#     # ----------------------------------------------------------------------
#     def prepare_training_data(
#         self,
#         interactions_df: pd.DataFrame,
#         data: Dict[str, pd.DataFrame],
#         user_stats: Dict,
#         author_stats: Dict,
#         following_dict: Dict,
#         embeddings: Dict,
#     ) -> pd.DataFrame:
#         print("\n🔧 Preparing Training Data...")

#         df = interactions_df.copy()
#         df.rename(
#             columns={
#                 "UserId": "user_id",
#                 "PostId": "post_id",
#                 "ReactionTypeId": "reaction_type_id",
#                 "CreateDate": "created_at",
#             },
#             inplace=True,
#         )

#         has_weights = "weight" in df.columns
#         if has_weights:
#             print("   ✅ Detected 'weight' column")
#         else:
#             print("   ℹ️  No 'weight' found. Will compute in train() using time-decay + multipliers.")

#         if "action" not in df.columns:
#             raise ValueError("interactions_df must include 'action' after joining ReactionType/PostView/Comment")

#         # Split positives/negatives theo POSITIVE_ACTIONS
#         positive = df[df["action"].isin(POSITIVE_ACTIONS)].copy()
#         negative = df[~df["action"].isin(POSITIVE_ACTIONS)].copy()

#         print(f"   Raw positive samples: {len(positive):,}")
#         print(f"   Raw negative samples: {len(negative):,}")
#         if len(positive) == 0:
#             raise ValueError("No positive samples found! Cannot train model.")

#         # Balance negatives (tối đa 5x positives)
#         n_neg = min(len(positive) * 5, len(negative))
#         negative = negative.sample(n=n_neg, random_state=42) if n_neg > 0 else negative
#         print(f"   Using negative samples: {len(negative):,}")

#         fe = FeatureEngineer(
#             data=data,
#             user_stats=user_stats,
#             author_stats=author_stats,
#             following=following_dict,
#             embeddings=embeddings,
#         )

#         rows = []
#         failed = 0

#         print("   Extracting features for positive samples...")
#         for _, row in positive.iterrows():
#             try:
#                 feats = fe.extract_features(int(row["user_id"]), int(row["post_id"]))
#                 feats["label"] = 1
#                 # giữ lại action/created_at để tính weight ở train()
#                 feats["action"] = str(row.get("action", "like"))
#                 feats["created_at"] = _ensure_datetime_utc_any(row.get("created_at"))
#                 if has_weights and "weight" in row:
#                     feats["weight"] = float(row["weight"])
#                 rows.append(feats)
#             except Exception:
#                 failed += 1

#         print("   Extracting features for negative samples...")
#         for _, row in negative.iterrows():
#             try:
#                 feats = fe.extract_features(int(row["user_id"]), int(row["post_id"]))
#                 feats["label"] = 0
#                 feats["action"] = str(row.get("action", "view"))
#                 feats["created_at"] = _ensure_datetime_utc_any(row.get("created_at"))
#                 if has_weights and "weight" in row:
#                     feats["weight"] = float(row["weight"])
#                 rows.append(feats)
#             except Exception:
#                 failed += 1

#         if failed:
#             print(f"   ⚠️  Feature extraction failed for {failed} samples")

#         training_df = pd.DataFrame(rows)
#         if training_df.empty:
#             raise ValueError("No training samples after feature extraction!")

#         label_counts = training_df["label"].value_counts()
#         print("\n   📊 Final label distribution:")
#         print(f"      Positive(1): {label_counts.get(1, 0):,}")
#         print(f"      Negative(0): {label_counts.get(0, 0):,}")

#         if "weight" in training_df.columns:
#             w = training_df["weight"]
#             print(
#                 f"   Weights: mean={w.mean():.4f}, median={w.median():.4f}, "
#                 f"min={w.min():.4f}, max={w.max():.4f}"
#             )

#         # Shuffle
#         training_df = training_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
#         print(f"\n✅ Training data prepared: {len(training_df):,} samples")
#         return training_df

#     # ----------------------------------------------------------------------
#     # helper: compute time-decay * action multiplier weights
#     # ----------------------------------------------------------------------
#     def _compute_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
#         base = _now_utc()
#         created = df.get("created_at")
#         actions = df.get("action")

#         if created is None or actions is None:
#             # fallback uniform weights
#             return np.ones(len(df), dtype=np.float32)

#         # đảm bảo datetime UTC
#         if not pd.api.types.is_datetime64_any_dtype(created):
#             created = pd.to_datetime(created, utc=True, errors="coerce")
#         else:
#             if created.dt.tz is None:
#                 created = created.dt.tz_localize("UTC")
#             else:
#                 created = created.dt.tz_convert("UTC")

#         age_days = (base - created).dt.total_seconds() / 86400.0
#         age_days = age_days.fillna(age_days.max() if len(age_days) else 0.0).clip(lower=0.0)

#         # time-decay
#         decay = np.power(0.5, age_days / max(self.half_life_days, 1e-6))

#         # multipliers
#         if self.use_action_multiplier:
#             mult = actions.map(lambda a: self.action_multipliers.get(str(a).lower(), 1.0)).astype(float)
#         else:
#             mult = 1.0

#         w = np.maximum(decay * mult, self.min_weight).astype(np.float32)
#         return w

#     # ----------------------------------------------------------------------
#     # 2) TRAIN (tự tính sample_weight nếu chưa có)
#     # ----------------------------------------------------------------------
#     def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[lgb.Booster, StandardScaler, list]:
#         print("\n🔧 Training Ranking Model...")

#         has_weights = "weight" in train_df.columns and "weight" in val_df.columns
#         if has_weights:
#             print("   ✅ Using sample weights from DataFrame")
#         else:
#             print("   ℹ️  No weights column. Will compute time-decay * multipliers automatically.")

#         # loại cột không phải feature
#         exclude_cols = ["label", "action", "created_at"]
#         if has_weights:
#             exclude_cols.append("weight")

#         self.feature_cols = [c for c in train_df.columns if c not in exclude_cols]
#         X_train = train_df[self.feature_cols].fillna(0)
#         y_train = train_df["label"]

#         X_val = val_df[self.feature_cols].fillna(0)
#         y_val = val_df["label"]

#         # weights
#         if has_weights:
#             train_w = train_df["weight"].values.astype(np.float32)
#             val_w = val_df["weight"].values.astype(np.float32)
#         else:
#             train_w = self._compute_sample_weights(train_df)
#             val_w = self._compute_sample_weights(val_df)

#         print(f"   Train: {X_train.shape} | Val: {X_val.shape} | #Features: {len(self.feature_cols)}")
#         print(f"   Train weights: mean={train_w.mean():.4f} min={train_w.min():.4f} max={train_w.max():.4f}")

#         print("   Scaling features...")
#         self.scaler = StandardScaler()
#         X_train_sc = self.scaler.fit_transform(X_train)
#         X_val_sc = self.scaler.transform(X_val)

#         dtrain = lgb.Dataset(X_train_sc, label=y_train, weight=train_w)
#         dval = lgb.Dataset(X_val_sc, label=y_val, weight=val_w, reference=dtrain)

#         print("   Training model...")
#         self.model = lgb.train(
#             self.model_params,
#             dtrain,
#             num_boost_round=self.num_boost_round,
#             valid_sets=[dtrain, dval],
#             valid_names=["train", "val"],
#             callbacks=[
#                 lgb.early_stopping(self.early_stopping_rounds, verbose=True),
#                 lgb.log_evaluation(self.log_every_n),
#             ],
#         )

#         print("✅ Training complete!")
#         if hasattr(self.model, "best_iteration") and self.model.best_iteration:
#             print(f"   Best iteration: {self.model.best_iteration}")
#             best_auc = self.model.best_score.get("val", {}).get("auc", None)
#             if best_auc is not None:
#                 print(f"   Best val AUC : {best_auc:.4f}")

#         return self.model, self.scaler, self.feature_cols

#     # ----------------------------------------------------------------------
#     # 3) EVALUATE
#     # ----------------------------------------------------------------------
#     def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
#         print("\n📊 Evaluating Model...")
#         if test_df is None or len(test_df) == 0:
#             print("⚠️  No test data available")
#             return {k: 0.0 for k in ["auc", "logloss", "precision", "recall", "precision@10", "precision@20", "precision@50"]}

#         label_counts = test_df["label"].value_counts()
#         print(f"   Test label distribution: {label_counts.to_dict()}")
#         if len(label_counts) < 2:
#             print("⚠️  Single-class test set; metrics not meaningful.")
#             return {k: 0.0 for k in ["auc", "logloss", "precision", "recall", "precision@10", "precision@20", "precision@50"]}

#         X_test = test_df[self.feature_cols].fillna(0)
#         y_test = test_df["label"]
#         X_test_sc = self.scaler.transform(X_test)
#         y_prob = self.model.predict(X_test_sc)
#         y_pred = (y_prob >= 0.5).astype(int)

#         metrics: Dict[str, float] = {}
#         try:
#             metrics["auc"] = roc_auc_score(y_test, y_prob)
#         except Exception:
#             metrics["auc"] = 0.0

#         try:
#             y_prob_clip = np.clip(y_prob, 1e-7, 1 - 1e-7)
#             metrics["logloss"] = log_loss(y_test, y_prob_clip)
#         except Exception:
#             metrics["logloss"] = 999.0

#         try:
#             metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)
#             metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)
#         except Exception:
#             metrics["precision"] = 0.0
#             metrics["recall"] = 0.0

#         for k in [10, 20, 50]:
#             try:
#                 if len(y_test) >= k:
#                     top_k_idx = np.argsort(y_prob)[-k:]
#                     metrics[f"precision@{k}"] = float(y_test.iloc[top_k_idx].sum() / k)
#                 else:
#                     metrics[f"precision@{k}"] = 0.0
#             except Exception:
#                 metrics[f"precision@{k}"] = 0.0

#         print("\n📈 Test Set Performance:")
#         print(f"   AUC          : {metrics['auc']:.4f}")
#         print(f"   LogLoss      : {metrics['logloss']:.4f}")
#         print(f"   Precision    : {metrics['precision']:.4f}")
#         print(f"   Recall       : {metrics['recall']:.4f}")
#         print(f"   Precision@10 : {metrics['precision@10']:.4f}")
#         print(f"   Precision@20 : {metrics['precision@20']:.4f}")
#         print(f"   Precision@50 : {metrics['precision@50']:.4f}")

#         return metrics

#     # ----------------------------------------------------------------------
#     # 4) SAVE
#     # ----------------------------------------------------------------------
#     def save_model(self, output_path_base: str) -> None:
#         """
#         output_path_base = "models/<version>/ranker"
#           -> ranker_model.txt / ranker_scaler.pkl / ranker_feature_cols.pkl
#         """
#         self.model.save_model(output_path_base + "_model.txt")
#         with open(output_path_base + "_scaler.pkl", "wb") as f:
#             pickle.dump(self.scaler, f)
#         with open(output_path_base + "_feature_cols.pkl", "wb") as f:
#             pickle.dump(self.feature_cols, f)
#         print(f"✅ Model saved to: {output_path_base}_*")


## version cập nhật 27/10 - Claude
"""
MODEL TRAINER - LIGHTGBM RANKING
=================================
Train LightGBM ranking model for post recommendations

Features:
- Multi-task learning (like, comment, share predictions)
- Time decay weighted training
- Feature importance analysis
- Train/val/test split with proper validation
- Model evaluation metrics (AUC, Precision@K, NDCG)

Usage:
    trainer = ModelTrainer()
    model, scaler, metrics = trainer.train(
        train_df, val_df, test_df, 
        feature_cols, target_col
    )
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time

# Setup logging
logger = logging.getLogger(__name__)

# Try import lightgbm
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.error("❌ lightgbm not available. Install: pip install lightgbm")

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score


class ModelTrainer:
    """
    Train LightGBM ranking model
    """
    
    def __init__(
        self,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
        verbose: int = 100
    ):
        """
        Initialize Model Trainer
        
        Args:
            learning_rate: Learning rate for boosting
            num_leaves: Maximum number of leaves per tree
            max_depth: Maximum depth of tree (-1 for unlimited)
            n_estimators: Number of boosting rounds
            early_stopping_rounds: Stop if no improvement after N rounds
            verbose: Print progress every N rounds
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm required. Install: pip install lightgbm")
        
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        
        logger.info(f"ModelTrainer initialized:")
        logger.info(f"   learning_rate: {learning_rate}")
        logger.info(f"   num_leaves: {num_leaves}")
        logger.info(f"   n_estimators: {n_estimators}")
    
    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fit_scaler: bool = False,
        scaler: Optional[StandardScaler] = None
    ) -> Tuple[np.ndarray, StandardScaler]:
        """
        Prepare and scale features
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            fit_scaler: Whether to fit new scaler
            scaler: Existing scaler (if fit_scaler=False)
        
        Returns:
            (scaled_features, scaler)
        """
        # Extract features
        X = df[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale
        if fit_scaler:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            if scaler is None:
                raise ValueError("Must provide scaler when fit_scaler=False")
            X_scaled = scaler.transform(X)
        
        return X_scaled, scaler
    
    def compute_sample_weights(
        self,
        df: pd.DataFrame,
        weight_col: str = 'time_decay_weight'
    ) -> np.ndarray:
        """
        Compute sample weights for training
        
        Args:
            df: DataFrame with weight column
            weight_col: Name of weight column
        
        Returns:
            Sample weights array
        """
        if weight_col in df.columns:
            weights = df[weight_col].values
        else:
            logger.warning(f"Weight column '{weight_col}' not found, using uniform weights")
            weights = np.ones(len(df))
        
        # Normalize weights (optional - helps with training stability)
        weights = weights / weights.mean()
        
        return weights
    
    # ========================================================================
    # MODEL TRAINING
    # ========================================================================
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'label',
        weight_col: str = 'time_decay_weight',
        use_weights: bool = True
    ) -> Tuple[lgb.Booster, StandardScaler, Dict, List[str]]:
        """
        Train LightGBM ranking model
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            feature_cols: List of feature columns
            target_col: Target column name
            weight_col: Sample weight column name
            use_weights: Whether to use sample weights
        
        Returns:
            (model, scaler, metrics, feature_cols)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING RANKING MODEL")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        # ────────────────────────────────────────────────────────────────────
        # Prepare features
        # ────────────────────────────────────────────────────────────────────
        
        logger.info(f"\nPreparing features...")
        logger.info(f"   Feature columns: {len(feature_cols)}")
        logger.info(f"   Train samples: {len(train_df):,}")
        logger.info(f"   Val samples: {len(val_df):,}")
        logger.info(f"   Test samples: {len(test_df):,}")
        
        # Scale features
        X_train, scaler = self.prepare_features(train_df, feature_cols, fit_scaler=True)
        X_val, _ = self.prepare_features(val_df, feature_cols, fit_scaler=False, scaler=scaler)
        X_test, _ = self.prepare_features(test_df, feature_cols, fit_scaler=False, scaler=scaler)
        
        # Extract targets
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        
        # Compute sample weights
        if use_weights:
            train_weights = self.compute_sample_weights(train_df, weight_col)
            logger.info(f"   Using sample weights (mean={train_weights.mean():.3f})")
        else:
            train_weights = None
            logger.info(f"   Not using sample weights")
        
        # ────────────────────────────────────────────────────────────────────
        # Create LightGBM datasets
        # ────────────────────────────────────────────────────────────────────
        
        logger.info(f"\nCreating LightGBM datasets...")
        
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=train_weights,
            feature_name=feature_cols,
            free_raw_data=False
        )
        
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_data,
            feature_name=feature_cols,
            free_raw_data=False
        )
        
        # ────────────────────────────────────────────────────────────────────
        # Train model
        # ────────────────────────────────────────────────────────────────────
        
        logger.info(f"\nTraining LightGBM model...")
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'min_child_samples': 20,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            'seed': 42
        }
        
        logger.info(f"Model parameters:")
        for key, value in params.items():
            logger.info(f"   {key}: {value}")
        
        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(self.verbose)
            ]
        )
        
        best_iteration = model.best_iteration
        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Best iteration: {best_iteration}")
        logger.info(f"   Train AUC: {model.best_score['train']['auc']:.4f}")
        logger.info(f"   Val AUC: {model.best_score['val']['auc']:.4f}")
        
        # ────────────────────────────────────────────────────────────────────
        # Feature importance
        # ────────────────────────────────────────────────────────────────────
        
        logger.info(f"\n📊 Top 10 Feature Importance:")
        
        feature_importance = model.feature_importance(importance_type='gain')
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance_df.head(10).iterrows():
            logger.info(f"   {row['feature']:30s}: {row['importance']:.0f}")
        
        # ────────────────────────────────────────────────────────────────────
        # Evaluate on test set
        # ────────────────────────────────────────────────────────────────────
        
        logger.info(f"\n📈 Evaluating on test set...")
        
        test_metrics = self.evaluate(
            model,
            X_test,
            y_test,
            test_df,
            k_values=[10, 20, 50, 100]
        )
        
        # ────────────────────────────────────────────────────────────────────
        # Summary
        # ────────────────────────────────────────────────────────────────────
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ MODEL TRAINING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"   Total time: {elapsed_time:.1f}s")
        logger.info(f"   Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"   Test Precision@10: {test_metrics['precision@10']:.4f}")
        logger.info(f"   Test Precision@20: {test_metrics['precision@20']:.4f}")
        logger.info(f"   Test Precision@50: {test_metrics['precision@50']:.4f}")
        
        return model, scaler, test_metrics, feature_cols
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    def evaluate(
        self,
        model: lgb.Booster,
        X: np.ndarray,
        y_true: np.ndarray,
        df: pd.DataFrame,
        k_values: List[int] = [10, 20, 50, 100]
    ) -> Dict:
        """
        Evaluate model performance
        
        Metrics:
        - AUC (overall discrimination)
        - Log Loss
        - Precision@K (relevance of top K)
        - NDCG (ranking quality)
        
        Args:
            model: Trained model
            X: Features
            y_true: True labels
            df: DataFrame with user/post IDs for per-user metrics
            k_values: List of K values for Precision@K
        
        Returns:
            Dict of metrics
        """
        # Predict
        y_pred = model.predict(X, num_iteration=model.best_iteration)
        
        metrics = {}
        
        # 1. AUC-ROC
        metrics['auc'] = roc_auc_score(y_true, y_pred)
        
        # 2. Log Loss
        metrics['log_loss'] = log_loss(y_true, y_pred)
        
        # 3. Precision@K (Per-user evaluation)
        if 'user_id' in df.columns:
            for k in k_values:
                precision_k = self._compute_precision_at_k(
                    df, y_true, y_pred, k
                )
                metrics[f'precision@{k}'] = precision_k
        else:
            # Global precision@k if no user_id
            for k in k_values:
                sorted_indices = np.argsort(y_pred)[::-1]
                top_k_true = y_true[sorted_indices[:k]]
                metrics[f'precision@{k}'] = top_k_true.mean()
        
        # 4. NDCG (Normalized Discounted Cumulative Gain)
        if 'user_id' in df.columns:
            ndcg = self._compute_ndcg_per_user(df, y_true, y_pred)
            metrics['ndcg'] = ndcg
        else:
            # Simple NDCG if no user grouping
            metrics['ndcg'] = ndcg_score([y_true], [y_pred])
        
        return metrics
    
    def _compute_precision_at_k(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int
    ) -> float:
        """
        Compute per-user Precision@K
        
        For each user:
        - Rank predictions
        - Take top K
        - Compute precision (% relevant in top K)
        
        Args:
            df: DataFrame with user_id column
            y_true: True labels
            y_pred: Predicted scores
            k: K value
        
        Returns:
            Average precision@k across users
        """
        df_eval = df.copy()
        df_eval['y_true'] = y_true
        df_eval['y_pred'] = y_pred
        
        precisions = []
        
        for user_id, group in df_eval.groupby('user_id'):
            if len(group) < k:
                continue
            
            # Sort by prediction score (descending)
            group_sorted = group.sort_values('y_pred', ascending=False)
            
            # Top K
            top_k = group_sorted.head(k)
            
            # Precision = # relevant in top K / K
            precision = top_k['y_true'].mean()
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def _compute_ndcg_per_user(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute per-user NDCG
        
        Args:
            df: DataFrame with user_id column
            y_true: True labels
            y_pred: Predicted scores
        
        Returns:
            Average NDCG across users
        """
        df_eval = df.copy()
        df_eval['y_true'] = y_true
        df_eval['y_pred'] = y_pred
        
        ndcg_scores = []
        
        for user_id, group in df_eval.groupby('user_id'):
            if len(group) < 2:
                continue
            
            y_true_user = group['y_true'].values
            y_pred_user = group['y_pred'].values
            
            # NDCG requires at least one positive sample
            if y_true_user.sum() == 0:
                continue
            
            ndcg = ndcg_score([y_true_user], [y_pred_user])
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    # ========================================================================
    # SAVE / LOAD
    # ========================================================================
    
    def save_model(
        self,
        model: lgb.Booster,
        scaler: StandardScaler,
        feature_cols: List[str],
        path_prefix: str
    ):
        """
        Save model, scaler, and feature columns
        
        Args:
            model: Trained LightGBM model
            scaler: Feature scaler
            feature_cols: List of feature column names
            path_prefix: Path prefix for saving (e.g., 'models/v1/ranking')
        """
        logger.info(f"\n💾 Saving model artifacts...")
        
        path_prefix = Path(path_prefix)
        path_prefix.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = f"{path_prefix}_model.txt"
        model.save_model(model_path)
        logger.info(f"   Saved model: {model_path}")
        
        # Save scaler
        scaler_path = f"{path_prefix}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"   Saved scaler: {scaler_path}")
        
        # Save feature columns
        feature_cols_path = f"{path_prefix}_feature_cols.pkl"
        with open(feature_cols_path, 'wb') as f:
            pickle.dump(feature_cols, f)
        logger.info(f"   Saved feature_cols: {feature_cols_path}")
        
        logger.info(f"✅ Model artifacts saved!")
    
    @staticmethod
    def load_model(path_prefix: str) -> Tuple[lgb.Booster, StandardScaler, List[str]]:
        """
        Load model, scaler, and feature columns
        
        Args:
            path_prefix: Path prefix (e.g., 'models/v1/ranking')
        
        Returns:
            (model, scaler, feature_cols)
        """
        logger.info(f"Loading model artifacts from {path_prefix}...")
        
        # Load model
        model_path = f"{path_prefix}_model.txt"
        model = lgb.Booster(model_file=model_path)
        logger.info(f"   Loaded model: {model_path}")
        
        # Load scaler
        scaler_path = f"{path_prefix}_scaler.pkl"
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"   Loaded scaler: {scaler_path}")
        
        # Load feature columns
        feature_cols_path = f"{path_prefix}_feature_cols.pkl"
        with open(feature_cols_path, 'rb') as f:
            feature_cols = pickle.load(f)
        logger.info(f"   Loaded feature_cols: {feature_cols_path}")
        
        logger.info(f"✅ Model artifacts loaded!")
        
        return model, scaler, feature_cols


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

def main():
    """Test Model Trainer"""
    
    logger.info(f"{'='*70}")
    logger.info(f"MODEL TRAINER TEST")
    logger.info(f"{'='*70}")
    
    # Initialize
    trainer = ModelTrainer(
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=100,
        early_stopping_rounds=10,
        verbose=10
    )
    
    # Create sample data
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (binary)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create DataFrames
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    df = pd.DataFrame(X, columns=feature_cols)
    df['label'] = y
    df['user_id'] = np.random.randint(1, 50, n_samples)
    df['time_decay_weight'] = np.random.uniform(0.5, 1.0, n_samples)
    
    # Split
    train_df = df[:600]
    val_df = df[600:800]
    test_df = df[800:]
    
    logger.info(f"Sample data:")
    logger.info(f"   Train: {len(train_df)}")
    logger.info(f"   Val: {len(val_df)}")
    logger.info(f"   Test: {len(test_df)}")
    
    # Train
    model, scaler, metrics, feature_cols = trainer.train(
        train_df,
        val_df,
        test_df,
        feature_cols,
        target_col='label',
        weight_col='time_decay_weight',
        use_weights=True
    )
    
    # Save
    trainer.save_model(model, scaler, feature_cols, 'test_ranking')
    
    # Load
    loaded_model, loaded_scaler, loaded_feature_cols = ModelTrainer.load_model('test_ranking')
    
    logger.info(f"\n✅ TEST COMPLETE")
    logger.info(f"   Test AUC: {metrics['auc']:.4f}")
    logger.info(f"   Feature columns: {len(loaded_feature_cols)}")


if __name__ == "__main__":
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    main()