
# recommender/offline/model_trainer.py
"""
Ranking Model Training with LightGBM (time-decay + evaluate)
------------------------------------------------------------

- POSITIVE_ACTIONS cập nhật: like, love, laugh, wow, sad, angry, care, comment, share, save
- Tự tính sample_weight nếu chưa có: weight = time_decay(age_days, half_life) * action_multiplier[action]
  * time_decay = 0.5 ** (age_days / half_life_days)
  * action_multipliers (mặc định):
      view: 0.5
      like: 1.0, love: 1.3, care: 1.25, laugh: 1.2, wow: 1.1, sad: 0.9, angry: 0.9
      comment: 1.5, share: 2.0, save: 1.2
- Lưu ý: để online loader đọc được, khi gọi save_model hãy truyền:
    output_path_base = "models/<version>/ranker"
  sẽ sinh ra:
    models/<version>/ranker_model.txt
    models/<version>/ranker_scaler.pkl
    models/<version>/ranker_feature_cols.pkl
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score

from recommender.common.feature_engineer import FeatureEngineer

# >>> updated positive actions <<<
POSITIVE_ACTIONS = {"like", "love", "laugh", "wow", "sad", "angry", "care", "comment", "share", "save"}

def _ensure_datetime_utc_any(v) -> Optional[pd.Timestamp]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        ts = pd.to_datetime(v, utc=True, errors="coerce")
        return ts
    except Exception:
        return None

def _ensure_datetime_series_utc(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, utc=True, errors="coerce")
    else:
        if series.dt.tz is None:
            series = series.dt.tz_localize("UTC")
        else:
            series = series.dt.tz_convert("UTC")
    return series

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class ModelTrainer:
    """
    Train ranking model với LightGBM (time-decay weights + evaluation)
    """

    def __init__(self, config: dict):
        self.config = config or {}

        # ---- model params ----
        model_cfg = (
            self.config.get("ranking_model", {})
            or self.config.get("model", {})
            or {}
        )
        self.model_params = model_cfg.get("params", {}) or {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "max_depth": -1,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }
        self.num_boost_round = int(model_cfg.get("num_boost_round", 500))
        self.early_stopping_rounds = int(model_cfg.get("early_stopping_rounds", 50))
        self.log_every_n = int(model_cfg.get("log_every_n", 50))

        # ---- time-decay & multipliers ----
        td_cfg = self.config.get("time_decay", {}) or {}
        tr_cfg = self.config.get("training", {}) or {}

        # ưu tiên time_decay.half_life_days, fallback training.half_life_days
        self.half_life_days = float(td_cfg.get("half_life_days", tr_cfg.get("half_life_days", 7)))
        self.min_weight = float(td_cfg.get("min_weight", tr_cfg.get("min_weight", 1e-3)))
        self.use_action_multiplier = bool(tr_cfg.get("use_action_multiplier", True))
        self.action_multipliers = (
            td_cfg.get("action_multipliers")
            or tr_cfg.get("action_multipliers")
            or {
                "view": 0.5,
                "like": 1.0, "love": 1.3, "care": 1.25, "laugh": 1.2, "wow": 1.1, "sad": 0.9, "angry": 0.9,
                "comment": 1.5, "share": 2.0, "save": 1.2,
            }
        )

        self.model: Optional[lgb.Booster] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: Optional[List[str]] = None

    # ----------------------------------------------------------------------
    # 1) PREPARE TRAINING DATA (row-by-row FE cho tương thích code cũ)
    # ----------------------------------------------------------------------
    def prepare_training_data(
        self,
        interactions_df: pd.DataFrame,
        data: Dict[str, pd.DataFrame],
        user_stats: Dict,
        author_stats: Dict,
        following_dict: Dict,
        embeddings: Dict,
    ) -> pd.DataFrame:
        print("\n🔧 Preparing Training Data...")

        df = interactions_df.copy()
        df.rename(
            columns={
                "UserId": "user_id",
                "PostId": "post_id",
                "ReactionTypeId": "reaction_type_id",
                "CreateDate": "created_at",
            },
            inplace=True,
        )

        has_weights = "weight" in df.columns
        if has_weights:
            print("   ✅ Detected 'weight' column")
        else:
            print("   ℹ️  No 'weight' found. Will compute in train() using time-decay + multipliers.")

        if "action" not in df.columns:
            raise ValueError("interactions_df must include 'action' after joining ReactionType/PostView/Comment")

        # Split positives/negatives theo POSITIVE_ACTIONS
        positive = df[df["action"].isin(POSITIVE_ACTIONS)].copy()
        negative = df[~df["action"].isin(POSITIVE_ACTIONS)].copy()

        print(f"   Raw positive samples: {len(positive):,}")
        print(f"   Raw negative samples: {len(negative):,}")
        if len(positive) == 0:
            raise ValueError("No positive samples found! Cannot train model.")

        # Balance negatives (tối đa 5x positives)
        n_neg = min(len(positive) * 5, len(negative))
        negative = negative.sample(n=n_neg, random_state=42) if n_neg > 0 else negative
        print(f"   Using negative samples: {len(negative):,}")

        fe = FeatureEngineer(
            data=data,
            user_stats=user_stats,
            author_stats=author_stats,
            following=following_dict,
            embeddings=embeddings,
        )

        rows = []
        failed = 0

        print("   Extracting features for positive samples...")
        for _, row in positive.iterrows():
            try:
                feats = fe.extract_features(int(row["user_id"]), int(row["post_id"]))
                feats["label"] = 1
                # giữ lại action/created_at để tính weight ở train()
                feats["action"] = str(row.get("action", "like"))
                feats["created_at"] = _ensure_datetime_utc_any(row.get("created_at"))
                if has_weights and "weight" in row:
                    feats["weight"] = float(row["weight"])
                rows.append(feats)
            except Exception:
                failed += 1

        print("   Extracting features for negative samples...")
        for _, row in negative.iterrows():
            try:
                feats = fe.extract_features(int(row["user_id"]), int(row["post_id"]))
                feats["label"] = 0
                feats["action"] = str(row.get("action", "view"))
                feats["created_at"] = _ensure_datetime_utc_any(row.get("created_at"))
                if has_weights and "weight" in row:
                    feats["weight"] = float(row["weight"])
                rows.append(feats)
            except Exception:
                failed += 1

        if failed:
            print(f"   ⚠️  Feature extraction failed for {failed} samples")

        training_df = pd.DataFrame(rows)
        if training_df.empty:
            raise ValueError("No training samples after feature extraction!")

        label_counts = training_df["label"].value_counts()
        print("\n   📊 Final label distribution:")
        print(f"      Positive(1): {label_counts.get(1, 0):,}")
        print(f"      Negative(0): {label_counts.get(0, 0):,}")

        if "weight" in training_df.columns:
            w = training_df["weight"]
            print(
                f"   Weights: mean={w.mean():.4f}, median={w.median():.4f}, "
                f"min={w.min():.4f}, max={w.max():.4f}"
            )

        # Shuffle
        training_df = training_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"\n✅ Training data prepared: {len(training_df):,} samples")
        return training_df

    # ----------------------------------------------------------------------
    # helper: compute time-decay * action multiplier weights
    # ----------------------------------------------------------------------
    def _compute_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        base = _now_utc()
        created = df.get("created_at")
        actions = df.get("action")

        if created is None or actions is None:
            # fallback uniform weights
            return np.ones(len(df), dtype=np.float32)

        # đảm bảo datetime UTC
        if not pd.api.types.is_datetime64_any_dtype(created):
            created = pd.to_datetime(created, utc=True, errors="coerce")
        else:
            if created.dt.tz is None:
                created = created.dt.tz_localize("UTC")
            else:
                created = created.dt.tz_convert("UTC")

        age_days = (base - created).dt.total_seconds() / 86400.0
        age_days = age_days.fillna(age_days.max() if len(age_days) else 0.0).clip(lower=0.0)

        # time-decay
        decay = np.power(0.5, age_days / max(self.half_life_days, 1e-6))

        # multipliers
        if self.use_action_multiplier:
            mult = actions.map(lambda a: self.action_multipliers.get(str(a).lower(), 1.0)).astype(float)
        else:
            mult = 1.0

        w = np.maximum(decay * mult, self.min_weight).astype(np.float32)
        return w

    # ----------------------------------------------------------------------
    # 2) TRAIN (tự tính sample_weight nếu chưa có)
    # ----------------------------------------------------------------------
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[lgb.Booster, StandardScaler, list]:
        print("\n🔧 Training Ranking Model...")

        has_weights = "weight" in train_df.columns and "weight" in val_df.columns
        if has_weights:
            print("   ✅ Using sample weights from DataFrame")
        else:
            print("   ℹ️  No weights column. Will compute time-decay * multipliers automatically.")

        # loại cột không phải feature
        exclude_cols = ["label", "action", "created_at"]
        if has_weights:
            exclude_cols.append("weight")

        self.feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        X_train = train_df[self.feature_cols].fillna(0)
        y_train = train_df["label"]

        X_val = val_df[self.feature_cols].fillna(0)
        y_val = val_df["label"]

        # weights
        if has_weights:
            train_w = train_df["weight"].values.astype(np.float32)
            val_w = val_df["weight"].values.astype(np.float32)
        else:
            train_w = self._compute_sample_weights(train_df)
            val_w = self._compute_sample_weights(val_df)

        print(f"   Train: {X_train.shape} | Val: {X_val.shape} | #Features: {len(self.feature_cols)}")
        print(f"   Train weights: mean={train_w.mean():.4f} min={train_w.min():.4f} max={train_w.max():.4f}")

        print("   Scaling features...")
        self.scaler = StandardScaler()
        X_train_sc = self.scaler.fit_transform(X_train)
        X_val_sc = self.scaler.transform(X_val)

        dtrain = lgb.Dataset(X_train_sc, label=y_train, weight=train_w)
        dval = lgb.Dataset(X_val_sc, label=y_val, weight=val_w, reference=dtrain)

        print("   Training model...")
        self.model = lgb.train(
            self.model_params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(self.log_every_n),
            ],
        )

        print("✅ Training complete!")
        if hasattr(self.model, "best_iteration") and self.model.best_iteration:
            print(f"   Best iteration: {self.model.best_iteration}")
            best_auc = self.model.best_score.get("val", {}).get("auc", None)
            if best_auc is not None:
                print(f"   Best val AUC : {best_auc:.4f}")

        return self.model, self.scaler, self.feature_cols

    # ----------------------------------------------------------------------
    # 3) EVALUATE
    # ----------------------------------------------------------------------
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        print("\n📊 Evaluating Model...")
        if test_df is None or len(test_df) == 0:
            print("⚠️  No test data available")
            return {k: 0.0 for k in ["auc", "logloss", "precision", "recall", "precision@10", "precision@20", "precision@50"]}

        label_counts = test_df["label"].value_counts()
        print(f"   Test label distribution: {label_counts.to_dict()}")
        if len(label_counts) < 2:
            print("⚠️  Single-class test set; metrics not meaningful.")
            return {k: 0.0 for k in ["auc", "logloss", "precision", "recall", "precision@10", "precision@20", "precision@50"]}

        X_test = test_df[self.feature_cols].fillna(0)
        y_test = test_df["label"]
        X_test_sc = self.scaler.transform(X_test)
        y_prob = self.model.predict(X_test_sc)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics: Dict[str, float] = {}
        try:
            metrics["auc"] = roc_auc_score(y_test, y_prob)
        except Exception:
            metrics["auc"] = 0.0

        try:
            y_prob_clip = np.clip(y_prob, 1e-7, 1 - 1e-7)
            metrics["logloss"] = log_loss(y_test, y_prob_clip)
        except Exception:
            metrics["logloss"] = 999.0

        try:
            metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)
            metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)
        except Exception:
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0

        for k in [10, 20, 50]:
            try:
                if len(y_test) >= k:
                    top_k_idx = np.argsort(y_prob)[-k:]
                    metrics[f"precision@{k}"] = float(y_test.iloc[top_k_idx].sum() / k)
                else:
                    metrics[f"precision@{k}"] = 0.0
            except Exception:
                metrics[f"precision@{k}"] = 0.0

        print("\n📈 Test Set Performance:")
        print(f"   AUC          : {metrics['auc']:.4f}")
        print(f"   LogLoss      : {metrics['logloss']:.4f}")
        print(f"   Precision    : {metrics['precision']:.4f}")
        print(f"   Recall       : {metrics['recall']:.4f}")
        print(f"   Precision@10 : {metrics['precision@10']:.4f}")
        print(f"   Precision@20 : {metrics['precision@20']:.4f}")
        print(f"   Precision@50 : {metrics['precision@50']:.4f}")

        return metrics

    # ----------------------------------------------------------------------
    # 4) SAVE
    # ----------------------------------------------------------------------
    def save_model(self, output_path_base: str) -> None:
        """
        output_path_base = "models/<version>/ranker"
          -> ranker_model.txt / ranker_scaler.pkl / ranker_feature_cols.pkl
        """
        self.model.save_model(output_path_base + "_model.txt")
        with open(output_path_base + "_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(output_path_base + "_feature_cols.pkl", "wb") as f:
            pickle.dump(self.feature_cols, f)
        print(f"✅ Model saved to: {output_path_base}_*")
