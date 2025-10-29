

# recommender/common/feature_engineer.py
"""
Feature Engineering (row-by-row + vector)
=========================================

- Bảng: users(Id, CreateDate), posts(Id, UserId, CreateDate, IsRepost, IsPin),
        friendships(UserId, FriendId), post_hashtags(PostId, HashtagId)
- interactions chuẩn: user_id, post_id, action, created_at
- Label: Positive = {like, love, laugh, wow, sad, angry, care, comment, share, save}
         Negative = {view}
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd

POSITIVE_ACTIONS = {"like", "love", "laugh", "wow", "sad", "angry", "care", "comment", "share", "save"}
NEGATIVE_ACTIONS = {"view"}

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _ensure_datetime_utc(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, utc=True, errors="coerce")
    else:
        if series.dt.tz is None:
            series = series.dt.tz_localize("UTC")
        else:
            series = series.dt.tz_convert("UTC")
    return series

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

@dataclass
class FeatureEngineer:
    data: Dict[str, pd.DataFrame] | None
    user_stats: Dict | None
    author_stats: Dict | None
    following: Dict[int, set] | None
    embeddings: Dict[str, Dict[int, np.ndarray]] | None

    def __post_init__(self):
        self.users = (self.data or {}).get("users")
        self.posts = (self.data or {}).get("posts")
        self.friendships = (self.data or {}).get("friendships")
        self.post_hashtags = (self.data or {}).get("post_hashtags")

        self.post_author: Dict[int, int] = {}
        if isinstance(self.posts, pd.DataFrame) and {"Id", "UserId"}.issubset(self.posts.columns):
            self.post_author = dict(zip(self.posts["Id"].astype(int), self.posts["UserId"].astype(int)))

        self.post_is_repost: Dict[int, int] = {}
        self.post_is_pin: Dict[int, int] = {}
        if isinstance(self.posts, pd.DataFrame) and "Id" in self.posts.columns:
            ids = self.posts["Id"].astype(int)

            # Lấy series nếu có, nếu không tạo series mặc định độ dài đúng bằng số dòng
            if "IsRepost" in self.posts.columns:
                is_repost_series = pd.to_numeric(self.posts["IsRepost"], errors="coerce").fillna(0)
            else:
                is_repost_series = pd.Series(0, index=self.posts.index)

            if "IsPin" in self.posts.columns:
                is_pin_series = pd.to_numeric(self.posts["IsPin"], errors="coerce").fillna(0)
            else:
                is_pin_series = pd.Series(0, index=self.posts.index)

            # Ép kiểu int (0/1) và build dict
            self.post_is_repost = dict(zip(ids, is_repost_series.astype(int)))
            self.post_is_pin = dict(zip(ids, is_pin_series.astype(int)))
        else:
            self.post_is_repost = {}
            self.post_is_pin = {}
            
        self.post_created = {}
        if isinstance(self.posts, pd.DataFrame) and "CreateDate" in self.posts.columns:
            tmp = self.posts[["Id", "CreateDate"]].copy()
            tmp["CreateDate"] = _ensure_datetime_utc(tmp["CreateDate"])
            self.post_created = dict(zip(tmp["Id"].astype(int), tmp["CreateDate"]))

        self.post_hashtag_count: Dict[int, int] = {}
        if isinstance(self.post_hashtags, pd.DataFrame) and "PostId" in self.post_hashtags.columns:
            cnt = self.post_hashtags.groupby("PostId").size().rename("cnt")
            self.post_hashtag_count = cnt.to_dict()

        self.user_created: Dict[int, pd.Timestamp] = {}
        if isinstance(self.users, pd.DataFrame) and {"Id", "CreateDate"}.issubset(self.users.columns):
            tmp = self.users[["Id", "CreateDate"]].copy()
            tmp["CreateDate"] = _ensure_datetime_utc(tmp["CreateDate"])
            self.user_created = dict(zip(tmp["Id"].astype(int), tmp["CreateDate"]))

        self.author_follower_count: Dict[int, int] = {}
        if isinstance(self.friendships, pd.DataFrame) and {"UserId", "FriendId"}.issubset(self.friendships.columns):
            cnt = self.friendships.groupby("FriendId").size()
            self.author_follower_count = cnt.to_dict()

        self.post_vecs = (self.embeddings or {}).get("post")
        self.user_vecs = (self.embeddings or {}).get("user")

    def extract_features(self, user_id: int, post_id: int) -> Dict[str, float]:
        feats: Dict[str, float] = {}

        # user stats
        ust = (self.user_stats or {}).get(int(user_id), {})
        feats["user_total_interactions"] = float(ust.get("total_interactions", 0.0))
        feats["user_positive_rate"] = float(ust.get("positive_rate", 0.0))

        acc_age = 0.0
        if int(user_id) in self.user_created:
            acc_age = max((_now_utc() - self.user_created[int(user_id)]).total_seconds() / 86400.0, 0.0)
        feats["user_account_age_days"] = acc_age

        feats["post_total_interactions"] = 0.0
        feats["post_positive_rate"] = 0.0

        feats["post_is_repost"] = float(self.post_is_repost.get(int(post_id), 0))
        feats["post_is_pin"] = float(self.post_is_pin.get(int(post_id), 0))
        feats["post_hashtag_count"] = float(self.post_hashtag_count.get(int(post_id), 0))

        age_h = 0.0
        if int(post_id) in self.post_created:
            age_h = max((_now_utc() - self.post_created[int(post_id)]).total_seconds() / 3600.0, 0.0)
        feats["post_age_hours"] = age_h

        author_id = self.post_author.get(int(post_id), -1)
        feats["author_id_known"] = 1.0 if author_id != -1 else 0.0

        ast = (self.author_stats or {}).get(int(author_id), {}) if author_id != -1 else {}
        feats["author_total_interactions"] = float(ast.get("total_interactions", 0.0))
        feats["author_positive_rate"] = float(ast.get("positive_rate", 0.0))
        feats["author_follower_count"] = float(self.author_follower_count.get(int(author_id), 0))

        foll = (self.following or {}).get(int(user_id), set())
        feats["is_following_author"] = 1.0 if author_id in foll else 0.0

        sim = 0.0
        if self.post_vecs is not None:
            pv = self.post_vecs.get(int(post_id))
            uv = self.user_vecs.get(int(user_id)) if self.user_vecs else None
            if pv is not None and uv is not None:
                na, nb = np.linalg.norm(pv), np.linalg.norm(uv)
                sim = float(np.dot(pv, uv) / (na * nb)) if (na > 0 and nb > 0) else 0.0
        feats["user_post_cosine"] = sim

        return feats

# ---------------------- Vector pipeline (batch FE) ---------------------------

def build_training_matrices(
    interactions: pd.DataFrame,
    users_df: Optional[pd.DataFrame] = None,
    posts_df: Optional[pd.DataFrame] = None,
    friendships_df: Optional[pd.DataFrame] = None,
    post_hashtags_df: Optional[pd.DataFrame] = None,
    embeddings: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    now_ts: Optional[datetime] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    """
    Build training matrices with enhanced features
    """
    if interactions is None or interactions.empty:
        raise ValueError("interactions is empty")

    df = interactions.copy()
    if "created_at" in df.columns:
        df["created_at"] = _ensure_datetime_utc(df["created_at"])

    if "label" not in df.columns:
        df["label"] = df["action"].isin(POSITIVE_ACTIONS).astype(int)

    # User stats
    u_cnt = df.groupby("user_id").size().rename("user_total_interactions")
    u_pos = df.groupby("user_id")["label"].mean().rename("user_positive_rate")
    u = pd.concat([u_cnt, u_pos], axis=1).reset_index()
    df = df.merge(u, on="user_id", how="left")

    df["user_account_age_days"] = 0.0
    if isinstance(users_df, pd.DataFrame) and {"Id", "CreateDate"}.issubset(users_df.columns):
        uu = users_df[["Id", "CreateDate"]].rename(columns={"Id": "user_id", "CreateDate": "user_created_at"}).copy()
        uu["user_created_at"] = _ensure_datetime_utc(uu["user_created_at"])
        base = now_ts or _now_utc()
        df = df.merge(uu, on="user_id", how="left")
        age_days = (base - df["user_created_at"]).dt.total_seconds() / 86400.0
        df["user_account_age_days"] = age_days.fillna(0.0).clip(lower=0.0)
        df.drop(columns=["user_created_at"], inplace=True)

    # Post stats
    p_cnt = df.groupby("post_id").size().rename("post_total_interactions")
    p_pos = df.groupby("post_id")["label"].mean().rename("post_positive_rate")
    p = pd.concat([p_cnt, p_pos], axis=1).reset_index()
    df = df.merge(p, on="post_id", how="left")

    # Posts meta
    df["post_age_hours"] = 0.0
    df["post_is_repost"] = 0.0
    df["post_is_pin"] = 0.0
    df["author_id_known"] = 0.0
    
    if isinstance(posts_df, pd.DataFrame) and {"Id", "UserId", "CreateDate"}.issubset(posts_df.columns):
        pp = posts_df[["Id", "UserId", "CreateDate"]].copy()
        
        if "IsRepost" in posts_df.columns:
            pp["IsRepost"] = posts_df["IsRepost"]
        else:
            pp["IsRepost"] = 0
            
        if "IsPin" in posts_df.columns:
            pp["IsPin"] = posts_df["IsPin"]
        else:
            pp["IsPin"] = 0
        
        pp.rename(columns={
            "Id": "post_id",
            "UserId": "author_id_from_posts",
            "CreateDate": "post_created_at"
        }, inplace=True)
        
        pp["post_created_at"] = _ensure_datetime_utc(pp["post_created_at"])
        base = now_ts or _now_utc()
        
        df = df.merge(pp, on="post_id", how="left")
        
        if "author_id" not in df.columns:
            df["author_id"] = df["author_id_from_posts"]
        else:
            df["author_id"] = df["author_id"].fillna(df["author_id_from_posts"])
        
        df.drop(columns=["author_id_from_posts"], inplace=True, errors='ignore')
        
        df["author_id_known"] = df["author_id"].notnull().astype(float)
        age_h = (base - df["post_created_at"]).dt.total_seconds() / 3600.0
        df["post_age_hours"] = age_h.fillna(0.0).clip(lower=0.0)
        df["post_is_repost"] = df["IsRepost"].fillna(0).astype(float)
        df["post_is_pin"] = df["IsPin"].fillna(0).astype(float)
        df.drop(columns=["post_created_at", "IsRepost", "IsPin"], inplace=True)
    else:
        if "author_id" in df.columns:
            df["author_id_known"] = df["author_id"].notnull().astype(float)

    # Hashtag count
    if isinstance(post_hashtags_df, pd.DataFrame) and "PostId" in post_hashtags_df.columns:
        hcnt = (
            post_hashtags_df.groupby("PostId")
            .size()
            .rename("ph_cnt")
            .reset_index()
            .rename(columns={"PostId": "post_id"})
        )
        df = df.merge(hcnt, on="post_id", how="left")
        df["post_hashtag_count"] = df["ph_cnt"].fillna(0.0)
        df.drop(columns=["ph_cnt"], inplace=True)
    else:
        df["post_hashtag_count"] = 0.0

    # Author stats
    df["author_total_interactions"] = 0.0
    df["author_positive_rate"] = 0.0
    df["author_follower_count"] = 0.0
    df["is_following_author"] = 0.0

    if "author_id" in df.columns:
        a_stats = (
            df[["author_id", "label"]]
            .dropna()
            .groupby("author_id")["label"]
            .agg(["count", "mean"])
            .rename(columns={"count": "author_total_interactions", "mean": "author_positive_rate"})
            .reset_index()
        )
        df = df.merge(a_stats, on="author_id", how="left", suffixes=("", "_a"))
        for col in ["author_total_interactions", "author_positive_rate"]:
            if f"{col}_a" in df.columns:
                df[col] = df[f"{col}_a"].fillna(df[col])
                df.drop(columns=[f"{col}_a"], inplace=True)

        if isinstance(friendships_df, pd.DataFrame) and {"UserId", "FriendId"}.issubset(friendships_df.columns):
            fcnt = (
                friendships_df.groupby("FriendId")
                .size()
                .rename("author_follower_count")
                .reset_index()
                .rename(columns={"FriendId": "author_id"})
            )
            df = df.merge(fcnt, on="author_id", how="left", suffixes=("", "_fc"))
            if "author_follower_count_fc" in df.columns:
                df["author_follower_count"] = df["author_follower_count_fc"].fillna(df["author_follower_count"])
                df.drop(columns=["author_follower_count_fc"], inplace=True)

            tmp = friendships_df.rename(columns={"UserId": "user_id", "FriendId": "author_id"})[["user_id", "author_id"]].copy()
            tmp["is_following_author"] = 1.0
            df = df.merge(tmp, on=["user_id", "author_id"], how="left", suffixes=("", "_if"))
            if "is_following_author_if" in df.columns:
                df["is_following_author"] = df["is_following_author_if"].fillna(df["is_following_author"])
                df.drop(columns=["is_following_author_if"], inplace=True)

    df["author_total_interactions"] = df["author_total_interactions"].fillna(0.0)
    df["author_positive_rate"] = df["author_positive_rate"].fillna(0.0)
    df["author_follower_count"] = df["author_follower_count"].fillna(0.0)
    df["is_following_author"] = df["is_following_author"].fillna(0.0)

    # Embeddings similarity
    df["user_post_cosine"] = 0.0
    if embeddings and isinstance(embeddings.get("post"), dict):
        post_vecs = embeddings["post"]
        user_vecs = embeddings.get("user")
        def _to_vec(ids: pd.Series, vec_dict: Dict[int, np.ndarray]):
            return [vec_dict.get(int(x)) if pd.notnull(x) else None for x in ids]
        pvec = _to_vec(df["post_id"], post_vecs)
        uvec = _to_vec(df["user_id"], user_vecs) if user_vecs else [None] * len(df)
        sims = []
        for pv, uv in zip(pvec, uvec):
            if pv is None or uv is None:
                sims.append(0.0); continue
            na, nb = np.linalg.norm(pv), np.linalg.norm(uv)
            sims.append(float(np.dot(pv, uv) / (na * nb)) if (na > 0 and nb > 0) else 0.0)
        df["user_post_cosine"] = sims

    # ✅ NEW: Advanced features
    df['user_post_ratio'] = df['post_total_interactions'] / (df['user_total_interactions'] + 1e-6)
    df['author_user_ratio'] = df['author_total_interactions'] / (df['user_total_interactions'] + 1e-6)
    
    df['post_age_days'] = df['post_age_hours'] / 24.0
    df['post_is_fresh'] = (df['post_age_hours'] < 24).astype(float)
    df['post_is_hot'] = (df['post_age_hours'] < 6).astype(float)
    
    df['post_engagement_rate'] = df['post_positive_rate'] * df['post_total_interactions']
    df['author_engagement_rate'] = df['author_positive_rate'] * df['author_total_interactions']
    
    df['following_x_engagement'] = df['is_following_author'] * df['post_engagement_rate']
    df['similarity_x_engagement'] = df['user_post_cosine'] * df['post_engagement_rate']
    
    df['user_is_active'] = (df['user_total_interactions'] >= 10).astype(float)
    df['user_is_super_active'] = (df['user_total_interactions'] >= 50).astype(float)
    
    df['post_is_popular'] = (df['post_total_interactions'] >= 10).astype(float)
    df['post_is_viral'] = (df['post_total_interactions'] >= 50).astype(float)

    # Feature columns list
    feature_cols = [
        "user_total_interactions", "user_positive_rate", "user_account_age_days",
        "post_total_interactions", "post_positive_rate", "post_age_hours",
        "post_is_repost", "post_is_pin", "post_hashtag_count",
        "author_id_known", "author_total_interactions", "author_positive_rate",
        "author_follower_count", "is_following_author",
        "user_post_cosine",
        "user_post_ratio", "author_user_ratio",
        "post_age_days", "post_is_fresh", "post_is_hot",
        "post_engagement_rate", "author_engagement_rate",
        "following_x_engagement", "similarity_x_engagement",
        "user_is_active", "user_is_super_active",
        "post_is_popular", "post_is_viral"
    ]
    
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feature_cols].fillna(0.0)
    y = df["label"].astype(int)

    interactions_df = df[["user_id", "post_id", "action", "created_at"]].copy()

    meta = {
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "features": feature_cols,
        "positive_rate": float(y.mean()) if len(y) else 0.0,
    }
    return X, y, interactions_df, meta
